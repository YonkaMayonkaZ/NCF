#!/usr/bin/env python3
"""
Question 2: Effect of MLP layers on NeuMF performance with and without pretraining.
Optimized version using project utilities.

FIXES APPLIED:
1. Fixed subprocess call typo: cstdout -> stdout
2. Removed Unicode emoji characters to avoid ASCII encoding errors
3. Maintained compatibility with Python 3.6 and existing utils
"""

import os
import sys
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Use project utilities
from src.utils.config import config
from src.utils.io import ensure_dir, save_json, load_json
from src.utils.logging import get_experiment_logger
from src.utils.visualization import plot_comparative_metrics, save_run_history

class Question2Experiment:
    def __init__(self, num_runs=10, epochs=20):
        """Initialize Q2 experiment with proper logging and config."""
        self.num_runs = num_runs
        self.epochs = epochs
        self.results = {}
        self.factor_num = config.factor_num  # Use config
        
        # Initialize logger
        self.logger = get_experiment_logger("question_2_experiment")
        
        # Ensure output directories exist using utility
        ensure_dir(config.figure_dir)
        ensure_dir(config.model_dir)
        ensure_dir(config.output_dir / "reports")
        
        self._log_experiment_info()
    
    def _log_experiment_info(self):
        """Log experiment configuration."""
        self.logger.info("="*60)
        self.logger.info("QUESTION 2: MLP LAYERS EFFECT ON NeuMF PERFORMANCE")
        self.logger.info("="*60)
        self.logger.info("Configuration:")
        self.logger.info(f"- Number of runs per setup: {self.num_runs}")
        self.logger.info(f"- Epochs per run: {self.epochs}")
        self.logger.info(f"- Layer configurations: 1, 2, 3")
        self.logger.info(f"- Factor number: {self.factor_num}")
        self.logger.info(f"- Training modes: With pretraining, Without pretraining")
        self.logger.info(f"- Model directory: {config.model_dir}")
        self.logger.info(f"- Results directory: {config.output_dir}")
        self.logger.info("="*60)
    
    def find_model(self, model_type, num_layers=None):
        """Find a model file based on naming convention using config paths."""
        if model_type == "GMF":
            pattern = f"GMF_{self.factor_num}f_best.pth"
        elif model_type == "MLP":
            pattern = f"MLP_{num_layers}l_{self.factor_num}f_best.pth"
        else:
            return None
        
        model_path = config.model_dir / pattern
        return model_path if model_path.exists() else None
    
    def ensure_pretrained_models_exist(self):
        """Ensure GMF and MLP models exist for pretraining."""
        self.logger.info("Checking for pretrained models...")
        
        # Check GMF
        gmf_path = self.find_model("GMF")
        if not gmf_path:
            self.logger.info("GMF model not found. Training GMF...")
            if not self._train_pretrain_model("GMF"):
                raise RuntimeError("Failed to train GMF model")
            self.logger.info("GMF model training completed")
        else:
            self.logger.info(f"GMF model found: {gmf_path.name}")
        
        # Check MLP models for each layer configuration
        for layers in [1, 2, 3]:
            mlp_path = self.find_model("MLP", layers)
            if not mlp_path:
                self.logger.info(f"MLP {layers}-layer model not found. Training...")
                if not self._train_pretrain_model("MLP", layers):
                    raise RuntimeError(f"Failed to train MLP {layers}-layer model")
                self.logger.info(f"MLP {layers}-layer model training completed")
            else:
                self.logger.info(f"MLP {layers}-layer model found: {mlp_path.name}")
        
        self.logger.info("All required pretrained models are available!")
    
    def _train_pretrain_model(self, model_type, num_layers=None):
        """Train a pretrained model using the pretrain script."""
        cmd = [
            "python", "scripts/pretrain.py",
            "--model", model_type,
            "--epochs", str(self.epochs),
            "--factor_num", str(self.factor_num)
        ]
        
        if model_type == "MLP" and num_layers:
            cmd.extend(["--num_layers", str(num_layers)])
        
        try:
            self.logger.info(f"Training {model_type} with command: {' '.join(cmd)}")
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                      universal_newlines=True, check=True)
            self.logger.info(f"{model_type} training completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to train {model_type}: {e}")
            self.logger.error(f"STDOUT: {e.stdout}")
            self.logger.error(f"STDERR: {e.stderr}")
            return False
    
    def run_single_experiment(self, num_layers, pretraining, run_id):
        """Run a single training experiment with enhanced error handling."""
        config_name = f"{num_layers}layers_{'pretrain' if pretraining else 'scratch'}"
        self.logger.info(f"Run {run_id}/{self.num_runs} - {config_name}")
        
        # Check if required pretrained models exist for pretraining
        if pretraining:
            gmf_path = self.find_model("GMF")
            mlp_path = self.find_model("MLP", num_layers)
            if not gmf_path or not mlp_path:
                self.logger.error(f"Required pretrained models not found")
                self.logger.error(f"GMF: {gmf_path}")
                self.logger.error(f"MLP: {mlp_path}")
                return None
        
        # Prepare command for training NeuMF using config
        cmd = [
            "python", "scripts/train_neumf.py",
            "--epochs", str(self.epochs),
            "--num_layers", str(num_layers),
            "--factor_num", str(self.factor_num),
            "--save"  # Ensure models are saved
        ]
        
        if pretraining:
            cmd.extend(["--model", "NeuMF-pre", "--pretraining"])
        else:
            cmd.extend(["--model", "NeuMF-end"])
        
        return self._execute_training_command(cmd, config_name, run_id)
    
    def _execute_training_command(self, cmd, config_name, run_id):
        """Execute training command and parse results."""
        try:
            self.logger.info(f"Executing: {' '.join(cmd)}")
            # FIXED: Changed cstdout to stdout
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                      universal_newlines=True, check=True)
            
            # Parse the output to extract results
            output_lines = result.stdout.split('\n')
            parsed_result = self._parse_training_output(output_lines, config_name, run_id)
            
            if parsed_result:
                # FIXED: Removed Unicode emoji characters
                self.logger.info(f"[OK] {config_name} Run {run_id}: HR@10={parsed_result['hr']:.4f}, "
                               f"NDCG@10={parsed_result['ndcg']:.4f}, Params={parsed_result['parameters']:,}")
                return parsed_result
            else:
                # FIXED: Removed Unicode emoji characters
                self.logger.error(f"[ERROR] {config_name} Run {run_id}: Failed to parse results")
                return None
                
        except subprocess.CalledProcessError as e:
            # FIXED: Removed Unicode emoji characters
            self.logger.error(f"[ERROR] {config_name} Run {run_id}: Training failed - {e}")
            if e.stdout:
                self.logger.error(f"STDOUT: {e.stdout[-500:]}")  # Last 500 chars
            if e.stderr:
                self.logger.error(f"STDERR: {e.stderr[-500:]}")  # Last 500 chars
            return None
        except Exception as e:
            # FIXED: Removed Unicode emoji characters
            self.logger.error(f"[ERROR] {config_name} Run {run_id}: Unexpected error - {e}")
            return None
    
    def _parse_training_output(self, output_lines, config_name, run_id):
        """Parse training output with multiple fallback strategies."""
        hr_value = None
        ndcg_value = None
        param_count = None
        
        # Strategy 1: Look for the results section (new format)
        in_results = False
        for line in output_lines:
            if "--- RESULTS ---" in line:
                in_results = True
                continue
            elif "--- END RESULTS ---" in line:
                in_results = False
                break
            elif in_results and ":" in line:
                try:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == f"HR@{config.top_k}":
                        hr_value = float(value)
                    elif key == f"NDCG@{config.top_k}":
                        ndcg_value = float(value)
                    elif key == "Parameters":
                        param_count = int(value)
                except ValueError:
                    continue
        
        # Strategy 2: Look for "Best Result:" line if new format not found
        if hr_value is None:
            for line in output_lines:
                if "Best Result:" in line and "HR=" in line and "NDCG=" in line:
                    try:
                        # Parse: "Best Result: Epoch 015: HR=0.6834, NDCG=0.4123"
                        hr_part = line.split("HR=")[1].split(",")[0].strip()
                        ndcg_part = line.split("NDCG=")[1].strip()
                        hr_value = float(hr_part)
                        ndcg_value = float(ndcg_part)
                        break
                    except (IndexError, ValueError):
                        continue
        
        # Strategy 3: Look for parameter count in various formats
        if param_count is None:
            for line in output_lines:
                if any(keyword in line.lower() for keyword in ["parameters:", "params:", "parameter count"]):
                    try:
                        # Extract number from line
                        import re
                        numbers = re.findall(r'[\d,]+', line)
                        for num_str in numbers:
                            clean_num = num_str.replace(',', '')
                            if len(clean_num) > 3:  # Reasonable parameter count
                                param_count = int(clean_num)
                                break
                        if param_count:
                            break
                    except ValueError:
                        continue
        
        # Validation and fallback estimation
        if hr_value is None:
            self.logger.warning(f"Could not parse HR@10 for {config_name} run {run_id}")
            return None
        
        if ndcg_value is None:
            self.logger.warning(f"Could not parse NDCG@10 for {config_name} run {run_id}, setting to 0")
            ndcg_value = 0.0
        
        if param_count is None:
            # Estimate based on architecture (from your Q3 analysis)
            num_layers = int(config_name[0])  # Extract from config name
            param_count = self._estimate_parameters(num_layers)
            self.logger.warning(f"Estimated parameter count for {config_name}: {param_count:,}")
        
        return {
            'hr': hr_value,
            'ndcg': ndcg_value,
            'parameters': param_count,
            'num_layers': int(config_name[0]),
            'pretraining': 'pretrain' in config_name,
            'run_id': run_id,
            'config_name': config_name
        }
    
    def _estimate_parameters(self, num_layers):
        """Estimate parameter count based on model architecture."""
        # Based on your NCF architecture and config values
        user_num, item_num = config.user_num, config.item_num
        factor_num = self.factor_num
        
        # GMF embeddings: constant
        gmf_params = (user_num + item_num) * factor_num
        
        # MLP embeddings: depends on layers
        mlp_factor = factor_num * (2 ** (num_layers - 1))
        mlp_embed_params = (user_num + item_num) * mlp_factor
        
        # MLP layers
        input_size = factor_num * (2 ** num_layers)
        mlp_layer_params = 0
        for i in range(num_layers):
            output_size = input_size // 2
            mlp_layer_params += input_size * output_size + output_size  # weights + bias
            input_size = output_size
        
        # Prediction layer
        pred_params = factor_num * 2 + 1  # NeuMF combines GMF + MLP
        
        total = gmf_params + mlp_embed_params + mlp_layer_params + pred_params
        return total
    
    def run_all_experiments(self):
        """Run all experimental configurations with proper progress tracking."""
        self.logger.info("Starting all experiments...")
        
        configurations = [
            (1, True),   (1, False),
            (2, True),   (2, False), 
            (3, True),   (3, False)
        ]
        
        total_experiments = len(configurations) * self.num_runs
        completed = 0
        
        for num_layers, pretraining in configurations:
            config_name = f"{num_layers}layers_{'pretrain' if pretraining else 'scratch'}"
            self.logger.info(f"Configuration: {config_name}")
            
            self.results[config_name] = {
                'num_layers': num_layers,
                'pretraining': pretraining,
                'runs': []
            }
            
            # Run experiments for this configuration
            for run_id in range(1, self.num_runs + 1):
                result = self.run_single_experiment(num_layers, pretraining, run_id)
                if result:
                    self.results[config_name]['runs'].append(result)
                else:
                    self.logger.warning(f"Skipping failed run {run_id} for {config_name}")
                
                completed += 1
                progress = (completed / total_experiments) * 100
                self.logger.info(f"Overall Progress: {completed}/{total_experiments} ({progress:.1f}%)")
        
        self.logger.info("All experiments completed!")
    
    def analyze_results(self):
        """Analyze and summarize results using statistical methods."""
        self.logger.info("Analyzing results...")
        
        analysis = {}
        
        for config_name, config_data in self.results.items():
            if not config_data['runs']:
                self.logger.warning(f"No successful runs for {config_name}")
                continue
            
            hrs = [run['hr'] for run in config_data['runs']]
            ndcgs = [run['ndcg'] for run in config_data['runs']]
            params = config_data['runs'][0]['parameters']  # Same for all runs
            
            analysis[config_name] = {
                'num_layers': config_data['num_layers'],
                'pretraining': config_data['pretraining'],
                'hr_mean': np.mean(hrs),
                'hr_std': np.std(hrs),
                'hr_min': np.min(hrs),
                'hr_max': np.max(hrs),
                'ndcg_mean': np.mean(ndcgs), 
                'ndcg_std': np.std(ndcgs),
                'ndcg_min': np.min(ndcgs),
                'ndcg_max': np.max(ndcgs),
                'parameters': params,
                'num_runs': len(hrs),
                'success_rate': len(hrs) / self.num_runs * 100
            }
            
            self.logger.info(f"{config_name:20s}: HR@10={np.mean(hrs):.4f}+/-{np.std(hrs):.4f} "
                           f"NDCG@10={np.mean(ndcgs):.4f}+/-{np.std(ndcgs):.4f} "
                           f"Params={params:,} Success={len(hrs)}/{self.num_runs}")
        
        return analysis
    
    def create_plots(self, analysis):
        """Create visualization plots using utilities where possible."""
        self.logger.info("Creating plots...")
        
        # Prepare data for plotting
        layers = [1, 2, 3]
        pretrain_data = {}
        scratch_data = {}
        
        for layer in layers:
            pretrain_key = f"{layer}layers_pretrain"
            scratch_key = f"{layer}layers_scratch"
            
            if pretrain_key in analysis:
                pretrain_data[layer] = analysis[pretrain_key]
            if scratch_key in analysis:
                scratch_data[layer] = analysis[scratch_key]
        
        # Create comprehensive plots
        self._create_main_comparison_plot(layers, pretrain_data, scratch_data)
        self._create_detailed_analysis_plots(layers, pretrain_data, scratch_data)
        self._create_summary_table(analysis)
    
    def _create_main_comparison_plot(self, layers, pretrain_data, scratch_data):
        """Create the main comparison plot for Q2 and Q3."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Extract data
        pretrain_hrs = [pretrain_data.get(l, {}).get('hr_mean', 0) for l in layers]
        scratch_hrs = [scratch_data.get(l, {}).get('hr_mean', 0) for l in layers]
        pretrain_stds = [pretrain_data.get(l, {}).get('hr_std', 0) for l in layers]
        scratch_stds = [scratch_data.get(l, {}).get('hr_std', 0) for l in layers]
        
        pretrain_params = [pretrain_data.get(l, {}).get('parameters', 0) for l in layers]
        scratch_params = [scratch_data.get(l, {}).get('parameters', 0) for l in layers]
        
        # Plot 1: HR@10 comparison
        x = np.arange(len(layers))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, pretrain_hrs, width, yerr=pretrain_stds,
                       label='With Pretraining', color='skyblue', capsize=5, alpha=0.8)
        bars2 = ax1.bar(x + width/2, scratch_hrs, width, yerr=scratch_stds,
                       label='Without Pretraining', color='lightcoral', capsize=5, alpha=0.8)
        
        ax1.set_xlabel('Number of MLP Layers')
        ax1.set_ylabel('HR@10')
        ax1.set_title('Question 2: HR@10 vs Number of MLP Layers')
        ax1.set_xticks(x)
        ax1.set_xticklabels(layers)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars, values, stds in [(bars1, pretrain_hrs, pretrain_stds), (bars2, scratch_hrs, scratch_stds)]:
            for bar, val, std in zip(bars, values, stds):
                if val > 0:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.005,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9, weight='bold')
        
        # Plot 2: Parameter count comparison
        bars3 = ax2.bar(x - width/2, pretrain_params, width, 
                       label='With Pretraining', color='skyblue', alpha=0.8)
        bars4 = ax2.bar(x + width/2, scratch_params, width,
                       label='Without Pretraining', color='lightcoral', alpha=0.8)
        
        ax2.set_xlabel('Number of MLP Layers')
        ax2.set_ylabel('Number of Parameters')
        ax2.set_title('Question 3: Parameters vs Number of MLP Layers')
        ax2.set_xticks(x)
        ax2.set_xticklabels(layers)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars, params in [(bars3, pretrain_params), (bars4, scratch_params)]:
            for bar, val in zip(bars, params):
                if val > 0:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{val:,}', ha='center', va='bottom', fontsize=8, 
                            rotation=45, weight='bold')
        
        plt.tight_layout()
        
        # Save plot using config
        plot_path = config.figure_dir / "question_02_mlp_layers_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Main comparison plot saved to: {plot_path}")
    
    def _create_detailed_analysis_plots(self, layers, pretrain_data, scratch_data):
        """Create detailed analysis plots."""
        # Performance distribution plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot distributions for each configuration
        for layer in layers:
            if layer in pretrain_data:
                data = pretrain_data[layer]
                if f"{layer}layers_pretrain" in self.results:
                    hrs = [run['hr'] for run in self.results[f"{layer}layers_pretrain"]['runs']]
                    ax1.hist(hrs, alpha=0.6, label=f'{layer}L Pretrain', bins=10)
        
        ax1.set_xlabel('HR@10')
        ax1.set_ylabel('Frequency')
        ax1.set_title('HR@10 Distribution - With Pretraining')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Similar for without pretraining
        for layer in layers:
            if layer in scratch_data:
                if f"{layer}layers_scratch" in self.results:
                    hrs = [run['hr'] for run in self.results[f"{layer}layers_scratch"]['runs']]
                    ax2.hist(hrs, alpha=0.6, label=f'{layer}L Scratch', bins=10)
        
        ax2.set_xlabel('HR@10')
        ax2.set_ylabel('Frequency')
        ax2.set_title('HR@10 Distribution - Without Pretraining')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Box plots
        pretrain_all_hrs = []
        scratch_all_hrs = []
        pretrain_labels = []
        scratch_labels = []
        
        for layer in layers:
            if f"{layer}layers_pretrain" in self.results:
                hrs = [run['hr'] for run in self.results[f"{layer}layers_pretrain"]['runs']]
                pretrain_all_hrs.append(hrs)
                pretrain_labels.append(f'{layer}L')
            
            if f"{layer}layers_scratch" in self.results:
                hrs = [run['hr'] for run in self.results[f"{layer}layers_scratch"]['runs']]
                scratch_all_hrs.append(hrs)
                scratch_labels.append(f'{layer}L')
        
        if pretrain_all_hrs:
            ax3.boxplot(pretrain_all_hrs, labels=pretrain_labels)
            ax3.set_xlabel('MLP Layers')
            ax3.set_ylabel('HR@10')
            ax3.set_title('HR@10 Variability - With Pretraining')
            ax3.grid(True, alpha=0.3)
        
        if scratch_all_hrs:
            ax4.boxplot(scratch_all_hrs, labels=scratch_labels)
            ax4.set_xlabel('MLP Layers')
            ax4.set_ylabel('HR@10')
            ax4.set_title('HR@10 Variability - Without Pretraining')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save detailed plot
        detailed_path = config.figure_dir / "question_02_detailed_analysis.png"
        plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Detailed analysis plot saved to: {detailed_path}")
    
    def _create_summary_table(self, analysis):
        """Create a comprehensive summary table."""
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        headers = ['Configuration', 'Layers', 'Pretraining', 'HR@10 (μ±σ)', 'NDCG@10 (μ±σ)', 
                  'Parameters', 'Success Rate', 'HR Range']
        
        for config_name, data in analysis.items():
            row = [
                config_name,
                str(data['num_layers']),
                'Yes' if data['pretraining'] else 'No',
                f"{data['hr_mean']:.4f}±{data['hr_std']:.4f}",
                f"{data['ndcg_mean']:.4f}±{data['ndcg_std']:.4f}",
                f"{data['parameters']:,}",
                f"{data['success_rate']:.0f}%",
                f"[{data['hr_min']:.3f}, {data['hr_max']:.3f}]"
            ]
            table_data.append(row)
        
        table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code the rows alternately
        colors = ['#F5F5F5', '#E8F5E8']
        for i in range(1, len(table_data) + 1):
            color = colors[i % 2]
            for j in range(len(headers)):
                table[(i, j)].set_facecolor(color)
        
        ax.set_title('Question 2: Comprehensive Results Summary\n'
                    f'({self.num_runs} runs per configuration, {self.epochs} epochs each)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Save table
        table_path = config.figure_dir / "question_02_results_table.png"
        plt.savefig(table_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Results table saved to: {table_path}")
    
    def save_results(self, analysis):
        """Save results using io utilities."""
        self.logger.info("Saving results using io utilities...")
        
        # Save detailed results using utility
        results_path = config.output_dir / "reports" / "question_02_results.json"
        save_json(self.results, results_path)
        
        # Save analysis summary using utility
        analysis_path = config.output_dir / "reports" / "question_02_analysis.json"
        save_json(analysis, analysis_path)
        
        # Save experiment metadata
        metadata = {
            'experiment_info': {
                'num_runs': self.num_runs,
                'epochs': self.epochs,
                'factor_num': self.factor_num,
                'total_experiments': len(self.results) * self.num_runs,
                'configurations': list(self.results.keys()),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'success_rates': {k: v.get('success_rate', 0) for k, v in analysis.items()}
            }
        }
        
        metadata_path = config.output_dir / "reports" / "question_02_metadata.json"
        save_json(metadata, metadata_path)
        
        self.logger.info(f"Detailed results saved to: {results_path}")
        self.logger.info(f"Analysis summary saved to: {analysis_path}")
        self.logger.info(f"Experiment metadata saved to: {metadata_path}")
    
    def run_experiment(self):
        """Run the complete experiment with proper error handling."""
        self.logger.info("Starting Question 2 experiment...")
        
        try:
            # Step 1: Ensure we have the required pretrained models
            self.ensure_pretrained_models_exist()
            
            # Step 2: Run all experimental configurations
            self.run_all_experiments()
            
            # Step 3: Analyze results
            analysis = self.analyze_results()
            
            if not analysis:
                self.logger.error("No successful experiments found!")
                return None
            
            # Step 4: Create visualizations
            self.create_plots(analysis)
            
            # Step 5: Save results
            self.save_results(analysis)
            
            # Step 6: Log completion
            self.logger.info("="*60)
            self.logger.info("QUESTION 2 EXPERIMENT COMPLETED!")
            self.logger.info("="*60)
            self.logger.info("Generated files:")
            self.logger.info(f"- {config.figure_dir}/question_02_mlp_layers_analysis.png")
            self.logger.info(f"- {config.figure_dir}/question_02_detailed_analysis.png")
            self.logger.info(f"- {config.figure_dir}/question_02_results_table.png")
            self.logger.info(f"- {config.output_dir}/reports/question_02_results.json")
            self.logger.info(f"- {config.output_dir}/reports/question_02_analysis.json")
            self.logger.info(f"- {config.output_dir}/reports/question_02_metadata.json")
            self.logger.info("="*60)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Critical error in experiment: {e}")
            raise


def main():
    """Main function with proper error handling and logging."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Question 2 experiment')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs per configuration')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs per run')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer runs')
    
    args = parser.parse_args()
    
    if args.quick:
        print("Running in quick mode (fewer runs for testing)")
        args.runs = 2
        args.epochs = 5
    
    print("Question 2: MLP Layers Effect on NeuMF Performance")
    print(f"Running {args.runs} runs per configuration with {args.epochs} epochs each")
    
    try:
        # Create and run experiment
        experiment = Question2Experiment(num_runs=args.runs, epochs=args.epochs)
        results = experiment.run_experiment()
        
        if results:
            print("\n[SUCCESS] Question 2 experiment completed!")
            print("Check the figures and reports directories for results.")
            
            # Print summary
            print("\nExperiment Summary:")
            for config_name, data in results.items():
                print(f"{config_name}: HR@10 = {data['hr_mean']:.4f}+/-{data['hr_std']:.4f}")
        else:
            print("\n[FAILED] Could not complete experiment.")
            print("Check the logs for details.")
    
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Experiment was interrupted by user.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        print("Check the logs for more details.")
        raise


if __name__ == "__main__":
    main()