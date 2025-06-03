#!/usr/bin/env python3
"""
Factor Number Experiment: Finding optimal embedding size for NCF models.

This experiment investigates how the number of latent factors (embedding size) 
affects model performance across different NCF variants.

Based on the original NCF paper which tested factors [8, 16, 32, 64].
Uses comprehensive utils infrastructure for consistency and maintainability.
"""

import os
import sys
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Use ALL available project utilities
from src.utils.config import config
from src.utils.io import ensure_dir, save_json, load_json, save_pickle, save_results
from src.utils.logging import get_experiment_logger, setup_logger
from src.utils.visualization import (plot_comparative_metrics, save_run_history, 
                                   aggregate_runs, plot_training_metrics)

class FactorExperiment:
    def __init__(self, num_runs=10, epochs=20):
        """Initialize factor experiment with proper logging and config using utils."""
        self.num_runs = num_runs
        self.epochs = epochs
        self.results = {}
        
        # Factor configurations to test (based on NCF paper)
        self.factor_configs = [8, 16, 32, 64]
        
        # Fixed architecture for consistency
        self.num_layers = 3
        
        # Initialize logger using utils
        self.logger = get_experiment_logger("factor_experiment")
        
        # Ensure ALL output directories exist using io utils
        ensure_dir(config.figure_dir)
        ensure_dir(config.model_dir)
        ensure_dir(config.log_dir)
        ensure_dir(config.output_dir / "reports")
        ensure_dir(config.output_dir / "tables")
        
        # Setup experiment metadata using utils approach
        self.experiment_metadata = {
            'experiment_type': 'factor_optimization',
            'start_time': datetime.now().isoformat(),
            'configuration': {
                'num_runs': num_runs,
                'epochs': epochs,
                'factor_configs': self.factor_configs,
                'num_layers': self.num_layers
            },
            'environment': {
                'config_factor_num': config.factor_num,
                'config_user_num': config.user_num,
                'config_item_num': config.item_num
            }
        }
        
        self._log_experiment_info()
    
    def _log_experiment_info(self):
        """Log experiment configuration."""
        self.logger.info("="*60)
        self.logger.info("FACTOR NUMBER EXPERIMENT: OPTIMAL EMBEDDING SIZE")
        self.logger.info("="*60)
        self.logger.info("Configuration:")
        self.logger.info(f"- Number of runs per setup: {self.num_runs}")
        self.logger.info(f"- Epochs per run: {self.epochs}")
        self.logger.info(f"- Factor configurations: {self.factor_configs}")
        self.logger.info(f"- Fixed MLP layers: {self.num_layers}")
        self.logger.info(f"- Models tested: GMF, MLP, NeuMF-end")
        self.logger.info("="*60)
    
    def run_single_experiment(self, model_type, factor_num, run_id):
        """Run a single training experiment for given model and factor number."""
        config_name = f"{model_type}_{factor_num}f"
        self.logger.info(f"Run {run_id}/{self.num_runs} - {config_name}")
        
        # Prepare command based on model type
        if model_type in ["GMF", "MLP"]:
            cmd = [
                "python", "scripts/pretrain.py",
                "--model", model_type,
                "--epochs", str(self.epochs),
                "--factor_num", str(factor_num),
                "--num_layers", str(self.num_layers)
            ]
        else:  # NeuMF-end
            cmd = [
                "python", "scripts/train_neumf.py",
                "--model", "NeuMF-end",
                "--epochs", str(self.epochs),
                "--factor_num", str(factor_num),
                "--num_layers", str(self.num_layers)
            ]
        
        return self._execute_training_command(cmd, config_name, run_id)
    
    def _execute_training_command(self, cmd, config_name, run_id):
        """Execute training command and parse results."""
        try:
            self.logger.info(f"Executing: {' '.join(cmd)}")
            # Python 3.6 compatible subprocess call
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                  universal_newlines=True, check=True)
            
            # Parse the output
            output_lines = result.stdout.split('\n')
            parsed_result = self._parse_training_output(output_lines, config_name, run_id)
            
            if parsed_result:
                self.logger.info(f"[OK] {config_name} Run {run_id}: HR@10={parsed_result['hr']:.4f}, "
                               f"NDCG@10={parsed_result['ndcg']:.4f}, Params={parsed_result['parameters']:,}")
                return parsed_result
            else:
                self.logger.error(f"[ERROR] {config_name} Run {run_id}: Failed to parse results")
                return None
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"[ERROR] {config_name} Run {run_id}: Training failed - {e}")
            if hasattr(e, 'stdout') and e.stdout:
                self.logger.error(f"STDOUT: {e.stdout[-500:]}")
            if hasattr(e, 'stderr') and e.stderr:
                self.logger.error(f"STDERR: {e.stderr[-500:]}")
            return None
        except Exception as e:
            self.logger.error(f"[ERROR] {config_name} Run {run_id}: Unexpected error - {e}")
            return None
    
    def _parse_training_output(self, output_lines, config_name, run_id):
        """Parse training output with multiple fallback strategies."""
        hr_value = None
        ndcg_value = None
        param_count = None
        
        # Strategy 1: Look for "--- RESULTS ---" section
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
        
        # Strategy 2: Look for "Best Result:" or "Final Results:"
        if hr_value is None:
            for line in output_lines:
                if any(keyword in line for keyword in ["Best Result:", "Final Results:", "HR@10:"]):
                    if "HR" in line and "NDCG" in line:
                        try:
                            # Parse various formats
                            if "HR=" in line:
                                hr_part = line.split("HR=")[1].split(",")[0].strip()
                                hr_value = float(hr_part)
                            elif "HR@10:" in line:
                                hr_part = line.split("HR@10:")[1].split(",")[0].strip()
                                hr_value = float(hr_part)
                            
                            if "NDCG=" in line:
                                ndcg_part = line.split("NDCG=")[1].split(",")[0].strip()
                                ndcg_value = float(ndcg_part)
                            elif "NDCG@10:" in line:
                                ndcg_part = line.split("NDCG@10:")[1].strip()
                                ndcg_value = float(ndcg_part)
                            
                            break
                        except (IndexError, ValueError):
                            continue
        
        # Strategy 3: Look for parameter count
        if param_count is None:
            for line in output_lines:
                if any(keyword in line.lower() for keyword in ["parameters:", "params:", "parameter count"]):
                    try:
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
        
        # Validation and fallback
        if hr_value is None:
            self.logger.warning(f"Could not parse HR@10 for {config_name} run {run_id}")
            return None
        
        if ndcg_value is None:
            self.logger.warning(f"Could not parse NDCG@10 for {config_name} run {run_id}, setting to 0")
            ndcg_value = 0.0
        
        if param_count is None:
            # Estimate parameters based on architecture
            factor_num = int(config_name.split('_')[1].replace('f', ''))
            param_count = self._estimate_parameters(config_name.split('_')[0], factor_num)
            self.logger.warning(f"Estimated parameter count for {config_name}: {param_count:,}")
        
        return {
            'hr': hr_value,
            'ndcg': ndcg_value,
            'parameters': param_count,
            'factor_num': int(config_name.split('_')[1].replace('f', '')),
            'model_type': config_name.split('_')[0],
            'run_id': run_id,
            'config_name': config_name
        }
    
    def _estimate_parameters(self, model_type, factor_num):
        """Estimate parameter count based on model architecture."""
        user_num, item_num = config.user_num, config.item_num
        
        if model_type == "GMF":
            # GMF: user embedding + item embedding + prediction layer
            return (user_num + item_num) * factor_num + factor_num + 1
        
        elif model_type == "MLP":
            # MLP embeddings
            mlp_factor = factor_num * (2 ** (self.num_layers - 1))
            mlp_embed_params = (user_num + item_num) * mlp_factor
            
            # MLP layers
            input_size = factor_num * (2 ** self.num_layers)
            mlp_layer_params = 0
            for i in range(self.num_layers):
                output_size = input_size // 2
                mlp_layer_params += input_size * output_size + output_size
                input_size = output_size
            
            # Prediction layer
            pred_params = factor_num + 1
            
            return mlp_embed_params + mlp_layer_params + pred_params
        
        else:  # NeuMF-end
            # Combination of GMF and MLP
            gmf_params = (user_num + item_num) * factor_num
            
            mlp_factor = factor_num * (2 ** (self.num_layers - 1))
            mlp_embed_params = (user_num + item_num) * mlp_factor
            
            input_size = factor_num * (2 ** self.num_layers)
            mlp_layer_params = 0
            for i in range(self.num_layers):
                output_size = input_size // 2
                mlp_layer_params += input_size * output_size + output_size
                input_size = output_size
            
            pred_params = factor_num * 2 + 1  # GMF + MLP outputs
            
            return gmf_params + mlp_embed_params + mlp_layer_params + pred_params
    
    def run_all_experiments(self):
        """Run all factor experiments for different models."""
        self.logger.info("Starting all factor experiments...")
        
        model_types = ["GMF", "MLP", "NeuMF-end"]
        total_experiments = len(model_types) * len(self.factor_configs) * self.num_runs
        completed = 0
        
        for model_type in model_types:
            self.logger.info(f"Testing {model_type} with different factor numbers...")
            
            for factor_num in self.factor_configs:
                config_name = f"{model_type}_{factor_num}f"
                self.logger.info(f"Configuration: {config_name}")
                
                self.results[config_name] = {
                    'model_type': model_type,
                    'factor_num': factor_num,
                    'runs': []
                }
                
                # Run experiments for this configuration
                for run_id in range(1, self.num_runs + 1):
                    result = self.run_single_experiment(model_type, factor_num, run_id)
                    if result:
                        self.results[config_name]['runs'].append(result)
                    else:
                        self.logger.warning(f"Skipping failed run {run_id} for {config_name}")
                    
                    completed += 1
                    progress = (completed / total_experiments) * 100
                    self.logger.info(f"Overall Progress: {completed}/{total_experiments} ({progress:.1f}%)")
        
        self.logger.info("All factor experiments completed!")
    
    def analyze_results(self):
        """Analyze and summarize results."""
        self.logger.info("Analyzing factor experiment results...")
        
        analysis = {}
        
        for config_name, config_data in self.results.items():
            if not config_data['runs']:
                self.logger.warning(f"No successful runs for {config_name}")
                continue
            
            hrs = [run['hr'] for run in config_data['runs']]
            ndcgs = [run['ndcg'] for run in config_data['runs']]
            params = config_data['runs'][0]['parameters']
            
            analysis[config_name] = {
                'model_type': config_data['model_type'],
                'factor_num': config_data['factor_num'],
                'hr_mean': np.mean(hrs),
                'hr_std': np.std(hrs),
                'hr_max': np.max(hrs),
                'hr_min': np.min(hrs),
                'ndcg_mean': np.mean(ndcgs),
                'ndcg_std': np.std(ndcgs),
                'parameters': params,
                'num_runs': len(hrs),
                'success_rate': len(hrs) / self.num_runs * 100
            }
            
            self.logger.info(f"{config_name:15s}: HR@10={np.mean(hrs):.4f}±{np.std(hrs):.4f} "
                           f"NDCG@10={np.mean(ndcgs):.4f}±{np.std(ndcgs):.4f} "
                           f"Params={params:,}")
        
        return analysis
    
    def create_plots(self, analysis):
        """Create comprehensive visualization plots using utils/visualization.py."""
        self.logger.info("Creating factor analysis plots using visualization utilities...")
        
        # Convert our results to the format expected by visualization utils
        self._create_plots_using_utils(analysis)
        
        # Create additional custom plots for factor-specific analysis
        self._create_factor_specific_plots(analysis)
    
    def _create_plots_using_utils(self, analysis):
        """Use existing visualization utilities for comparative plots."""
        # Prepare data by model type for comparative plotting
        model_types = ["GMF", "MLP", "NeuMF-end"]
        
        for model_type in model_types:
            # Get all factor configurations for this model
            model_configs = {k: v for k, v in self.results.items() 
                           if k.startswith(model_type)}
            
            if not model_configs:
                continue
            
            # Convert to format expected by visualization utils
            run_histories_list = []
            model_names = []
            
            for config_name, config_data in model_configs.items():
                if not config_data['runs']:
                    continue
                
                # Convert individual runs to history format
                factor_num = config_data['factor_num']
                runs_as_histories = []
                
                for run in config_data['runs']:
                    # Create a single-epoch history (since we only have final results)
                    history = [{
                        'epoch': 1,
                        'hr': run['hr'],
                        'ndcg': run['ndcg'],
                        'loss': 0.1  # Dummy loss value
                    }]
                    runs_as_histories.append(history)
                
                run_histories_list.append(runs_as_histories)
                model_names.append(f"{factor_num} factors")
            
            if run_histories_list:
                # Use the existing comparative plotting utility
                try:
                    output_path = config.figure_dir / f"factor_comparison_{model_type.lower()}.png"
                    plot_comparative_metrics(
                        run_histories_list=run_histories_list,
                        model_names=model_names,
                        output_path=output_path,
                        metrics=["hr", "ndcg"],
                        metric_labels={"hr": "HR@10", "ndcg": "NDCG@10"}
                    )
                    self.logger.info(f"Comparative plot for {model_type} saved using visualization utils")
                except Exception as e:
                    self.logger.warning(f"Could not use visualization utils for {model_type}: {e}")
                    # Fallback to manual plotting if needed
                    self._create_manual_comparison_plot(model_type, model_configs)
    
    def _create_factor_specific_plots(self, analysis):
        """Create factor-specific plots that showcase the key insights."""
        # Prepare data by model type
        model_data = {'GMF': {}, 'MLP': {}, 'NeuMF-end': {}}
        
        for config_name, data in analysis.items():
            model_type = data['model_type']
            factor_num = data['factor_num']
            model_data[model_type][factor_num] = data
        
        # Create a comprehensive factor analysis plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: HR@10 vs Factor Number (using utils style)
        self._plot_metric_vs_factors(ax1, model_data, 'hr_mean', 'hr_std', 
                                    'HR@10', 'HR@10 vs Number of Factors')
        
        # Plot 2: NDCG@10 vs Factor Number
        self._plot_metric_vs_factors(ax2, model_data, 'ndcg_mean', 'ndcg_std', 
                                    'NDCG@10', 'NDCG@10 vs Number of Factors')
        
        # Plot 3: Parameters vs Factor Number
        self._plot_parameters_vs_factors(ax3, model_data)
        
        # Plot 4: Performance vs Model Complexity
        self._plot_performance_vs_complexity(ax4, model_data)
        
        plt.tight_layout()
        
        # Save plot using config paths
        plot_path = config.figure_dir / "factor_experiment_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Factor analysis plot saved to: {plot_path}")
        
        # Create optimal factors summary using utils style
        self._create_optimal_factors_summary_with_utils(analysis)
    
    def _plot_metric_vs_factors(self, ax, model_data, mean_key, std_key, ylabel, title):
        """Plot a metric vs factors with error bars, using utils style."""
        colors = {'GMF': '#1f77b4', 'MLP': '#ff7f0e', 'NeuMF-end': '#2ca02c'}  # matplotlib default colors
        markers = {'GMF': 'o', 'MLP': 's', 'NeuMF-end': '^'}
        
        for model_type in ['GMF', 'MLP', 'NeuMF-end']:
            if not model_data[model_type]:
                continue
                
            factors = sorted(model_data[model_type].keys())
            means = [model_data[model_type][f][mean_key] for f in factors]
            stds = [model_data[model_type][f][std_key] for f in factors]
            
            ax.errorbar(factors, means, yerr=stds, 
                       marker=markers[model_type], label=model_type,
                       color=colors[model_type], linewidth=2, markersize=8, 
                       capsize=5, capthick=2)
        
        ax.set_xlabel('Number of Factors')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(self.factor_configs)
        
        # Add value annotations (following utils style)
        for model_type in ['GMF', 'MLP', 'NeuMF-end']:
            if not model_data[model_type]:
                continue
            factors = sorted(model_data[model_type].keys())
            means = [model_data[model_type][f][mean_key] for f in factors]
            for f, m in zip(factors, means):
                ax.annotate(f'{m:.3f}', (f, m), 
                           textcoords="offset points", xytext=(0,10), 
                           ha='center', fontsize=8, alpha=0.8)
    
    def _plot_parameters_vs_factors(self, ax, model_data):
        """Plot parameters vs factors."""
        colors = {'GMF': '#1f77b4', 'MLP': '#ff7f0e', 'NeuMF-end': '#2ca02c'}
        markers = {'GMF': 'o', 'MLP': 's', 'NeuMF-end': '^'}
        
        for model_type in ['GMF', 'MLP', 'NeuMF-end']:
            if not model_data[model_type]:
                continue
                
            factors = sorted(model_data[model_type].keys())
            params = [model_data[model_type][f]['parameters'] for f in factors]
            
            ax.plot(factors, params, marker=markers[model_type], label=model_type,
                   color=colors[model_type], linewidth=2, markersize=8)
        
        ax.set_xlabel('Number of Factors')
        ax.set_ylabel('Number of Parameters')
        ax.set_title('Model Complexity vs Number of Factors')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(self.factor_configs)
        ax.set_yscale('log')
        
        # Add parameter count annotations
        for model_type in ['GMF', 'MLP', 'NeuMF-end']:
            if not model_data[model_type]:
                continue
            factors = sorted(model_data[model_type].keys())
            params = [model_data[model_type][f]['parameters'] for f in factors]
            for f, p in zip(factors, params):
                ax.annotate(f'{p:,}', (f, p), 
                           textcoords="offset points", xytext=(0,10), 
                           ha='center', fontsize=7, alpha=0.8, rotation=45)
    
    def _plot_performance_vs_complexity(self, ax, model_data):
        """Plot performance vs model complexity."""
        colors = {'GMF': '#1f77b4', 'MLP': '#ff7f0e', 'NeuMF-end': '#2ca02c'}
        
        for model_type in ['GMF', 'MLP', 'NeuMF-end']:
            if not model_data[model_type]:
                continue
                
            factors = sorted(model_data[model_type].keys())
            params = [model_data[model_type][f]['parameters'] for f in factors]
            hrs = [model_data[model_type][f]['hr_mean'] for f in factors]
            
            # Add factor labels to points
            for f, p, h in zip(factors, params, hrs):
                ax.scatter(p, h, s=120, color=colors[model_type], alpha=0.7)
                ax.annotate(f'{f}f', (p, h), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8,
                           fontweight='bold')
            
            ax.plot(params, hrs, label=model_type, color=colors[model_type], 
                   linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Number of Parameters')
        ax.set_ylabel('HR@10')
        ax.set_title('Performance vs Model Complexity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    def _create_optimal_factors_summary_with_utils(self, analysis):
        """Create optimal factors summary following utils visualization style."""
        # Find optimal factors for each model
        optimal_results = {}
        for model_type in ["GMF", "MLP", "NeuMF-end"]:
            model_configs = {k: v for k, v in analysis.items() if v['model_type'] == model_type}
            if model_configs:
                best_config = max(model_configs.items(), key=lambda x: x[1]['hr_mean'])
                optimal_results[model_type] = best_config[1]
        
        if not optimal_results:
            return
        
        # Create bar chart following utils style
        fig, ax = plt.subplots(figsize=(12, 8))
        
        models = list(optimal_results.keys())
        factors = [optimal_results[m]['factor_num'] for m in models]
        hrs = [optimal_results[m]['hr_mean'] for m in models]
        hr_stds = [optimal_results[m]['hr_std'] for m in models]
        
        # Use consistent colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        bars = ax.bar(models, hrs, yerr=hr_stds, capsize=5, 
                     color=colors[:len(models)], alpha=0.8, 
                     edgecolor='black', linewidth=1)
        
        # Add value labels following utils style
        for i, (model, bar) in enumerate(zip(models, bars)):
            height = bar.get_height()
            factor_num = optimal_results[model]['factor_num']
            params = optimal_results[model]['parameters']
            
            # Value on top of bar
            ax.text(bar.get_x() + bar.get_width()/2., height + hr_stds[i] + 0.005,
                   f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
            
            # Factor and parameter info below
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'{factor_num} factors\n{params:,} params',
                   ha='center', va='center', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.set_ylabel('HR@10')
        ax.set_title('Optimal Factor Configuration for Each Model Type\n'
                    f'(Based on {self.num_runs} runs per configuration)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis to start from reasonable value
        y_min = min(hrs) - max(hr_stds) - 0.02
        y_max = max(hrs) + max(hr_stds) + 0.03
        ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        
        # Save using config paths
        summary_path = config.figure_dir / "optimal_factors_summary.png"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Optimal factors summary saved to: {summary_path}")
        
        return optimal_results
    
    def _create_manual_comparison_plot(self, model_type, model_configs):
        """Fallback manual plotting if utils don't work."""
        self.logger.info(f"Creating fallback plot for {model_type}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        factors = []
        hr_means = []
        hr_stds = []
        ndcg_means = []
        ndcg_stds = []
        
        for config_name, config_data in sorted(model_configs.items()):
            if not config_data['runs']:
                continue
            
            factor_num = config_data['factor_num']
            hrs = [run['hr'] for run in config_data['runs']]
            ndcgs = [run['ndcg'] for run in config_data['runs']]
            
            factors.append(factor_num)
            hr_means.append(np.mean(hrs))
            hr_stds.append(np.std(hrs))
            ndcg_means.append(np.mean(ndcgs))
            ndcg_stds.append(np.std(ndcgs))
        
        # HR@10 plot
        ax1.errorbar(factors, hr_means, yerr=hr_stds, marker='o', 
                    linewidth=2, markersize=8, capsize=5)
        ax1.set_xlabel('Number of Factors')
        ax1.set_ylabel('HR@10')
        ax1.set_title(f'{model_type}: HR@10 vs Factors')
        ax1.grid(True, alpha=0.3)
        
        # NDCG@10 plot
        ax2.errorbar(factors, ndcg_means, yerr=ndcg_stds, marker='s', 
                    linewidth=2, markersize=8, capsize=5)
        ax2.set_xlabel('Number of Factors')
        ax2.set_ylabel('NDCG@10')
        ax2.set_title(f'{model_type}: NDCG@10 vs Factors')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        fallback_path = config.figure_dir / f"factor_analysis_{model_type.lower()}_fallback.png"
        plt.savefig(fallback_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Fallback plot for {model_type} saved to: {fallback_path}")
    
    def save_results(self, analysis):
        """Save results using comprehensive io utilities."""
        self.logger.info("Saving factor experiment results using io utilities...")
        
        # Update experiment metadata
        self.experiment_metadata.update({
            'end_time': datetime.now().isoformat(),
            'total_experiments_run': len(self.results) * self.num_runs,
            'successful_configs': len(analysis),
            'completion_status': 'success'
        })
        
        # Use save_results utility (follows your project pattern)
        experiment_name = f"factor_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare comprehensive results for save_results utility
        comprehensive_results = {
            'metadata': self.experiment_metadata,
            'raw_results': self.results,
            'analysis': analysis,
            'summary_statistics': self._generate_summary_statistics(analysis)
        }
        
        # Use the save_results utility from io.py
        save_results(comprehensive_results, experiment_name, config.output_dir)
        
        # Also save individual components using specific io utilities
        
        # 1. Detailed results as JSON (using save_json)
        results_path = config.output_dir / "reports" / "factor_experiment_results.json"
        save_json(self.results, results_path)
        
        # 2. Analysis as JSON (using save_json)
        analysis_path = config.output_dir / "reports" / "factor_experiment_analysis.json"
        save_json(analysis, analysis_path)
        
        # 3. Metadata as JSON (using save_json)
        metadata_path = config.output_dir / "reports" / "factor_experiment_metadata.json"
        save_json(self.experiment_metadata, metadata_path)
        
        # 4. Results as pickle for exact reproduction (using save_pickle)
        pickle_path = config.output_dir / "reports" / "factor_experiment_results.pkl"
        save_pickle(self.results, pickle_path)
        
        # 5. Create summary tables using io utils pattern
        self._save_summary_tables(analysis, experiment_name)
        
        # 6. Save individual run histories using visualization utils
        self._save_individual_run_histories()
        
        self.logger.info(f"Comprehensive results saved using io utilities:")
        self.logger.info(f"- Main results: {results_path}")
        self.logger.info(f"- Analysis: {analysis_path}")
        self.logger.info(f"- Metadata: {metadata_path}")
        self.logger.info(f"- Pickle backup: {pickle_path}")
        self.logger.info(f"- Comprehensive package: results/reports/{experiment_name}_results.json")
    
    def _generate_summary_statistics(self, analysis):
        """Generate summary statistics using utils pattern."""
        summary_stats = {
            'best_overall_config': None,
            'best_per_model': {},
            'factor_rankings': {},
            'performance_ranges': {}
        }
        
        # Find best overall configuration
        if analysis:
            best_config = max(analysis.items(), key=lambda x: x[1]['hr_mean'])
            summary_stats['best_overall_config'] = {
                'config_name': best_config[0],
                'model_type': best_config[1]['model_type'],
                'factor_num': best_config[1]['factor_num'],
                'hr_mean': best_config[1]['hr_mean'],
                'hr_std': best_config[1]['hr_std'],
                'parameters': best_config[1]['parameters']
            }
        
        # Best configuration per model type
        for model_type in ["GMF", "MLP", "NeuMF-end"]:
            model_configs = {k: v for k, v in analysis.items() if v['model_type'] == model_type}
            if model_configs:
                best_model_config = max(model_configs.items(), key=lambda x: x[1]['hr_mean'])
                summary_stats['best_per_model'][model_type] = {
                    'optimal_factors': best_model_config[1]['factor_num'],
                    'hr_mean': best_model_config[1]['hr_mean'],
                    'hr_std': best_model_config[1]['hr_std'],
                    'parameters': best_model_config[1]['parameters']
                }
        
        # Factor rankings across all models
        for factor in self.factor_configs:
            factor_configs = {k: v for k, v in analysis.items() if v['factor_num'] == factor}
            if factor_configs:
                avg_hr = np.mean([v['hr_mean'] for v in factor_configs.values()])
                summary_stats['factor_rankings'][factor] = avg_hr
        
        # Performance ranges
        if analysis:
            hr_values = [v['hr_mean'] for v in analysis.values()]
            summary_stats['performance_ranges'] = {
                'hr_min': min(hr_values),
                'hr_max': max(hr_values),
                'hr_range': max(hr_values) - min(hr_values),
                'hr_mean': np.mean(hr_values),
                'hr_std': np.std(hr_values)
            }
        
        return summary_stats
    
    def _save_summary_tables(self, analysis, experiment_name):
        """Save summary tables using io utils approach."""
        import pandas as pd
        
        # Create summary DataFrame
        summary_data = []
        for config_name, data in analysis.items():
            summary_data.append({
                'Configuration': config_name,
                'Model_Type': data['model_type'],
                'Factors': data['factor_num'],
                'HR_Mean': data['hr_mean'],
                'HR_Std': data['hr_std'],
                'NDCG_Mean': data['ndcg_mean'],
                'NDCG_Std': data['ndcg_std'],
                'Parameters': data['parameters'],
                'Success_Rate': data['success_rate']
            })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            
            # Save as CSV using ensure_dir pattern
            csv_dir = config.output_dir / "tables"
            ensure_dir(csv_dir)
            csv_path = csv_dir / f"{experiment_name}_summary.csv"
            df.to_csv(csv_path, index=False)
            
            self.logger.info(f"Summary table saved to: {csv_path}")
    
    def _save_individual_run_histories(self):
        """Save individual run histories using visualization utils."""
        for config_name, config_data in self.results.items():
            if not config_data['runs']:
                continue
            
            # Convert runs to history format for visualization utils
            for i, run in enumerate(config_data['runs']):
                # Create a simple history for this run
                history = [{
                    'epoch': 1,  # Single epoch since we only have final results
                    'hr': run['hr'],
                    'ndcg': run['ndcg'],
                    'loss': 0.1,  # Dummy loss
                    'parameters': run['parameters']
                }]
                
                # Save using visualization utility
                history_path = config.log_dir / f"{config_name}_run_{i+1}_history.json"
                try:
                    save_run_history(history, history_path)
                except Exception as e:
                    self.logger.warning(f"Could not save run history for {config_name} run {i+1}: {e}")
        
        self.logger.info("Individual run histories saved using visualization utils")
    
    def run_experiment(self):
        """Run the complete factor experiment using comprehensive utils."""
        self.logger.info("Starting factor number experiment using full utils infrastructure...")
        
        try:
            # Log experiment start using metadata
            start_time = time.time()
            self.logger.info(f"Experiment metadata: {self.experiment_metadata['experiment_type']}")
            
            # Run all experiments
            self.run_all_experiments()
            
            # Analyze results
            analysis = self.analyze_results()
            
            if not analysis:
                self.logger.error("No successful experiments found!")
                self.experiment_metadata['completion_status'] = 'failed'
                return None
            
            # Create visualizations using utils
            self.create_plots(analysis)
            
            # Save results using comprehensive io utils
            self.save_results(analysis)
            
            # Calculate total experiment time
            total_time = time.time() - start_time
            self.experiment_metadata['total_runtime_seconds'] = total_time
            
            # Final completion log using utils pattern
            self._log_experiment_completion(analysis, total_time)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Critical error in factor experiment: {e}")
            self.experiment_metadata['completion_status'] = 'error'
            self.experiment_metadata['error_message'] = str(e)
            
            # Save error state using io utils
            error_path = config.output_dir / "reports" / "factor_experiment_error.json"
            save_json(self.experiment_metadata, error_path)
            raise
    
    def _log_experiment_completion(self, analysis, total_time):
        """Log experiment completion using comprehensive utils logging."""
        self.logger.info("="*60)
        self.logger.info("FACTOR EXPERIMENT COMPLETED SUCCESSFULLY!")
        self.logger.info("="*60)
        self.logger.info(f"Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        self.logger.info(f"Successful configurations: {len(analysis)}")
        self.logger.info(f"Total experiments run: {len(self.results) * self.num_runs}")
        
        # Log best results
        if analysis:
            best_config = max(analysis.items(), key=lambda x: x[1]['hr_mean'])
            self.logger.info(f"Best configuration: {best_config[0]}")
            self.logger.info(f"Best HR@10: {best_config[1]['hr_mean']:.4f} ± {best_config[1]['hr_std']:.4f}")
        
        self.logger.info("Generated files using utils:")
        self.logger.info(f"- {config.figure_dir}/factor_experiment_analysis.png")
        self.logger.info(f"- {config.figure_dir}/optimal_factors_summary.png")
        self.logger.info(f"- {config.figure_dir}/factor_comparison_*.png (per model)")
        self.logger.info(f"- {config.output_dir}/reports/factor_experiment_results.json")
        self.logger.info(f"- {config.output_dir}/reports/factor_experiment_analysis.json")
        self.logger.info(f"- {config.output_dir}/reports/factor_experiment_metadata.json")
        self.logger.info(f"- {config.output_dir}/reports/factor_experiment_results.pkl")
        self.logger.info(f"- {config.output_dir}/tables/factor_experiment_*_summary.csv")
        self.logger.info(f"- {config.log_dir}/*_run_*_history.json (individual histories)")
        self.logger.info("="*60)


def main():
    """Main function with proper error handling."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Factor Number Experiment')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs per configuration')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs per run')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer runs')
    
    args = parser.parse_args()
    
    if args.quick:
        print("Running in quick mode (fewer runs for testing)")
        args.runs = 2
        args.epochs = 5
    
    print("Factor Number Experiment: Finding Optimal Embedding Size")
    print(f"Testing factors [8, 16, 32, 64] with {args.runs} runs per configuration")
    
    try:
        # Create and run experiment
        experiment = FactorExperiment(num_runs=args.runs, epochs=args.epochs)
        results = experiment.run_experiment()
        
        if results:
            print("\n[SUCCESS] Factor experiment completed!")
            print("Check the figures and reports directories for results.")
            
            # Print best configurations
            print("\nBest Configurations:")
            for model_type in ["GMF", "MLP", "NeuMF-end"]:
                model_configs = {k: v for k, v in results.items() if v['model_type'] == model_type}
                if model_configs:
                    best_config = max(model_configs.items(), key=lambda x: x[1]['hr_mean'])
                    config_name, data = best_config
                    print(f"{model_type}: {data['factor_num']} factors, "
                          f"HR@10={data['hr_mean']:.4f}±{data['hr_std']:.4f}")
        else:
            print("\n[FAILED] Could not complete factor experiment.")
            print("Check the logs for details.")
    
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Experiment was interrupted by user.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        print("Check the logs for more details.")
        raise


if __name__ == "__main__":
    main()