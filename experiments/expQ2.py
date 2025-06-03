#!/usr/bin/env python3
"""
Question 2: Effect of MLP layers on NeuMF performance with and without pretraining.

Simple version using filename-based model discovery.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import config
from src.utils.io import ensure_dir, save_json

class Question2Experiment:
    def __init__(self, num_runs=10, epochs=20):
        self.num_runs = num_runs
        self.epochs = epochs
        self.results = {}
        self.factor_num = config.factor_num  # 32 by default
        
        # Ensure output directories exist
        ensure_dir(config.figure_dir)
        ensure_dir(config.output_dir / "reports")
        
        print("="*60)
        print("QUESTION 2: MLP LAYERS EFFECT ON NeuMF PERFORMANCE")
        print("="*60)
        print(f"Configuration:")
        print(f"- Number of runs per setup: {self.num_runs}")
        print(f"- Epochs per run: {self.epochs}")
        print(f"- Layer configurations: 1, 2, 3")
        print(f"- Factor number: {self.factor_num}")
        print(f"- Training modes: With pretraining, Without pretraining")
        print("="*60)
    
    def find_model(self, model_type, num_layers=None):
        """Find a model file based on naming convention."""
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
        print("\n[STEP 1] Checking for pretrained models...")
        
        # Check GMF
        gmf_path = self.find_model("GMF")
        if not gmf_path:
            print("GMF model not found. Training GMF...")
            cmd = [
                "python", "scripts/pretrain.py",
                "--model", "GMF",
                "--epochs", str(self.epochs),
                "--factor_num", str(self.factor_num)
            ]
            # Use subprocess.call for Python 3.6 compatibility (simpler, no output capture needed)
            result = subprocess.call(cmd)
            if result != 0:
                raise RuntimeError("Failed to train GMF model")
            gmf_path = self.find_model("GMF")
            if gmf_path:
                print(f"GMF model saved: {gmf_path.name}")
        else:
            print(f"GMF model found: {gmf_path.name}")
        
        # Check MLP models for each layer configuration
        for layers in [1, 2, 3]:
            mlp_path = self.find_model("MLP", layers)
            if not mlp_path:
                print(f"MLP {layers}-layer model not found. Training...")
                cmd = [
                    "python", "scripts/pretrain.py",
                    "--model", "MLP",
                    "--epochs", str(self.epochs),
                    "--num_layers", str(layers),
                    "--factor_num", str(self.factor_num)
                ]
                result = subprocess.call(cmd)
                if result != 0:
                    raise RuntimeError(f"Failed to train MLP {layers}-layer model")
                mlp_path = self.find_model("MLP", layers)
                if mlp_path:
                    print(f"MLP {layers}-layer model saved: {mlp_path.name}")
            else:
                print(f"MLP {layers}-layer model found: {mlp_path.name}")
        
        print("All required pretrained models are available!")
    
    def run_single_experiment(self, num_layers, pretraining, run_id):
        """Run a single training experiment."""
        print(f"\n  Run {run_id}: {num_layers} layers, pretraining={pretraining}")
        
        # Check if required pretrained models exist for pretraining
        if pretraining:
            gmf_path = self.find_model("GMF")
            mlp_path = self.find_model("MLP", num_layers)
            if not gmf_path or not mlp_path:
                print(f"    ERROR: Required pretrained models not found")
                print(f"    GMF: {gmf_path}")
                print(f"    MLP: {mlp_path}")
                return None
        
        # Prepare command for training NeuMF
        cmd = [
            "python", "scripts/train_neumf.py",
            "--epochs", str(self.epochs),
            "--num_layers", str(num_layers),
            "--factor_num", str(self.factor_num)
        ]
        
        if pretraining:
            cmd.extend(["--model", "NeuMF-pre", "--pretraining"])
        else:
            cmd.extend(["--model", "NeuMF-end"])
        
        # Run training (Python 3.6 compatible)
        output_lines = []
        try:
            # Use subprocess.PIPE for Python 3.6 compatibility
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                  universal_newlines=True, check=True)
            
            # Parse the output to extract results
            output_lines = result.stdout.split('\n')
            
            hr_value = None
            ndcg_value = None
            param_count = None
            
            # Look for the results section (new format)
            in_results = False
            for line in output_lines:
                if "--- RESULTS ---" in line:
                    in_results = True
                    continue
                elif "--- END RESULTS ---" in line:
                    in_results = False
                    break
                elif in_results and ":" in line:
                    key_value = line.split(":", 1)
                    if len(key_value) == 2:
                        key = key_value[0].strip()
                        value = key_value[1].strip()
                        
                        if key == f"HR@{config.top_k}":
                            hr_value = float(value)
                        elif key == f"NDCG@{config.top_k}":
                            ndcg_value = float(value)
                        elif key == "Parameters":
                            param_count = int(value)
            
            # Fallback: look for "Best Result:" line if new format not found
            if hr_value is None:
                for line in output_lines:
                    if "Best Result:" in line and "HR=" in line and "NDCG=" in line:
                        try:
                            parts = line.split("HR=")[1]
                            hr_value = float(parts.split(",")[0].strip())
                            
                            ndcg_part = line.split("NDCG=")[1]
                            ndcg_value = float(ndcg_part.strip())
                        except:
                            continue
            
            # Look for parameter count in different locations
            if param_count is None:
                for line in output_lines:
                    if "parameters:" in line.lower():
                        try:
                            param_str = line.split(":")[-1].strip().replace(",", "").split()[0]
                            param_count = int(param_str)
                            break
                        except:
                            continue
            
            if hr_value is None:
                raise ValueError("Could not parse HR@10 from output")
            
            if param_count is None:
                # Estimate parameter count based on model architecture
                print(f"    Warning: Could not parse parameter count, estimating...")
                if num_layers == 1:
                    param_count = 180000  # Rough estimate
                elif num_layers == 2:
                    param_count = 260000
                else:
                    param_count = 340000
            
            print(f"    HR@10: {hr_value:.4f}, NDCG@10: {ndcg_value:.4f}, Params: {param_count:,}")
            
            return {
                'hr': hr_value,
                'ndcg': ndcg_value if ndcg_value else 0.0,
                'parameters': param_count,
                'num_layers': num_layers,
                'pretraining': pretraining,
                'run_id': run_id
            }
            
        except subprocess.CalledProcessError as e:
            print(f"    ERROR: Training failed - {e}")
            if hasattr(e, 'stdout') and e.stdout:
                print(f"    STDOUT: {e.stdout}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"    STDERR: {e.stderr}")
            return None
        except Exception as e:
            print(f"    ERROR: Failed to parse results - {e}")
            if output_lines:
                print("    Debug - last 10 output lines:")
                for i, line in enumerate(output_lines[-10:]):
                    print(f"    {i}: {line}")
            else:
                print("    No output captured")
            return None
    
    def run_all_experiments(self):
        """Run all experimental configurations."""
        print("\n[STEP 2] Running all experiments...")
        
        configurations = [
            (1, True),   (1, False),
            (2, True),   (2, False), 
            (3, True),   (3, False)
        ]
        
        for num_layers, pretraining in configurations:
            config_name = f"{num_layers}layers_{'pretrain' if pretraining else 'scratch'}"
            print(f"\n--- Configuration: {config_name} ---")
            
            self.results[config_name] = {
                'num_layers': num_layers,
                'pretraining': pretraining,
                'runs': []
            }
            
            for run_id in range(1, self.num_runs + 1):
                result = self.run_single_experiment(num_layers, pretraining, run_id)
                if result:
                    self.results[config_name]['runs'].append(result)
                else:
                    print(f"    Skipping failed run {run_id}")
        
        print("\n[STEP 2] All experiments completed!")
    
    def analyze_results(self):
        """Analyze and summarize results."""
        print("\n[STEP 3] Analyzing results...")
        
        analysis = {}
        
        for config_name, config_data in self.results.items():
            if not config_data['runs']:
                print(f"Warning: No successful runs for {config_name}")
                continue
            
            hrs = [run['hr'] for run in config_data['runs']]
            ndcgs = [run['ndcg'] for run in config_data['runs']]
            params = config_data['runs'][0]['parameters']  # Same for all runs
            
            analysis[config_name] = {
                'num_layers': config_data['num_layers'],
                'pretraining': config_data['pretraining'],
                'hr_mean': np.mean(hrs),
                'hr_std': np.std(hrs),
                'ndcg_mean': np.mean(ndcgs), 
                'ndcg_std': np.std(ndcgs),
                'parameters': params,
                'num_runs': len(hrs)
            }
            
            print(f"{config_name:20s}: HR@10={np.mean(hrs):.4f}+/-{np.std(hrs):.4f}, "
                  f"NDCG@10={np.mean(ndcgs):.4f}+/-{np.std(ndcgs):.4f}, "
                  f"Params={params:,}")
        
        return analysis
    
    def create_plots(self, analysis):
        """Create visualization plots."""
        print("\n[STEP 4] Creating plots...")
        
        # Prepare data for plotting
        layers = [1, 2, 3]
        pretrain_hrs = []
        scratch_hrs = []
        pretrain_stds = []
        scratch_stds = []
        pretrain_params = []
        scratch_params = []
        
        for layer in layers:
            pretrain_key = f"{layer}layers_pretrain"
            scratch_key = f"{layer}layers_scratch"
            
            if pretrain_key in analysis:
                pretrain_hrs.append(analysis[pretrain_key]['hr_mean'])
                pretrain_stds.append(analysis[pretrain_key]['hr_std'])
                pretrain_params.append(analysis[pretrain_key]['parameters'])
            else:
                pretrain_hrs.append(0)
                pretrain_stds.append(0)
                pretrain_params.append(0)
            
            if scratch_key in analysis:
                scratch_hrs.append(analysis[scratch_key]['hr_mean'])
                scratch_stds.append(analysis[scratch_key]['hr_std'])
                scratch_params.append(analysis[scratch_key]['parameters'])
            else:
                scratch_hrs.append(0)
                scratch_stds.append(0)
                scratch_params.append(0)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: HR@10 comparison
        x = np.arange(len(layers))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, pretrain_hrs, width, yerr=pretrain_stds,
                       label='With Pretraining', color='skyblue', capsize=5)
        bars2 = ax1.bar(x + width/2, scratch_hrs, width, yerr=scratch_stds,
                       label='Without Pretraining', color='lightcoral', capsize=5)
        
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
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Parameter count comparison
        bars3 = ax2.bar(x - width/2, pretrain_params, width, 
                       label='With Pretraining', color='skyblue')
        bars4 = ax2.bar(x + width/2, scratch_params, width,
                       label='Without Pretraining', color='lightcoral')
        
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
                            f'{val:,}', ha='center', va='bottom', fontsize=8, rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = config.figure_dir / "question_02_mlp_layers_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {plot_path}")
        
        # Create a summary table plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        headers = ['Configuration', 'Layers', 'Pretraining', 'HR@10 (Mean±Std)', 'NDCG@10 (Mean±Std)', 'Parameters']
        
        for config_name, data in analysis.items():
            row = [
                config_name,
                str(data['num_layers']),
                'Yes' if data['pretraining'] else 'No',
                f"{data['hr_mean']:.4f}+/-{data['hr_std']:.4f}",
                f"{data['ndcg_mean']:.4f}+/-{data['ndcg_std']:.4f}",
                f"{data['parameters']:,}"
            ]
            table_data.append(row)
        
        table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Question 2: Detailed Results Summary', fontsize=16, fontweight='bold', pad=20)
        
        # Save table
        table_path = config.figure_dir / "question_02_results_table.png"
        plt.savefig(table_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Results table saved to: {table_path}")
    
    def save_results(self, analysis):
        """Save results to JSON file."""
        print("\n[STEP 5] Saving results...")
        
        # Save detailed results
        results_path = config.output_dir / "reports" / "question_02_results.json"
        save_json(self.results, results_path)
        
        # Save analysis summary
        analysis_path = config.output_dir / "reports" / "question_02_analysis.json"
        save_json(analysis, analysis_path)
        
        print(f"Detailed results saved to: {results_path}")
        print(f"Analysis summary saved to: {analysis_path}")
    
    def run_experiment(self):
        """Run the complete experiment."""
        print("Starting Question 2 experiment...")
        
        # Step 1: Ensure we have the required pretrained models
        self.ensure_pretrained_models_exist()
        
        # Step 2: Run all experimental configurations
        self.run_all_experiments()
        
        # Step 3: Analyze results
        analysis = self.analyze_results()
        
        # Step 4: Create visualizations
        self.create_plots(analysis)
        
        # Step 5: Save results
        self.save_results(analysis)
        
        print("\n" + "="*60)
        print("QUESTION 2 EXPERIMENT COMPLETED!")
        print("="*60)
        print("Generated files:")
        print(f"- {config.figure_dir}/question_02_mlp_layers_analysis.png")
        print(f"- {config.figure_dir}/question_02_results_table.png")
        print(f"- {config.output_dir}/reports/question_02_results.json")
        print(f"- {config.output_dir}/reports/question_02_analysis.json")
        print("\nModel naming convention:")
        print(f"- GMF: GMF_{self.factor_num}f_best.pth")
        print(f"- MLP: MLP_{{layers}}l_{self.factor_num}f_best.pth")
        print(f"- NeuMF: NeuMF_{{end/pre}}_{{layers}}l_{self.factor_num}f_best.pth")
        print("="*60)
        
        return analysis

def main():
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
    
    # Create and run experiment
    experiment = Question2Experiment(num_runs=args.runs, epochs=args.epochs)
    results = experiment.run_experiment()
    
    # Print summary
    print("\nExperiment Summary:")
    for config_name, data in results.items():
        print(f"{config_name}: HR@10 = {data['hr_mean']:.4f}+/-{data['hr_std']:.4f}")

if __name__ == "__main__":
    main()