#!/usr/bin/env python3
"""
Factor Number Experiment: Finding optimal embedding size for NCF models.
Simple script that properly uses existing project utilities.
"""

import os
import sys
import subprocess
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import config
from src.utils.logging import get_experiment_logger
from src.utils.io import save_json, ensure_dir
from src.utils.visualization import plot_comparative_metrics, save_run_history

class FactorExperiment:
    def __init__(self, num_runs=10, epochs=20):
        self.num_runs = num_runs
        self.epochs = epochs
        self.factor_configs = [8, 16, 32, 64]  # From NCF paper
        self.num_layers = 3
        self.results = {}
        
        # Use existing logging utility
        self.logger = get_experiment_logger("factor_experiment")
        
        # Ensure directories exist using io utils
        ensure_dir(config.figure_dir)
        ensure_dir(config.model_dir)
        
        self.logger.info(f"Factor experiment: {num_runs} runs x {len(self.factor_configs)} factors")
    
    def run_single_experiment(self, model_type, factor_num, run_id):
        """Run a single training experiment."""
        self.logger.info(f"Running {model_type} with {factor_num} factors (run {run_id})")
        
        # Build command
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
        
        try:
            # Execute training
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  universal_newlines=True, check=True)
            
            # Parse results with robust parsing
            parsed = self._parse_output(result.stdout)
            if parsed:
                self.logger.info(f"  Result: HR@10={parsed['hr']:.4f}, Params={parsed['parameters']:,}")
                return parsed
            else:
                self.logger.warning(f"  Failed to parse results for {model_type}_{factor_num}f run {run_id}")
                return None
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"  Training failed: {e}")
            return None
    
    def _parse_output(self, output):
        """Robust parsing of training script output."""
        lines = output.split('\n')
        hr_value = None
        ndcg_value = None
        param_count = None
        
        # Strategy 1: Look for --- RESULTS --- section
        in_results = False
        for line in lines:
            if "--- RESULTS ---" in line:
                in_results = True
                continue
            elif "--- END RESULTS ---" in line:
                break
            elif in_results and ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                
                if key == f"HR@{config.top_k}":
                    hr_value = float(value)
                elif key == f"NDCG@{config.top_k}":
                    ndcg_value = float(value)
                elif key == "Parameters":
                    param_count = int(value.replace(',', ''))
        
        # Strategy 2: Look for Final Results section (pretrain.py format)
        if hr_value is None:
            in_final = False
            for line in lines:
                if "Final Results:" in line:
                    in_final = True
                    continue
                elif in_final and ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == f"HR@{config.top_k}":
                        hr_value = float(value)
                    elif key == f"NDCG@{config.top_k}":
                        ndcg_value = float(value)
                    elif key == "Parameters":
                        param_count = int(value.replace(',', ''))
                elif in_final and line.strip() == "":
                    break
        
        # Return parsed results
        if hr_value is not None:
            return {
                'hr': hr_value,
                'ndcg': ndcg_value or 0.0,
                'parameters': param_count or 0
            }
        return None
    
    def run_all_experiments(self):
        """Run all factor experiments."""
        model_types = ["GMF", "MLP", "NeuMF-end"]
        
        for model_type in model_types:
            self.logger.info(f"\n=== Testing {model_type} ===")
            
            for factor_num in self.factor_configs:
                config_name = f"{model_type}_{factor_num}f"
                self.results[config_name] = {
                    'model_type': model_type,
                    'factor_num': factor_num,
                    'runs': []
                }
                
                for run_id in range(1, self.num_runs + 1):
                    result = self.run_single_experiment(model_type, factor_num, run_id)
                    if result:
                        self.results[config_name]['runs'].append(result)
        
        self.logger.info("\nAll experiments completed!")
    
    def analyze_results(self):
        """Analyze results and compute statistics."""
        analysis = {}
        
        for config_name, config_data in self.results.items():
            if not config_data['runs']:
                continue
            
            hrs = [run['hr'] for run in config_data['runs']]
            ndcgs = [run['ndcg'] for run in config_data['runs']]
            params = config_data['runs'][0]['parameters'] if config_data['runs'] else 0
            
            analysis[config_name] = {
                'model_type': config_data['model_type'],
                'factor_num': config_data['factor_num'],
                'hr_mean': np.mean(hrs),
                'hr_std': np.std(hrs),
                'ndcg_mean': np.mean(ndcgs),
                'ndcg_std': np.std(ndcgs),
                'parameters': params,
                'num_runs': len(hrs)
            }
            
            self.logger.info(f"{config_name}: HR@10={np.mean(hrs):.4f}±{np.std(hrs):.4f}")
        
        return analysis
    
    def create_plots(self, analysis):
        """Create plots using existing visualization utilities."""
        self.logger.info("Creating visualization plots...")
        
        # Group by model type and create comparison plots
        for model_type in ["GMF", "MLP", "NeuMF-end"]:
            model_configs = {k: v for k, v in analysis.items() 
                           if v['model_type'] == model_type}
            
            if not model_configs:
                continue
            
            # Convert to format expected by visualization utils
            run_histories_list = []
            model_names = []
            
            for config_name, data in sorted(model_configs.items(), key=lambda x: x[1]['factor_num']):
                # Create dummy history for visualization utils
                histories = []
                for i in range(data['num_runs']):
                    history = [{
                        'epoch': 1,
                        'hr': data['hr_mean'],  # Use mean for visualization
                        'ndcg': data['ndcg_mean'],
                        'loss': 0.1
                    }]
                    histories.append(history)
                
                run_histories_list.append(histories)
                model_names.append(f"{data['factor_num']} factors")
            
            if run_histories_list:
                # Use existing plot_comparative_metrics utility
                output_path = config.figure_dir / f"factor_comparison_{model_type.lower()}.png"
                try:
                    plot_comparative_metrics(
                        run_histories_list=run_histories_list,
                        model_names=model_names,
                        output_path=output_path,
                        metrics=["hr", "ndcg"],
                        metric_labels={"hr": "HR@10", "ndcg": "NDCG@10"}
                    )
                    self.logger.info(f"Saved comparison plot: {output_path}")
                except Exception as e:
                    self.logger.warning(f"Could not create plot for {model_type}: {e}")
    
    def save_results(self, analysis):
        """Save results using existing io utilities."""
        self.logger.info("Saving results...")
        
        # Save raw results using save_json utility
        results_path = config.output_dir / "reports" / "factor_experiment_results.json"
        save_json(self.results, results_path)
        
        # Save analysis using save_json utility  
        analysis_path = config.output_dir / "reports" / "factor_experiment_analysis.json"
        save_json(analysis, analysis_path)
        
        self.logger.info(f"Results saved to: {results_path}")
        self.logger.info(f"Analysis saved to: {analysis_path}")
    
    def run_experiment(self):
        """Run the complete factor experiment."""
        self.logger.info("Starting factor number experiment...")
        
        # Run experiments
        self.run_all_experiments()
        
        # Analyze results
        analysis = self.analyze_results()
        
        if not analysis:
            self.logger.error("No successful experiments!")
            return None
        
        # Create plots using utils
        self.create_plots(analysis)
        
        # Save results using utils
        self.save_results(analysis)
        
        # Log best results
        if analysis:
            best_config = max(analysis.items(), key=lambda x: x[1]['hr_mean'])
            self.logger.info(f"\nBest configuration: {best_config[0]}")
            self.logger.info(f"Best HR@10: {best_config[1]['hr_mean']:.4f} ± {best_config[1]['hr_std']:.4f}")
        
        return analysis

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Factor Number Experiment')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs per configuration')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs per run')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    
    args = parser.parse_args()
    
    if args.quick:
        print("Running in quick mode...")
        args.runs = 2
        args.epochs = 5
    
    print(f"Factor Number Experiment: {args.runs} runs x 4 factors x 3 models")
    
    try:
        experiment = FactorExperiment(num_runs=args.runs, epochs=args.epochs)
        results = experiment.run_experiment()
        
        if results:
            print("\n[SUCCESS] Factor experiment completed!")
            print("Check results/figures/ and results/reports/ for outputs.")
        else:
            print("\n[FAILED] Factor experiment failed.")
    
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Experiment interrupted.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise

if __name__ == "__main__":
    main()