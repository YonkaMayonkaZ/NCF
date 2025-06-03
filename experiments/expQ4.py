#!/usr/bin/env python3
"""
Question 4: Training curves analysis using Q2 experiment results.

Greek: Show in 3 figures how (i) training loss, (ii) HR@10 and (iii) NDCG@10 
are affected for each iteration/epoch when training the model.

This script analyzes the training histories from Q2 experiment results.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import config
from src.utils.io import ensure_dir, save_json

class Question4Experiment:
    def __init__(self):
        self.results = {}
        
        # Ensure output directories exist
        ensure_dir(config.figure_dir)
        ensure_dir(config.output_dir / "reports")
        
        print("="*60)
        print("QUESTION 4: TRAINING CURVES ANALYSIS FROM Q2 RESULTS")
        print("="*60)
        print("Analyzing training curves from Question 2 experiment results")
        print("Metrics: (i) Training Loss, (ii) HR@10, (iii) NDCG@10 vs Epochs")
        print("="*60)
    
    def load_q2_results(self):
        """Load results from Question 2 experiment."""
        print("\n[STEP 1] Loading Q2 experiment results...")
        
        q2_results_path = config.output_dir / "reports" / "question_02_results.json"
        
        if not q2_results_path.exists():
            print(f"ERROR: Q2 results not found at {q2_results_path}")
            print("Please run Question 2 experiment first:")
            print("python experiments/expQ2.py --runs 10 --epochs 20")
            return False
        
        try:
            with open(q2_results_path, 'r') as f:
                self.q2_data = json.load(f)
            
            print(f"Q2 results loaded from: {q2_results_path}")
            
            # Check what configurations are available
            print("\nAvailable configurations:")
            for config_name in self.q2_data.keys():
                num_runs = len(self.q2_data[config_name].get('runs', []))
                print(f"  - {config_name}: {num_runs} runs")
            
            return True
            
        except Exception as e:
            print(f"ERROR loading Q2 results: {e}")
            return False
    
    def extract_training_histories(self):
        """Extract and process training histories from Q2 data."""
        print("\n[STEP 2] Processing training histories...")
        
        self.training_curves = {}
        
        for config_name, config_data in self.q2_data.items():
            if 'runs' not in config_data or not config_data['runs']:
                print(f"  Warning: No runs found for {config_name}")
                continue
            
            histories = []
            for run in config_data['runs']:
                if 'history' in run:
                    histories.append(run['history'])
            
            if not histories:
                print(f"  Warning: No training histories found for {config_name}")
                continue
            
            # Process the histories
            processed = self.process_histories(histories, config_name)
            self.training_curves[config_name] = processed
            
            print(f"  Processed {config_name}: {len(histories)} runs, {len(processed['epochs'])} epochs")
    
    def process_histories(self, histories, config_name):
        """Process multiple training histories to get mean and std."""
        if not histories:
            return None
        
        # Get number of epochs (assume all runs have same number)
        num_epochs = len(histories[0])
        epochs = list(range(1, num_epochs + 1))
        
        # Extract metrics for each epoch across all runs
        losses = []
        hrs = []
        ndcgs = []
        
        for epoch_idx in range(num_epochs):
            epoch_losses = []
            epoch_hrs = []
            epoch_ndcgs = []
            
            for history in histories:
                if epoch_idx < len(history):
                    epoch_data = history[epoch_idx]
                    epoch_losses.append(epoch_data.get('loss', 0))
                    epoch_hrs.append(epoch_data.get('hr', 0))
                    epoch_ndcgs.append(epoch_data.get('ndcg', 0))
            
            losses.append(epoch_losses)
            hrs.append(epoch_hrs)
            ndcgs.append(epoch_ndcgs)
        
        # Calculate mean and std for each epoch
        loss_means = [np.mean(epoch_losses) for epoch_losses in losses]
        loss_stds = [np.std(epoch_losses) for epoch_losses in losses]
        
        hr_means = [np.mean(epoch_hrs) for epoch_hrs in hrs]
        hr_stds = [np.std(epoch_hrs) for epoch_hrs in hrs]
        
        ndcg_means = [np.mean(epoch_ndcgs) for epoch_ndcgs in ndcgs]
        ndcg_stds = [np.std(epoch_ndcgs) for epoch_ndcgs in ndcgs]
        
        return {
            'config_name': config_name,
            'epochs': epochs,
            'num_runs': len(histories),
            'loss': {'mean': loss_means, 'std': loss_stds},
            'hr': {'mean': hr_means, 'std': hr_stds},
            'ndcg': {'mean': ndcg_means, 'std': ndcg_stds}
        }
    
    def format_legend_label(self, config_name):
        """Convert config name to readable legend label."""
        if config_name == '1layers_pretrain':
            return '1 layer (pretraining)'
        elif config_name == '1layers_scratch':
            return '1 layer (scratch)'
        elif config_name == '2layers_pretrain':
            return '2 layers (pretraining)'
        elif config_name == '2layers_scratch':
            return '2 layers (scratch)'
        elif config_name == '3layers_pretrain':
            return '3 layers (pretraining)'
        elif config_name == '3layers_scratch':
            return '3 layers (scratch)'
        else:
            return config_name
    
    def create_training_curves_plots(self):
        """Create the three plots as required by Question 4."""
        print("\n[STEP 3] Creating training curves plots...")
        
        # Create the three subplots as shown in Figure 6 of the paper
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Colors for different configurations
        colors = {
            '1layers_pretrain': 'blue',
            '1layers_scratch': 'orange', 
            '2layers_pretrain': 'green',
            '2layers_scratch': 'red',
            '3layers_pretrain': 'purple',
            '3layers_scratch': 'brown'
        }
        
        # Line styles for distinction
        line_styles = {
            '1layers_pretrain': '-',
            '1layers_scratch': '--',
            '2layers_pretrain': '-',
            '2layers_scratch': '--',
            '3layers_pretrain': '-',
            '3layers_scratch': '--'
        }
        
        # Plot 1: Training Loss vs Epoch
        for config_name, curve_data in self.training_curves.items():
            epochs = curve_data['epochs']
            loss_mean = curve_data['loss']['mean']
            loss_std = curve_data['loss']['std']
            
            color = colors.get(config_name, 'gray')
            linestyle = line_styles.get(config_name, '-')
            label = self.format_legend_label(config_name)
            
            ax1.plot(epochs, loss_mean, color=color, linestyle=linestyle, linewidth=2, label=label)
            ax1.fill_between(epochs, 
                           [m - s for m, s in zip(loss_mean, loss_std)],
                           [m + s for m, s in zip(loss_mean, loss_std)],
                           color=color, alpha=0.2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('(i) Training Loss vs Epoch')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: HR@10 vs Epoch
        for config_name, curve_data in self.training_curves.items():
            epochs = curve_data['epochs']
            hr_mean = curve_data['hr']['mean']
            hr_std = curve_data['hr']['std']
            
            color = colors.get(config_name, 'gray')
            linestyle = line_styles.get(config_name, '-')
            label = self.format_legend_label(config_name)
            
            ax2.plot(epochs, hr_mean, color=color, linestyle=linestyle, linewidth=2, label=label)
            ax2.fill_between(epochs,
                           [m - s for m, s in zip(hr_mean, hr_std)],
                           [m + s for m, s in zip(hr_mean, hr_std)],
                           color=color, alpha=0.2)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('HR@10')
        ax2.set_title('(ii) HR@10 vs Epoch')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: NDCG@10 vs Epoch
        for config_name, curve_data in self.training_curves.items():
            epochs = curve_data['epochs']
            ndcg_mean = curve_data['ndcg']['mean']
            ndcg_std = curve_data['ndcg']['std']
            
            color = colors.get(config_name, 'gray')
            linestyle = line_styles.get(config_name, '-')
            label = self.format_legend_label(config_name)
            
            ax3.plot(epochs, ndcg_mean, color=color, linestyle=linestyle, linewidth=2, label=label)
            ax3.fill_between(epochs,
                           [m - s for m, s in zip(ndcg_mean, ndcg_std)],
                           [m + s for m, s in zip(ndcg_mean, ndcg_std)],
                           color=color, alpha=0.2)
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('NDCG@10')
        ax3.set_title('(iii) NDCG@10 vs Epoch')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = config.figure_dir / "question_04_training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves plot saved to: {plot_path}")
    
    def create_individual_plots(self):
        """Create individual plots for each metric (better readability)."""
        print("\n[STEP 4] Creating individual metric plots...")
        
        metrics = [
            ('loss', 'Training Loss', 'Training Loss'),
            ('hr', 'HR@10', 'Hit Ratio @ 10'),
            ('ndcg', 'NDCG@10', 'NDCG @ 10')
        ]
        
        colors = {
            '1layers_pretrain': 'blue',
            '1layers_scratch': 'orange', 
            '2layers_pretrain': 'green',
            '2layers_scratch': 'red',
            '3layers_pretrain': 'purple',
            '3layers_scratch': 'brown'
        }
        
        line_styles = {
            '1layers_pretrain': '-',
            '1layers_scratch': '--',
            '2layers_pretrain': '-',
            '2layers_scratch': '--',
            '3layers_pretrain': '-',
            '3layers_scratch': '--'
        }
        
        for metric_key, metric_name, ylabel in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for config_name, curve_data in self.training_curves.items():
                epochs = curve_data['epochs']
                values_mean = curve_data[metric_key]['mean']
                values_std = curve_data[metric_key]['std']
                
                color = colors.get(config_name, 'gray')
                linestyle = line_styles.get(config_name, '-')
                label = self.format_legend_label(config_name)
                
                ax.plot(epochs, values_mean, color=color, linestyle=linestyle, 
                       linewidth=2, label=label)
                ax.fill_between(epochs,
                               [m - s for m, s in zip(values_mean, values_std)],
                               [m + s for m, s in zip(values_mean, values_std)],
                               color=color, alpha=0.2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.set_title(f'Question 4: {metric_name} vs Epoch')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save individual plot
            plot_path = config.figure_dir / f"question_04_{metric_key}_curves.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"{metric_name} plot saved to: {plot_path}")
    
    def analyze_convergence(self):
        """Analyze convergence patterns from the training curves."""
        print("\n[STEP 5] Analyzing convergence patterns...")
        
        convergence_analysis = {}
        
        for config_name, curve_data in self.training_curves.items():
            epochs = curve_data['epochs']
            hr_mean = curve_data['hr']['mean']
            loss_mean = curve_data['loss']['mean']
            
            # Find best HR@10 and when it occurred
            best_hr = max(hr_mean)
            best_hr_epoch = epochs[hr_mean.index(best_hr)]
            
            # Find lowest loss and when it occurred
            lowest_loss = min(loss_mean)
            lowest_loss_epoch = epochs[loss_mean.index(lowest_loss)]
            
            # Check for early convergence (no improvement in last 5 epochs)
            last_5_hr = hr_mean[-5:] if len(hr_mean) >= 5 else hr_mean
            hr_improvement = max(last_5_hr) - min(last_5_hr)
            early_convergence = hr_improvement < 0.001  # Very small improvement
            
            convergence_analysis[config_name] = {
                'best_hr': best_hr,
                'best_hr_epoch': best_hr_epoch,
                'lowest_loss': lowest_loss,
                'lowest_loss_epoch': lowest_loss_epoch,
                'early_convergence': early_convergence,
                'final_hr': hr_mean[-1],
                'final_loss': loss_mean[-1]
            }
            
            print(f"  {config_name}:")
            print(f"    Best HR@10: {best_hr:.4f} at epoch {best_hr_epoch}")
            print(f"    Lowest loss: {lowest_loss:.4f} at epoch {lowest_loss_epoch}")
            print(f"    Early convergence: {early_convergence}")
        
        return convergence_analysis
    
    def save_results(self, convergence_analysis):
        """Save results to JSON file."""
        print("\n[STEP 6] Saving results...")
        
        # Prepare results
        results = {
            'training_curves': self.training_curves,
            'convergence_analysis': convergence_analysis,
            'experiment_info': {
                'description': 'Training curves analysis from Q2 experiment',
                'metrics_analyzed': ['training_loss', 'hr@10', 'ndcg@10'],
                'configurations': list(self.training_curves.keys())
            }
        }
        
        # Save detailed results
        results_path = config.output_dir / "reports" / "question_04_results.json"
        save_json(results, results_path)
        
        print(f"Results saved to: {results_path}")
    
    def run_experiment(self):
        """Run the complete Question 4 experiment."""
        print("Starting Question 4 experiment...")
        
        # Step 1: Load Q2 results
        if not self.load_q2_results():
            return None
        
        # Step 2: Extract training histories
        self.extract_training_histories()
        
        if not self.training_curves:
            print("ERROR: No training curves found in Q2 results")
            return None
        
        # Step 3: Create plots
        self.create_training_curves_plots()
        
        # Step 4: Create individual plots
        self.create_individual_plots()
        
        # Step 5: Analyze convergence
        convergence_analysis = self.analyze_convergence()
        
        # Step 6: Save results
        self.save_results(convergence_analysis)
        
        print("\n" + "="*60)
        print("QUESTION 4 EXPERIMENT COMPLETED!")
        print("="*60)
        print("Generated files:")
        print(f"- {config.figure_dir}/question_04_training_curves.png")
        print(f"- {config.figure_dir}/question_04_loss_curves.png")
        print(f"- {config.figure_dir}/question_04_hr_curves.png")
        print(f"- {config.figure_dir}/question_04_ndcg_curves.png")
        print(f"- {config.output_dir}/reports/question_04_results.json")
        print("="*60)
        
        return self.training_curves

def main():
    print("Question 4: Training Curves Analysis")
    print("Analyzing training histories from Q2 experiment results")
    
    # Create and run experiment
    experiment = Question4Experiment()
    results = experiment.run_experiment()
    
    if results:
        print("\nSUCCESS: Training curves analysis completed!")
        print("Check the figures directory for the three required plots.")
    else:
        print("\nFAILED: Could not complete analysis.")
        print("Make sure Q2 experiment has been run first.")

if __name__ == "__main__":
    main()