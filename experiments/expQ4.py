#!/usr/bin/env python3
"""
Question 4: Training curves analysis using Q2 experiment results.
Optimized version using project utilities.

Greek: Show in 3 figures how (i) training loss, (ii) HR@10 and (iii) NDCG@10 
are affected for each iteration/epoch when training the model.

This script analyzes the training histories from Q2 experiment results.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Use project utilities
from src.utils.config import config
from src.utils.io import ensure_dir, save_json, load_json
from src.utils.logging import get_experiment_logger
from src.utils.visualization import plot_training_metrics, plot_comparative_metrics

class Question4Experiment:
    def __init__(self):
        """Initialize Q4 experiment with proper logging and config."""
        self.results = {}
        self.training_curves = {}
        self.logger = get_experiment_logger("question_4")
        
        # Ensure output directories exist using utility
        ensure_dir(config.figure_dir)
        ensure_dir(config.output_dir / "reports")
        
        self.logger.info("="*60)
        self.logger.info("QUESTION 4: TRAINING CURVES ANALYSIS FROM Q2 RESULTS")
        self.logger.info("="*60)
        self.logger.info("Analyzing training curves from Question 2 experiment results")
        self.logger.info("Metrics: (i) Training Loss, (ii) HR@10, (iii) NDCG@10 vs Epochs")
        self.logger.info("="*60)
    
    def load_q2_results(self):
        """Load results from Question 2 experiment using io utilities."""
        self.logger.info("Loading Q2 experiment results...")
        
        q2_results_path = config.output_dir / "reports" / "question_02_results.json"
        
        if not q2_results_path.exists():
            self.logger.error(f"Q2 results not found at {q2_results_path}")
            self.logger.error("Please run Question 2 experiment first:")
            self.logger.error("python experiments/expQ2.py --runs 10 --epochs 20")
            return False
        
        try:
            # Use utility function instead of manual JSON loading
            self.q2_data = load_json(q2_results_path)
            
            self.logger.info(f"Q2 results loaded successfully from: {q2_results_path}")
            
            # Log available configurations
            self.logger.info("Available configurations:")
            for config_name in self.q2_data.keys():
                num_runs = len(self.q2_data[config_name].get('runs', []))
                self.logger.info(f"  - {config_name}: {num_runs} runs")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading Q2 results: {e}")
            return False
    
    def extract_training_histories(self):
        """Extract and process training histories from Q2 data."""
        self.logger.info("Processing training histories...")
        
        for config_name, config_data in self.q2_data.items():
            if 'runs' not in config_data or not config_data['runs']:
                self.logger.warning(f"No runs found for {config_name}")
                continue
            
            histories = []
            for run in config_data['runs']:
                if 'history' in run:
                    histories.append(run['history'])
            
            if not histories:
                self.logger.warning(f"No training histories found for {config_name}")
                continue
            
            # Process the histories using existing aggregation logic
            processed = self.process_histories(histories, config_name)
            if processed:
                self.training_curves[config_name] = processed
                self.logger.info(f"Processed {config_name}: {len(histories)} runs, {len(processed['epochs'])} epochs")
    
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
        label_mapping = {
            '1layers_pretrain': '1 layer (pretraining)',
            '1layers_scratch': '1 layer (scratch)',
            '2layers_pretrain': '2 layers (pretraining)',
            '2layers_scratch': '2 layers (scratch)',
            '3layers_pretrain': '3 layers (pretraining)',
            '3layers_scratch': '3 layers (scratch)'
        }
        return label_mapping.get(config_name, config_name)
    
    def create_training_curves_plots(self):
        """Create the three plots as required by Question 4."""
        self.logger.info("Creating training curves plots...")
        
        # Create the three subplots as shown in Figure 6 of the paper
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Color and line style configuration
        plot_config = self._get_plot_configuration()
        
        # Plot 1: Training Loss vs Epoch
        self._plot_metric_curves(ax1, 'loss', 'Training Loss', '(i) Training Loss vs Epoch', plot_config)
        
        # Plot 2: HR@10 vs Epoch
        self._plot_metric_curves(ax2, 'hr', 'HR@10', '(ii) HR@10 vs Epoch', plot_config)
        
        # Plot 3: NDCG@10 vs Epoch
        self._plot_metric_curves(ax3, 'ndcg', 'NDCG@10', '(iii) NDCG@10 vs Epoch', plot_config)
        
        plt.tight_layout()
        
        # Save plot using config paths
        plot_path = config.figure_dir / "question_04_training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training curves plot saved to: {plot_path}")
    
    def _get_plot_configuration(self):
        """Get consistent plot configuration for all metrics."""
        return {
            'colors': {
                '1layers_pretrain': 'blue',
                '1layers_scratch': 'orange', 
                '2layers_pretrain': 'green',
                '2layers_scratch': 'red',
                '3layers_pretrain': 'purple',
                '3layers_scratch': 'brown'
            },
            'line_styles': {
                '1layers_pretrain': '-',
                '1layers_scratch': '--',
                '2layers_pretrain': '-',
                '2layers_scratch': '--',
                '3layers_pretrain': '-',
                '3layers_scratch': '--'
            }
        }
    
    def _plot_metric_curves(self, ax, metric_key, ylabel, title, plot_config):
        """Plot curves for a specific metric on given axes."""
        for config_name, curve_data in self.training_curves.items():
            epochs = curve_data['epochs']
            values_mean = curve_data[metric_key]['mean']
            values_std = curve_data[metric_key]['std']
            
            color = plot_config['colors'].get(config_name, 'gray')
            linestyle = plot_config['line_styles'].get(config_name, '-')
            label = self.format_legend_label(config_name)
            
            ax.plot(epochs, values_mean, color=color, linestyle=linestyle, 
                   linewidth=2, label=label)
            ax.fill_between(epochs, 
                           [m - s for m, s in zip(values_mean, values_std)],
                           [m + s for m, s in zip(values_mean, values_std)],
                           color=color, alpha=0.2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def create_individual_plots_using_utils(self):
        """Create individual plots using existing visualization utilities."""
        self.logger.info("Creating individual metric plots using visualization utils...")
        
        # Prepare data in format expected by visualization utilities
        for metric_key, metric_name, ylabel in [
            ('loss', 'Training Loss', 'Training Loss'),
            ('hr', 'HR@10', 'Hit Ratio @ 10'),
            ('ndcg', 'NDCG@10', 'NDCG @ 10')
        ]:
            # Convert our data format to the format expected by visualization utils
            run_histories_list = []
            model_names = []
            
            for config_name, curve_data in self.training_curves.items():
                # Convert back to run format for visualization utility
                epochs = curve_data['epochs']
                mean_values = curve_data[metric_key]['mean']
                
                # Create synthetic runs for visualization (using means as single runs)
                synthetic_history = []
                for epoch, value in zip(epochs, mean_values):
                    synthetic_history.append({
                        'epoch': epoch,
                        metric_key: value
                    })
                
                run_histories_list.append([synthetic_history])
                model_names.append(self.format_legend_label(config_name))
            
            # Use existing visualization utility
            if run_histories_list:  # Only if we have data
                output_path = config.figure_dir / f"question_04_{metric_key}_curves.png"
                try:
                    plot_comparative_metrics(
                        run_histories_list=run_histories_list,
                        model_names=model_names,
                        output_path=output_path,
                        metrics=[metric_key],
                        metric_labels={metric_key: ylabel}
                    )
                    self.logger.info(f"{metric_name} plot saved using utils")
                except Exception as e:
                    self.logger.warning(f"Could not use visualization utils for {metric_key}: {e}")
                    # Fallback to manual plotting
                    self._create_individual_plot_manual(metric_key, metric_name, ylabel)
    
    def _create_individual_plot_manual(self, metric_key, metric_name, ylabel):
        """Fallback manual plotting if utils don't work."""
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_config = self._get_plot_configuration()
        
        self._plot_metric_curves(ax, metric_key, ylabel, f'Question 4: {metric_name} vs Epoch', plot_config)
        
        plot_path = config.figure_dir / f"question_04_{metric_key}_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"{metric_name} plot saved to: {plot_path}")
    
    def analyze_convergence(self):
        """Analyze convergence patterns from the training curves."""
        self.logger.info("Analyzing convergence patterns...")
        
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
            
            self.logger.info(f"  {config_name}:")
            self.logger.info(f"    Best HR@10: {best_hr:.4f} at epoch {best_hr_epoch}")
            self.logger.info(f"    Lowest loss: {lowest_loss:.4f} at epoch {lowest_loss_epoch}")
            self.logger.info(f"    Early convergence: {early_convergence}")
        
        return convergence_analysis
    
    def save_results(self, convergence_analysis):
        """Save results using io utilities."""
        self.logger.info("Saving results using io utilities...")
        
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
        
        # Save detailed results using utility
        results_path = config.output_dir / "reports" / "question_04_results.json"
        save_json(results, results_path)
        
        # Save a simplified summary for easy reading
        summary = {
            'convergence_summary': convergence_analysis,
            'total_configurations': len(self.training_curves),
            'metrics_count': 3,
            'generated_plots': [
                'question_04_training_curves.png',
                'question_04_loss_curves.png', 
                'question_04_hr_curves.png',
                'question_04_ndcg_curves.png'
            ]
        }
        
        summary_path = config.output_dir / "reports" / "question_04_summary.json"
        save_json(summary, summary_path)
        
        self.logger.info(f"Detailed results saved to: {results_path}")
        self.logger.info(f"Summary saved to: {summary_path}")
    
    def run_experiment(self):
        """Run the complete Question 4 experiment."""
        self.logger.info("Starting Question 4 experiment...")
        
        # Step 1: Load Q2 results
        if not self.load_q2_results():
            return None
        
        # Step 2: Extract training histories
        self.extract_training_histories()
        
        if not self.training_curves:
            self.logger.error("No training curves found in Q2 results")
            return None
        
        # Step 3: Create main plots
        self.create_training_curves_plots()
        
        # Step 4: Create individual plots using utilities
        self.create_individual_plots_using_utils()
        
        # Step 5: Analyze convergence
        convergence_analysis = self.analyze_convergence()
        
        # Step 6: Save results using utilities
        self.save_results(convergence_analysis)
        
        self.logger.info("="*60)
        self.logger.info("QUESTION 4 EXPERIMENT COMPLETED!")
        self.logger.info("="*60)
        self.logger.info("Generated files:")
        self.logger.info(f"- {config.figure_dir}/question_04_training_curves.png")
        self.logger.info(f"- {config.figure_dir}/question_04_loss_curves.png")
        self.logger.info(f"- {config.figure_dir}/question_04_hr_curves.png")
        self.logger.info(f"- {config.figure_dir}/question_04_ndcg_curves.png")
        self.logger.info(f"- {config.output_dir}/reports/question_04_results.json")
        self.logger.info(f"- {config.output_dir}/reports/question_04_summary.json")
        self.logger.info("="*60)
        
        return self.training_curves

def main():
    """Main function with proper error handling and logging."""
    print("Question 4: Training Curves Analysis")
    print("Analyzing training histories from Q2 experiment results")
    
    try:
        # Create and run experiment
        experiment = Question4Experiment()
        results = experiment.run_experiment()
        
        if results:
            print("\n✅ SUCCESS: Training curves analysis completed!")
            print("📊 Check the figures directory for the three required plots.")
            print("📝 Check the reports directory for detailed analysis.")
        else:
            print("\n❌ FAILED: Could not complete analysis.")
            print("💡 Make sure Q2 experiment has been run first:")
            print("   python experiments/expQ2.py --runs 10 --epochs 20")
    
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        print("🔍 Check the logs for more details.")
        raise

if __name__ == "__main__":
    main()