#!/usr/bin/env python3
"""
Question 3: Parameter count analysis for NeuMF with different MLP layers.

Greek: "Δείξτε πώς επηρεάζεται ο αριθμός των παραμέτρων (weight parameters), 
μεταβάλλοντας τα MLP layers από 1 έως 3 με βήμα 1 για NeuMF χωρίς pretraining"

Translation: "Show how the number of parameters (weight parameters) is affected 
by varying MLP layers from 1 to 3 in steps of 1 for NeuMF without pretraining"
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ncf.models import NCF
from src.utils.config import config
from src.utils.io import ensure_dir, save_json

class Question3Experiment:
    def __init__(self):
        self.results = {}
        self.factor_num = config.factor_num  # 32 by default
        self.user_num = config.user_num      # 944
        self.item_num = config.item_num      # 1683
        
        # Ensure output directories exist
        ensure_dir(config.figure_dir)
        ensure_dir(config.output_dir / "reports")
        
        print("="*60)
        print("QUESTION 3: PARAMETER COUNT ANALYSIS FOR NeuMF")
        print("="*60)
        print(f"Configuration:")
        print(f"- Model: NeuMF-end (without pretraining)")
        print(f"- Layer configurations: 1, 2, 3")
        print(f"- Factor number: {self.factor_num}")
        print(f"- Users: {self.user_num}, Items: {self.item_num}")
        print("="*60)
    
    def count_parameters(self, model):
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def analyze_model_parameters(self, num_layers):
        """Detailed parameter analysis for a specific layer configuration."""
        print(f"\n--- Analyzing {num_layers} MLP layers ---")
        
        # Create NeuMF-end model (without pretraining)
        model = NCF(self.user_num, self.item_num, self.factor_num, num_layers, 0.0, "NeuMF-end")
        
        total_params = 0
        param_breakdown = {}
        
        print(f"Parameter breakdown:")
        
        for name, param in model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            param_breakdown[name] = {
                'count': param_count,
                'shape': list(param.shape)
            }
            print(f"  {name:25s}: {param_count:8,} parameters {list(param.shape)}")
        
        print(f"\nTotal parameters: {total_params:,}")
        
        # Manual calculation verification
        print(f"\nManual calculation verification:")
        
        # GMF embeddings
        gmf_user_params = self.user_num * self.factor_num
        gmf_item_params = self.item_num * self.factor_num
        gmf_total = gmf_user_params + gmf_item_params
        print(f"  GMF embeddings: {self.user_num} x {self.factor_num} + {self.item_num} x {self.factor_num} = {gmf_total:,}")
        
        # MLP embeddings
        mlp_factor = self.factor_num * (2 ** (num_layers - 1))
        mlp_user_params = self.user_num * mlp_factor
        mlp_item_params = self.item_num * mlp_factor
        mlp_embed_total = mlp_user_params + mlp_item_params
        print(f"  MLP embeddings: {self.user_num} x {mlp_factor} + {self.item_num} x {mlp_factor} = {mlp_embed_total:,}")
        
        # MLP layers
        input_size = self.factor_num * (2 ** num_layers)
        mlp_layers_total = 0
        for i in range(num_layers):
            output_size = input_size // 2
            layer_params = input_size * output_size + output_size  # weights + bias
            mlp_layers_total += layer_params
            print(f"  MLP layer {i+1}: {input_size} -> {output_size} = {layer_params:,} parameters")
            input_size = output_size
        
        # Prediction layer
        pred_input = self.factor_num * 2  # GMF output + MLP output
        pred_params = pred_input * 1 + 1  # weights + bias
        print(f"  Prediction layer: {pred_input} -> 1 = {pred_params:,} parameters")
        
        # Verification
        manual_total = gmf_total + mlp_embed_total + mlp_layers_total + pred_params
        print(f"\nVerification: {manual_total:,} (manual) vs {total_params:,} (PyTorch)")
        
        if manual_total == total_params:
            print("[OK] Parameter count verified!")
        else:
            print("[ERROR] Parameter count mismatch!")
        
        return {
            'num_layers': num_layers,
            'total_parameters': total_params,
            'breakdown': {
                'gmf_embeddings': gmf_total,
                'mlp_embeddings': mlp_embed_total,
                'mlp_layers': mlp_layers_total,
                'prediction_layer': pred_params
            },
            'detailed_breakdown': param_breakdown,
            'manual_verification': manual_total
        }
    
    def run_analysis(self):
        """Run parameter analysis for all layer configurations."""
        print("\n[ANALYSIS] Running parameter analysis for 1, 2, 3 layers...")
        
        for num_layers in [1, 2, 3]:
            result = self.analyze_model_parameters(num_layers)
            self.results[f"{num_layers}layers"] = result
        
        print("\n[ANALYSIS] Parameter analysis completed!")
    
    def create_plots(self):
        """Create visualization plots for parameter analysis."""
        print("\n[PLOTS] Creating visualization plots...")
        
        # Extract data for plotting
        layers = [1, 2, 3]
        total_params = [self.results[f"{l}layers"]['total_parameters'] for l in layers]
        
        # Extract breakdown data
        gmf_params = [self.results[f"{l}layers"]['breakdown']['gmf_embeddings'] for l in layers]
        mlp_embed_params = [self.results[f"{l}layers"]['breakdown']['mlp_embeddings'] for l in layers]
        mlp_layer_params = [self.results[f"{l}layers"]['breakdown']['mlp_layers'] for l in layers]
        pred_params = [self.results[f"{l}layers"]['breakdown']['prediction_layer'] for l in layers]
        
        # Create figure with multiple subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Total parameter count
        ax1.plot(layers, total_params, 'bo-', linewidth=3, markersize=8)
        ax1.set_xlabel('Number of MLP Layers')
        ax1.set_ylabel('Total Parameters')
        ax1.set_title('Question 3: Total Parameters vs MLP Layers (NeuMF without pretraining)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(layers)
        
        # Add value labels on points
        for i, (x, y) in enumerate(zip(layers, total_params)):
            ax1.annotate(f'{y:,}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        
        # Plot 2: Stacked bar chart showing parameter breakdown
        width = 0.6
        x = np.array(layers)
        
        p1 = ax2.bar(x, gmf_params, width, label='GMF Embeddings', color='skyblue')
        p2 = ax2.bar(x, mlp_embed_params, width, bottom=gmf_params, label='MLP Embeddings', color='lightcoral')
        bottom2 = np.array(gmf_params) + np.array(mlp_embed_params)
        p3 = ax2.bar(x, mlp_layer_params, width, bottom=bottom2, label='MLP Layers', color='lightgreen')
        bottom3 = bottom2 + np.array(mlp_layer_params)
        p4 = ax2.bar(x, pred_params, width, bottom=bottom3, label='Prediction Layer', color='gold')
        
        ax2.set_xlabel('Number of MLP Layers')
        ax2.set_ylabel('Number of Parameters')
        ax2.set_title('Parameter Breakdown by Component')
        ax2.legend()
        ax2.set_xticks(layers)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: MLP layers parameters only
        ax3.plot(layers, mlp_layer_params, 'go-', linewidth=3, markersize=8)
        ax3.set_xlabel('Number of MLP Layers')
        ax3.set_ylabel('MLP Layers Parameters')
        ax3.set_title('MLP Layers Parameters Growth')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(layers)
        
        # Add value labels
        for i, (x, y) in enumerate(zip(layers, mlp_layer_params)):
            ax3.annotate(f'{y:,}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        
        # Plot 4: Parameter growth rate
        param_increases = [0]  # No increase for first layer
        for i in range(1, len(total_params)):
            increase = total_params[i] - total_params[i-1]
            param_increases.append(increase)
        
        ax4.bar(layers, param_increases, color='purple', alpha=0.7)
        ax4.set_xlabel('Number of MLP Layers')
        ax4.set_ylabel('Parameter Increase from Previous')
        ax4.set_title('Parameter Increase per Additional Layer')
        ax4.set_xticks(layers)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (x, y) in enumerate(zip(layers, param_increases)):
            if y > 0:  # Don't label zero increase
                ax4.annotate(f'{y:,}', (x, y), textcoords="offset points", xytext=(0,5), ha='center')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = config.figure_dir / "question_03_parameter_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Parameter analysis plot saved to: {plot_path}")
        
        # Create a summary table
        self._create_summary_table()
    
    def _create_summary_table(self):
        """Create a detailed summary table."""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        headers = ['MLP Layers', 'Total Parameters', 'GMF Embeddings', 'MLP Embeddings', 
                  'MLP Layers', 'Prediction Layer', 'Increase from Previous']
        
        prev_total = 0
        for layers in [1, 2, 3]:
            result = self.results[f"{layers}layers"]
            total = result['total_parameters']
            breakdown = result['breakdown']
            increase = total - prev_total if prev_total > 0 else 0
            
            row = [
                str(layers),
                f"{total:,}",
                f"{breakdown['gmf_embeddings']:,}",
                f"{breakdown['mlp_embeddings']:,}",
                f"{breakdown['mlp_layers']:,}",
                f"{breakdown['prediction_layer']:,}",
                f"{increase:,}" if increase > 0 else "-"
            ]
            table_data.append(row)
            prev_total = total
        
        table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code the rows
        colors = ['#E3F2FD', '#F3E5F5', '#E8F5E8']
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                table[(i, j)].set_facecolor(colors[i-1])
        
        ax.set_title('Question 3: Detailed Parameter Analysis - NeuMF without pretraining', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Save table
        table_path = config.figure_dir / "question_03_parameter_table.png"
        plt.savefig(table_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Parameter summary table saved to: {table_path}")
    
    def save_results(self):
        """Save results to JSON file."""
        print("\n[SAVE] Saving results...")
        
        # Save detailed results
        results_path = config.output_dir / "reports" / "question_03_results.json"
        save_json(self.results, results_path)
        
        # Create a simplified summary for easy reading
        summary = {}
        for layers in [1, 2, 3]:
            result = self.results[f"{layers}layers"]
            summary[f"{layers}_layers"] = {
                'total_parameters': result['total_parameters'],
                'gmf_embeddings': result['breakdown']['gmf_embeddings'],
                'mlp_embeddings': result['breakdown']['mlp_embeddings'],
                'mlp_layers': result['breakdown']['mlp_layers'],
                'prediction_layer': result['breakdown']['prediction_layer']
            }
        
        summary_path = config.output_dir / "reports" / "question_03_summary.json"
        save_json(summary, summary_path)
        
        print(f"Detailed results saved to: {results_path}")
        print(f"Summary results saved to: {summary_path}")
    
    def print_summary(self):
        """Print a text summary of the results."""
        print("\n" + "="*60)
        print("QUESTION 3 RESULTS SUMMARY")
        print("="*60)
        print("Parameter count for NeuMF-end (without pretraining):")
        print()
        
        prev_total = 0
        for layers in [1, 2, 3]:
            result = self.results[f"{layers}layers"]
            total = result['total_parameters']
            increase = total - prev_total if prev_total > 0 else 0
            increase_pct = (increase / prev_total * 100) if prev_total > 0 else 0
            
            print(f"{layers} MLP layer(s): {total:,} parameters", end="")
            if increase > 0:
                print(f" (+{increase:,}, +{increase_pct:.1f}%)")
            else:
                print()
            
            breakdown = result['breakdown']
            print(f"  - GMF embeddings:    {breakdown['gmf_embeddings']:,}")
            print(f"  - MLP embeddings:    {breakdown['mlp_embeddings']:,}")
            print(f"  - MLP layers:        {breakdown['mlp_layers']:,}")
            print(f"  - Prediction layer:  {breakdown['prediction_layer']:,}")
            print()
            
            prev_total = total
        
        print("Key observations:")
        print("- GMF embeddings remain constant (independent of MLP layers)")
        print("- MLP embeddings grow exponentially with layers")
        print("- MLP layer parameters grow significantly with depth")
        print("- Prediction layer remains constant (64 -> 1)")
    
    def run_experiment(self):
        """Run the complete Question 3 experiment."""
        print("Starting Question 3 experiment...")
        
        # Run the analysis
        self.run_analysis()
        
        # Create visualizations
        self.create_plots()
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
        
        print("\n" + "="*60)
        print("QUESTION 3 EXPERIMENT COMPLETED!")
        print("="*60)
        print("Generated files:")
        print(f"- {config.figure_dir}/question_03_parameter_analysis.png")
        print(f"- {config.figure_dir}/question_03_parameter_table.png")
        print(f"- {config.output_dir}/reports/question_03_results.json")
        print(f"- {config.output_dir}/reports/question_03_summary.json")
        print("="*60)
        
        return self.results

def main():
    print("Question 3: Parameter Analysis for NeuMF")
    print("Analyzing how parameter count changes with MLP layers (1-3)")
    
    # Create and run experiment
    experiment = Question3Experiment()
    results = experiment.run_experiment()

if __name__ == "__main__":
    main()