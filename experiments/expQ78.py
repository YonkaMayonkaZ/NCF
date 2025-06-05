#!/usr/bin/env python3
"""
Questions 7 & 8: Negative sampling analysis (10 runs each)
Q7: HR@10 vs number of negative samples (1 to 10) 
Q8: NDCG@10 vs number of negative samples (1 to 10)
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ncf.models import NCF
from src.data.datasets import load_all, NCFData
from src.training.metrics import metrics
from src.utils.config import config
from src.utils.io import save_json, ensure_dir

def train_with_negatives(num_negatives, run_id, epochs=15):
    """Train NeuMF model with specific number of negative samples."""
    print(f"  Run {run_id}/10 with {num_negatives} negative samples...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_data, test_data, user_num, item_num, train_mat = load_all()
    
    # Create datasets with specific number of negatives
    train_dataset = NCFData(train_data, item_num, train_mat, num_negatives, True)
    test_dataset = NCFData(test_data, item_num, train_mat, 0, False)
    
    train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, 
                                  shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=config.test_num_ng + 1, 
                                 shuffle=False, num_workers=0)
    
    # Create model with different random seed for each run
    model = NCF(user_num, item_num, config.factor_num, config.num_layers, 
               config.dropout, "NeuMF-end")
    model.to(device)
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    best_hr = 0
    best_ndcg = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loader.dataset.ng_sample()
        
        for user, item, label in train_loader:
            user = user.to(device)
            item = item.to(device)
            label = label.float().to(device)
            
            optimizer.zero_grad()
            prediction = model(user, item)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()
        
        # Quick evaluation every few epochs
        if epoch % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                HR, NDCG = metrics(model, test_loader, 10)
            
            hr = np.mean(HR)
            ndcg = np.mean(NDCG)
            
            if hr > best_hr:
                best_hr = hr
                best_ndcg = ndcg
    
    print(f"    Final: HR@10 = {best_hr:.4f}, NDCG@10 = {best_ndcg:.4f}")
    return best_hr, best_ndcg

def run_negative_sampling_experiment(num_runs=10):
    """Run experiments with different numbers of negative samples, 10 times each."""
    negative_range = list(range(1, 11))  # 1 to 10
    results = {}
    
    print(f"Running negative sampling experiments ({num_runs} runs each)...")
    
    for num_neg in negative_range:
        print(f"\nTesting {num_neg} negative samples...")
        
        hr_runs = []
        ndcg_runs = []
        
        for run in range(num_runs):
            hr, ndcg = train_with_negatives(num_neg, run + 1, epochs=15)
            hr_runs.append(hr)
            ndcg_runs.append(ndcg)
        
        # Calculate statistics
        results[num_neg] = {
            'hr_mean': np.mean(hr_runs),
            'hr_std': np.std(hr_runs),
            'ndcg_mean': np.mean(ndcg_runs),
            'ndcg_std': np.std(ndcg_runs)
        }
        
        print(f"  {num_neg} negatives: HR@10 = {np.mean(hr_runs):.4f}±{np.std(hr_runs):.4f}, "
              f"NDCG@10 = {np.mean(ndcg_runs):.4f}±{np.std(ndcg_runs):.4f}")
    
    return results

def create_plots(results):
    """Create plots for questions 7 and 8 with error bars."""
    ensure_dir(config.figure_dir)
    
    neg_values = sorted(results.keys())
    hr_means = [results[n]['hr_mean'] for n in neg_values]
    hr_stds = [results[n]['hr_std'] for n in neg_values]
    ndcg_means = [results[n]['ndcg_mean'] for n in neg_values]
    ndcg_stds = [results[n]['ndcg_std'] for n in neg_values]
    
    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Question 7: HR@10 vs Number of Negatives with error bars
    ax1.errorbar(neg_values, hr_means, yerr=hr_stds, fmt='go-', 
                capsize=5, linewidth=3, markersize=8)
    ax1.set_xlabel('Number of Negative Samples')
    ax1.set_ylabel('HR@10')
    ax1.set_title('Question 7: HR@10 vs Number of Negative Samples')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(neg_values)
    ax1.set_ylim(0, max(hr_means) * 1.2)
    
    # Add value labels with std
    for x, y, std in zip(neg_values, hr_means, hr_stds):
        ax1.annotate(f'{y:.3f}±{std:.3f}', (x, y + max(hr_means) * 0.03),
                    ha='center', va='bottom', fontsize=8, weight='bold')
    
    # Question 8: NDCG@10 vs Number of Negatives with error bars
    ax2.errorbar(neg_values, ndcg_means, yerr=ndcg_stds, fmt='mo-', 
                capsize=5, linewidth=3, markersize=8)
    ax2.set_xlabel('Number of Negative Samples')
    ax2.set_ylabel('NDCG@10')
    ax2.set_title('Question 8: NDCG@10 vs Number of Negative Samples')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(neg_values)
    ax2.set_ylim(0, max(ndcg_means) * 1.2)
    
    # Add value labels with std
    for x, y, std in zip(neg_values, ndcg_means, ndcg_stds):
        ax2.annotate(f'{y:.3f}±{std:.3f}', (x, y + max(ndcg_means) * 0.03),
                    ha='center', va='bottom', fontsize=8, weight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = config.figure_dir / "questions_07_08_negative_sampling.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved: {plot_path}")
    
    # Create individual plots
    _create_individual_plots(neg_values, hr_means, hr_stds, ndcg_means, ndcg_stds)

def _create_individual_plots(neg_values, hr_means, hr_stds, ndcg_means, ndcg_stds):
    """Create individual plots for each question with error bars."""
    
    # Question 7 plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(neg_values, hr_means, yerr=hr_stds, fmt='go-', 
                capsize=5, linewidth=3, markersize=10)
    plt.xlabel('Number of Negative Samples', fontsize=12)
    plt.ylabel('HR@10', fontsize=12)
    plt.title('Question 7: HR@10 vs Number of Negative Samples (10 runs)', fontsize=14, weight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(neg_values)
    
    for x, y, std in zip(neg_values, hr_means, hr_stds):
        plt.annotate(f'{y:.3f}±{std:.3f}', (x, y + max(hr_means) * 0.03),
                    ha='center', va='bottom', fontsize=9, weight='bold')
    
    plt.tight_layout()
    q7_path = config.figure_dir / "question_07_hr_negative_sampling.png"
    plt.savefig(q7_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Question 8 plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(neg_values, ndcg_means, yerr=ndcg_stds, fmt='mo-', 
                capsize=5, linewidth=3, markersize=10)
    plt.xlabel('Number of Negative Samples', fontsize=12)
    plt.ylabel('NDCG@10', fontsize=12)
    plt.title('Question 8: NDCG@10 vs Number of Negative Samples (10 runs)', fontsize=14, weight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(neg_values)
    
    for x, y, std in zip(neg_values, ndcg_means, ndcg_stds):
        plt.annotate(f'{y:.3f}±{std:.3f}', (x, y + max(ndcg_means) * 0.03),
                    ha='center', va='bottom', fontsize=9, weight='bold')
    
    plt.tight_layout()
    q8_path = config.figure_dir / "question_08_ndcg_negative_sampling.png"
    plt.savefig(q8_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Individual plots saved: {q7_path}, {q8_path}")

def main():
    print("Questions 7 & 8: Negative Sampling Analysis (10 runs each)")
    print("This will train models with different numbers of negative samples (1-10)")
    print("Total models to train: 10 configurations 10 runs = 100 models")
    print("Estimated time: 2-3 hours")
    print()
    
    # Run experiments
    results = run_negative_sampling_experiment(num_runs=10)
    
    # Create plots
    create_plots(results)
    
    # Save results
    ensure_dir(config.output_dir / "reports")
    results_path = config.output_dir / "reports" / "questions_07_08_results.json"
    save_json(results, results_path)
    print(f"Results saved: {results_path}")
    
    # Find optimal number of negatives
    best_hr_negatives = max(results.keys(), key=lambda x: results[x]['hr_mean'])
    best_ndcg_negatives = max(results.keys(), key=lambda x: results[x]['ndcg_mean'])
    
    print("\nSummary (mean ± std over 10 runs):")
    for neg in sorted(results.keys()):
        hr_mean = results[neg]['hr_mean']
        hr_std = results[neg]['hr_std']
        ndcg_mean = results[neg]['ndcg_mean']
        ndcg_std = results[neg]['ndcg_std']
        print(f"  {neg:2d} negatives: HR@10 = {hr_mean:.4f}±{hr_std:.4f}, NDCG@10 = {ndcg_mean:.4f}±{ndcg_std:.4f}")
    
    print(f"\nOptimal configurations:")
    print(f"  Best HR@10: {best_hr_negatives} negatives "
          f"(HR@10 = {results[best_hr_negatives]['hr_mean']:.4f}±{results[best_hr_negatives]['hr_std']:.4f})")
    print(f"  Best NDCG@10: {best_ndcg_negatives} negatives "
          f"(NDCG@10 = {results[best_ndcg_negatives]['ndcg_mean']:.4f}±{results[best_ndcg_negatives]['ndcg_std']:.4f})")

if __name__ == "__main__":
    main()