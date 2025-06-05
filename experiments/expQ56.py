#!/usr/bin/env python3
"""
Questions 5 & 6: Top-K performance evaluation (10 runs each)
Q5: HR@K vs K (K from 1 to 10)
Q6: NDCG@K vs K (K from 1 to 10)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ncf.models import NCF
from src.data.datasets import load_all, NCFData
from src.training.metrics import metrics
from src.utils.config import config
from src.utils.io import save_json, ensure_dir

def train_single_model(run_id, epochs=10):
    """Train a single NeuMF model for evaluation."""
    print(f"Training model {run_id}/10...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_data, test_data, user_num, item_num, train_mat = load_all()
    train_dataset = NCFData(train_data, item_num, train_mat, config.num_ng, True)
    
    train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, 
                                  shuffle=True, num_workers=4)
    
    # Create model
    model = NCF(user_num, item_num, config.factor_num, config.num_layers, 
               config.dropout, "NeuMF-end")
    model.to(device)
    
    # Training setup
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    # Quick training
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
    
    model.eval()
    return model, device

def run_top_k_experiments(num_runs=10, max_k=10):
    """Run top-K experiments multiple times."""
    print(f"Running top-K experiments {num_runs} times...")
    
    # Load test data once
    train_data, test_data, user_num, item_num, train_mat = load_all()
    test_dataset = NCFData(test_data, item_num, train_mat, 0, False)
    test_loader = data.DataLoader(test_dataset, batch_size=config.test_num_ng + 1, 
                                 shuffle=False, num_workers=0)
    
    all_results = []
    
    for run in range(num_runs):
        # Train a model for this run
        model, device = train_single_model(run + 1, epochs=15)
        
        # Evaluate this model for all K values
        run_results = {}
        for k in range(1, max_k + 1):
            HR, NDCG = metrics(model, test_loader, k)
            run_results[k] = {
                'hr': np.mean(HR),
                'ndcg': np.mean(NDCG)
            }
        
        all_results.append(run_results)
        print(f"  Run {run+1}: HR@10 = {run_results[10]['hr']:.4f}, NDCG@10 = {run_results[10]['ndcg']:.4f}")
    
    # Calculate statistics across runs
    final_results = {}
    for k in range(1, max_k + 1):
        hr_values = [result[k]['hr'] for result in all_results]
        ndcg_values = [result[k]['ndcg'] for result in all_results]
        
        final_results[k] = {
            'hr_mean': np.mean(hr_values),
            'hr_std': np.std(hr_values),
            'ndcg_mean': np.mean(ndcg_values),
            'ndcg_std': np.std(ndcg_values)
        }
    
    return final_results

def create_plots(results):
    """Create plots for questions 5 and 6 with error bars."""
    ensure_dir(config.figure_dir)
    
    k_values = sorted(results.keys())
    hr_means = [results[k]['hr_mean'] for k in k_values]
    hr_stds = [results[k]['hr_std'] for k in k_values]
    ndcg_means = [results[k]['ndcg_mean'] for k in k_values]
    ndcg_stds = [results[k]['ndcg_std'] for k in k_values]
    
    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Question 5: HR@K vs K with error bars
    ax1.errorbar(k_values, hr_means, yerr=hr_stds, fmt='bo-', 
                capsize=5, linewidth=3, markersize=8)
    ax1.set_xlabel('K')
    ax1.set_ylabel('HR@K')
    ax1.set_title('Question 5: Hit Ratio @ K')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)
    ax1.set_ylim(0, max(hr_means) * 1.2)
    
    # Add value labels with std
    for x, y, std in zip(k_values, hr_means, hr_stds):
        ax1.annotate(f'{y:.3f}±{std:.3f}', (x, y + max(hr_means) * 0.03),
                    ha='center', va='bottom', fontsize=8, weight='bold')
    
    # Question 6: NDCG@K vs K with error bars
    ax2.errorbar(k_values, ndcg_means, yerr=ndcg_stds, fmt='ro-', 
                capsize=5, linewidth=3, markersize=8)
    ax2.set_xlabel('K')
    ax2.set_ylabel('NDCG@K')
    ax2.set_title('Question 6: NDCG @ K')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_values)
    ax2.set_ylim(0, max(ndcg_means) * 1.2)
    
    # Add value labels with std
    for x, y, std in zip(k_values, ndcg_means, ndcg_stds):
        ax2.annotate(f'{y:.3f}±{std:.3f}', (x, y + max(ndcg_means) * 0.03),
                    ha='center', va='bottom', fontsize=8, weight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = config.figure_dir / "questions_05_06_top_k_performance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved: {plot_path}")
    
    # Create individual plots
    _create_individual_plots(k_values, hr_means, hr_stds, ndcg_means, ndcg_stds)

def _create_individual_plots(k_values, hr_means, hr_stds, ndcg_means, ndcg_stds):
    """Create individual plots for each question with error bars."""
    
    # Question 5 plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(k_values, hr_means, yerr=hr_stds, fmt='bo-', 
                capsize=5, linewidth=3, markersize=10)
    plt.xlabel('K', fontsize=12)
    plt.ylabel('HR@K', fontsize=12)
    plt.title('Question 5: Hit Ratio @ K (10 runs)', fontsize=14, weight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    
    for x, y, std in zip(k_values, hr_means, hr_stds):
        plt.annotate(f'{y:.3f}±{std:.3f}', (x, y + max(hr_means) * 0.03),
                    ha='center', va='bottom', fontsize=9, weight='bold')
    
    plt.tight_layout()
    q5_path = config.figure_dir / "question_05_hr_at_k.png"
    plt.savefig(q5_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Question 6 plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(k_values, ndcg_means, yerr=ndcg_stds, fmt='ro-', 
                capsize=5, linewidth=3, markersize=10)
    plt.xlabel('K', fontsize=12)
    plt.ylabel('NDCG@K', fontsize=12)
    plt.title('Question 6: NDCG @ K (10 runs)', fontsize=14, weight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    
    for x, y, std in zip(k_values, ndcg_means, ndcg_stds):
        plt.annotate(f'{y:.3f}±{std:.3f}', (x, y + max(ndcg_means) * 0.03),
                    ha='center', va='bottom', fontsize=9, weight='bold')
    
    plt.tight_layout()
    q6_path = config.figure_dir / "question_06_ndcg_at_k.png"
    plt.savefig(q6_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Individual plots saved: {q5_path}, {q6_path}")

def main():
    print("Questions 5 & 6: Top-K Performance Evaluation (10 runs)")
    print("Training 10 models and evaluating each with K=1 to 10")
    print()
    
    # Run experiments 10 times
    results = run_top_k_experiments(num_runs=10, max_k=10)
    
    # Create plots
    create_plots(results)
    
    # Save results
    ensure_dir(config.output_dir / "reports")
    results_path = config.output_dir / "reports" / "questions_05_06_results.json"
    save_json(results, results_path)
    print(f"Results saved: {results_path}")
    
    # Print summary with mean ± std
    print("\nSummary (mean ± std over 10 runs):")
    for k in sorted(results.keys()):
        hr_mean = results[k]['hr_mean']
        hr_std = results[k]['hr_std']
        ndcg_mean = results[k]['ndcg_mean']
        ndcg_std = results[k]['ndcg_std']
        print(f"  K={k:2d}: HR@{k} = {hr_mean:.4f}±{hr_std:.4f}, NDCG@{k} = {ndcg_mean:.4f}±{ndcg_std:.4f}")

if __name__ == "__main__":
    main()