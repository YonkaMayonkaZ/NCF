#!/usr/bin/env python3
"""
Question 9: NMF with different latent factors, showing HR@10 and NDCG@10
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ncf.nmf_model import NMFRecommender, NMFEvaluator
from src.data.datasets import load_all
from src.utils.config import config
from src.utils.io import save_json, ensure_dir

def main():
    print("Q9: NMF HR@10 and NDCG@10 vs Latent Factors")
    
    # Load data
    train_data, test_data, user_num, item_num, train_mat = load_all()
    
    # Factor range: 1 to 30 step 5
    factors = list(range(1, 32, 5))  # [1, 6, 11, 16, 21, 26]
    num_runs = 10
    
    results = {}
    
    for n_factors in factors:
        print(f"Testing {n_factors} factors...")
        
        ndcg_runs = []
        hr_runs = []
        
        for run in range(num_runs):
            # Train NMF with more iterations
            nmf = NMFRecommender(n_components=n_factors, random_state=42+run, max_iter=500)
            nmf.fit(train_mat)
            
            # Evaluate
            evaluator = NMFEvaluator(nmf, test_data, train_mat)
            hr, ndcg = evaluator.evaluate()
            
            ndcg_runs.append(ndcg)
            hr_runs.append(hr)
        
        results[n_factors] = {
            'ndcg_mean': np.mean(ndcg_runs),
            'ndcg_std': np.std(ndcg_runs),
            'hr_mean': np.mean(hr_runs),
            'hr_std': np.std(hr_runs)
        }
        
        print(f"  {n_factors} factors: HR@10 = {np.mean(hr_runs):.4f} +/- {np.std(hr_runs):.4f}, NDCG@10 = {np.mean(ndcg_runs):.4f} +/- {np.std(ndcg_runs):.4f}")
    
    # Plot
    ensure_dir(config.figure_dir)
    
    factors_list = sorted(results.keys())
    ndcg_means = [results[f]['ndcg_mean'] for f in factors_list]
    ndcg_stds = [results[f]['ndcg_std'] for f in factors_list]
    hr_means = [results[f]['hr_mean'] for f in factors_list]
    hr_stds = [results[f]['hr_std'] for f in factors_list]
    
    # Create subplot with both metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # HR@10 plot
    ax1.errorbar(factors_list, hr_means, yerr=hr_stds, 
                fmt='go-', capsize=5, linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Latent Factors')
    ax1.set_ylabel('HR@10')
    ax1.set_title('Question 9: NMF HR@10 vs Latent Factors')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(factors_list)
    
    # NDCG@10 plot
    ax2.errorbar(factors_list, ndcg_means, yerr=ndcg_stds, 
                fmt='bo-', capsize=5, linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Latent Factors')
    ax2.set_ylabel('NDCG@10')
    ax2.set_title('Question 9: NMF NDCG@10 vs Latent Factors')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(factors_list)
    
    plt.tight_layout()
    plot_path = config.figure_dir / "question_09_nmf_metrics.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    ensure_dir(config.output_dir / "reports")
    results_path = config.output_dir / "reports" / "question_09_results.json"
    save_json(results, results_path)
    
    print(f"Plot saved: {plot_path}")
    print(f"Results saved: {results_path}")
    
    # Find best configurations
    best_ndcg_factors = max(results.keys(), key=lambda x: results[x]['ndcg_mean'])
    best_hr_factors = max(results.keys(), key=lambda x: results[x]['hr_mean'])
    
    print(f"Best NDCG@10: {best_ndcg_factors} factors with NDCG@10 = {results[best_ndcg_factors]['ndcg_mean']:.4f}")
    print(f"Best HR@10: {best_hr_factors} factors with HR@10 = {results[best_hr_factors]['hr_mean']:.4f}")

if __name__ == "__main__":
    main()