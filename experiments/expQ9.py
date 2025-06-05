#!/usr/bin/env python3
"""
Question 9: NMF performance vs latent factors (1-30, step 5)
Simple implementation focusing on NDCG@10 as requested.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.ncf.nmf_model import NMFRecommender, NMFEvaluator
from src.data.datasets import load_all
from src.utils.config import config
from src.utils.io import save_json
from src.utils.logging import get_experiment_logger

def get_iterations_for_factors(n_factors):
    """
    Adaptive iterations: fewer factors need more iterations to converge.
    """
    if n_factors <= 5:
        return 350
    elif n_factors <= 10:
        return 300
    elif n_factors <= 15:
        return 250
    elif n_factors <= 20:
        return 200
    elif n_factors <= 25:
        return 150
    else:
        return 100

def main():
    logger = get_experiment_logger("question_9")
    logger.info("Starting Question 9: NMF latent factors experiment")
    
    # Load data
    train_data, test_data, user_num, item_num, train_mat = load_all()
    logger.info(f"Data: {user_num} users, {item_num} items")
    
    # Test different numbers of latent factors
    factors_list = list(range(1, 32, 5))  # [1, 6, 11, 16, 21, 26]
    num_runs = 10
    results = {}
    
    logger.info(f"Testing factors: {factors_list} with {num_runs} runs each")
    
    for n_factors in factors_list:
        max_iter = get_iterations_for_factors(n_factors)
        logger.info(f"Testing {n_factors} factors (max_iter={max_iter})...")
        
        hrs = []
        ndcgs = []
        
        for run in range(num_runs):
            # Train NMF with adaptive iterations
            nmf_model = NMFRecommender(n_components=n_factors, 
                                     max_iter=max_iter,
                                     random_state=42+run)
            nmf_model.fit(train_mat)
            
            # Evaluate
            evaluator = NMFEvaluator(nmf_model, test_data, train_mat)
            hr, ndcg = evaluator.evaluate()
            
            hrs.append(hr)
            ndcgs.append(ndcg)
        
        # Store results (convert numpy types to Python types for JSON)
        results[n_factors] = {
            'hr_mean': float(np.mean(hrs)),
            'hr_std': float(np.std(hrs)),
            'ndcg_mean': float(np.mean(ndcgs)),
            'ndcg_std': float(np.std(ndcgs)),
            'parameters': int(user_num + item_num) * int(n_factors),
            'max_iterations': int(max_iter)
        }
        
        logger.info(f"{n_factors} factors: HR@10 = {np.mean(hrs):.4f} +/- {np.std(hrs):.4f}, NDCG@10 = {np.mean(ndcgs):.4f} +/- {np.std(ndcgs):.4f}")
    
    # Find best configuration
    best_factors = max(results.keys(), key=lambda k: results[k]['ndcg_mean'])
    logger.info(f"Best configuration: {best_factors} factors (NDCG@10 = {results[best_factors]['ndcg_mean']:.4f})")
    
    # Create plot with both NDCG@10 and HR@10
    factors = list(results.keys())
    ndcg_means = [results[f]['ndcg_mean'] for f in factors]
    ndcg_stds = [results[f]['ndcg_std'] for f in factors]
    hr_means = [results[f]['hr_mean'] for f in factors]
    hr_stds = [results[f]['hr_std'] for f in factors]
    
    # Create subplot with both metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # NDCG@10 plot
    ax1.errorbar(factors, ndcg_means, yerr=ndcg_stds, marker='o', capsize=5, color='blue')
    ax1.set_xlabel('Number of Latent Factors')
    ax1.set_ylabel('NDCG@10')
    ax1.set_title('Question 9: NDCG@10 vs Latent Factors')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(factors)
    
    # Highlight best NDCG
    best_idx = factors.index(best_factors)
    ax1.plot(best_factors, ndcg_means[best_idx], 'ro', markersize=10, label=f'Best: {best_factors}')
    ax1.legend()
    
    # HR@10 plot
    ax2.errorbar(factors, hr_means, yerr=hr_stds, marker='s', capsize=5, color='green')
    ax2.set_xlabel('Number of Latent Factors')
    ax2.set_ylabel('HR@10')
    ax2.set_title('Question 9: HR@10 vs Latent Factors')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(factors)
    
    # Highlight best NDCG config on HR plot too
    ax2.plot(best_factors, hr_means[best_idx], 'ro', markersize=10, label=f'Best NDCG: {best_factors}')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = config.figure_dir / "question_09_nmf_performance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved: {plot_path}")
    
    # Also create a simple combined plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(factors, ndcg_means, yerr=ndcg_stds, marker='o', capsize=5, 
                label='NDCG@10', color='blue')
    plt.errorbar(factors, hr_means, yerr=hr_stds, marker='s', capsize=5, 
                label='HR@10', color='green')
    plt.xlabel('Number of Latent Factors')
    plt.ylabel('Performance')
    plt.title('Question 9: NMF Performance vs Latent Factors')
    plt.grid(True, alpha=0.3)
    plt.xticks(factors)
    plt.legend()
    
    # Highlight best
    plt.plot(best_factors, ndcg_means[best_idx], 'ro', markersize=8)
    plt.plot(best_factors, hr_means[best_idx], 'ro', markersize=8)
    
    # Save combined plot
    combined_path = config.figure_dir / "question_09_nmf_combined.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Combined plot saved: {combined_path}")
    
    # Save results (all values are now Python types, not numpy)
    save_json(results, config.output_dir / "reports" / "question_09_results.json")
    
    summary = {
        'best_factors': int(best_factors),
        'best_ndcg': float(results[best_factors]['ndcg_mean']),
        'all_results': results
    }
    save_json(summary, config.output_dir / "reports" / "question_09_summary.json")
    
    logger.info("Question 9 completed!")
    return results

if __name__ == "__main__":
    main()