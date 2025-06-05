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

def main():
    logger = get_experiment_logger("question_9")
    logger.info("Starting Question 9: NMF latent factors experiment")
    
    # Load data
    train_data, test_data, user_num, item_num, train_mat = load_all()
    logger.info(f"Data: {user_num} users, {item_num} items")
    
    # Test different numbers of latent factors
    factors_list = list(range(1, 31, 5))  # [1, 6, 11, 16, 21, 26]
    num_runs = 10
    results = {}
    
    logger.info(f"Testing factors: {factors_list} with {num_runs} runs each")
    
    for n_factors in factors_list:
        logger.info(f"Testing {n_factors} factors...")
        
        hrs = []
        ndcgs = []
        
        for run in range(num_runs):
            # Train NMF
            nmf_model = NMFRecommender(n_components=n_factors, random_state=42+run)
            nmf_model.fit(train_mat)
            
            # Evaluate
            evaluator = NMFEvaluator(nmf_model, test_data, train_mat)
            hr, ndcg = evaluator.evaluate()
            
            hrs.append(hr)
            ndcgs.append(ndcg)
        
        # Store results (convert numpy types to Python types)
        results[n_factors] = {
            'hr_mean': float(np.mean(hrs)),
            'hr_std': float(np.std(hrs)),
            'ndcg_mean': float(np.mean(ndcgs)),
            'ndcg_std': float(np.std(ndcgs)),
            'parameters': int((user_num + item_num) * n_factors)
        }
        
        logger.info(f"{n_factors} factors: NDCG@10 = {np.mean(ndcgs):.4f} +/- {np.std(ndcgs):.4f}")
    
    # Find best configuration
    best_factors = max(results.keys(), key=lambda k: results[k]['ndcg_mean'])
    logger.info(f"Best configuration: {best_factors} factors (NDCG@10 = {results[best_factors]['ndcg_mean']:.4f})")
    
    # Create plot
    factors = list(results.keys())
    ndcg_means = [results[f]['ndcg_mean'] for f in factors]
    ndcg_stds = [results[f]['ndcg_std'] for f in factors]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(factors, ndcg_means, yerr=ndcg_stds, marker='o', capsize=5)
    plt.xlabel('Number of Latent Factors')
    plt.ylabel('NDCG@10')
    plt.title('Question 9: NMF Performance vs Latent Factors')
    plt.grid(True, alpha=0.3)
    plt.xticks(factors)
    
    # Highlight best
    best_idx = factors.index(best_factors)
    plt.plot(best_factors, ndcg_means[best_idx], 'ro', markersize=10, label=f'Best: {best_factors}')
    plt.legend()
    
    # Save plot
    plot_path = config.figure_dir / "question_09_nmf_performance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved: {plot_path}")
    
    # Save results
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