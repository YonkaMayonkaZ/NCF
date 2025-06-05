#!/usr/bin/env python3
"""
Question 10: NMF parameters vs latent factors
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ncf.nmf_model import NMFRecommender
from src.data.datasets import load_all
from src.utils.config import config
from src.utils.io import save_json, ensure_dir

def main():
    print("Q10: NMF Parameters vs Latent Factors")
    
    # Load data to get dimensions
    train_data, test_data, user_num, item_num, train_mat = load_all()
    
    # Factor range: 1 to 30 step 5
    factors = list(range(1, 31, 5))  # [1, 6, 11, 16, 21, 26]
    
    results = {}
    
    for n_factors in factors:
        # Create NMF model to get parameter count
        nmf = NMFRecommender(n_components=n_factors)
        nmf.fit(train_mat)
        
        n_params = nmf.get_n_parameters()
        results[n_factors] = n_params
        
        print(f"{n_factors} factors: {n_params:,} parameters")
    
    # Plot
    ensure_dir(config.figure_dir)
    
    factors_list = sorted(results.keys())
    params_list = [results[f] for f in factors_list]
    
    plt.figure(figsize=(10, 6))
    plt.plot(factors_list, params_list, 'ro-', linewidth=3, markersize=10)
    plt.xlabel('Number of Latent Factors')
    plt.ylabel('Number of Parameters')
    plt.title('Question 10: NMF Parameters vs Latent Factors')
    plt.grid(True, alpha=0.3)
    plt.xticks(factors_list)
    
    # Add parameter labels
    for x, y in zip(factors_list, params_list):
        plt.annotate(f'{y:,}', (x, y + max(params_list) * 0.02),
                    ha='center', va='bottom', fontsize=10, weight='bold')
    
    plot_path = config.figure_dir / "question_10_nmf_parameters.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    ensure_dir(config.output_dir / "reports")
    results_path = config.output_dir / "reports" / "question_10_results.json"
    save_json(results, results_path)
    
    print(f"Plot saved: {plot_path}")
    print(f"Results saved: {results_path}")
    
    # Show parameter growth
    print(f"Parameter range: {min(params_list):,} to {max(params_list):,}")

if __name__ == "__main__":
    main()