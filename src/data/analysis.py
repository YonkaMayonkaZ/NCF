import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from src.utils.io import save_results
from src.utils.logging import get_experiment_logger

class RatingDataAnalyzer:
    def __init__(self, train_path="data/processed/u.train.rating", output_dir="results/figures"):
        self.train_path = Path(train_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df = pd.read_csv(self.train_path, sep='\t', names=['user_id', 'item_id'])
        self.logger = get_experiment_logger("data_analysis")
        self.results = {}

    def basic_stats(self):
        num_users = self.df["user_id"].nunique()
        num_items = self.df["item_id"].nunique()
        num_interactions = len(self.df)
        sparsity = 1 - (num_interactions / (num_users * num_items))

        self.logger.info(f"Users: {num_users}, Items: {num_items}, Interactions: {num_interactions}, Sparsity: {sparsity:.4f}")
        self.results['basic_stats'] = {
            'num_users': num_users,
            'num_items': num_items,
            'num_interactions': num_interactions,
            'sparsity': sparsity
        }

    def plot_distributions(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Dataset Analysis", fontsize=16)

        user_counts = self.df.groupby('user_id').size()
        item_counts = self.df.groupby('item_id').size()

        ax = axes[0, 0]
        ax.hist(user_counts, bins=50, color='skyblue', edgecolor='black')
        ax.set_title("Interactions per User")
        ax.set_xlabel("Count")
        ax.set_ylabel("Users")

        ax = axes[0, 1]
        ax.hist(item_counts, bins=50, color='salmon', edgecolor='black')
        ax.set_title("Interactions per Item")
        ax.set_xlabel("Count")
        ax.set_ylabel("Items")

        ax = axes[1, 0]
        top_users = user_counts.sort_values(ascending=False).head(10)
        ax.bar(top_users.index.astype(str), top_users.values, color='green')
        ax.set_title("Top 10 Active Users")
        ax.set_ylabel("Interactions")
        ax.set_xticks(np.arange(len(top_users)))  # Set tick positions
        ax.set_xticklabels(top_users.index.astype(str), rotation=45)

        ax = axes[1, 1]
        top_items = item_counts.sort_values(ascending=False).head(10)
        ax.bar(top_items.index.astype(str), top_items.values, color='purple')
        ax.set_title("Top 10 Popular Items")
        ax.set_ylabel("Interactions")
        ax.set_xticks(np.arange(len(top_items)))  # Set tick positions
        ax.set_xticklabels(top_items.index.astype(str), rotation=45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = self.output_dir / "dataset_analysis.png"
        plt.savefig(plot_path)
        plt.close()  # Close figure to free memory
        self.logger.info(f"Analysis plots saved to {plot_path}")  # Remove emoji

    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(self.results, f"data_analysis_{timestamp}")
        self.logger.info("Saved analysis results to results/reports/")

    def run_all(self):
        self.basic_stats()
        self.plot_distributions()
        self.save_results()