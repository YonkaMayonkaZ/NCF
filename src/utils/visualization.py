import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

def aggregate_runs(run_histories, metrics=["loss", "hr", "ndcg"]):
    """Aggregate metrics from multiple runs to compute mean and std.

    Args:
        run_histories (list): List of histories, each a list of dicts with 'epoch' and metrics.
        metrics (list): Metrics to aggregate (e.g., ['loss', 'hr', 'ndcg']).

    Returns:
        dict: Aggregated results with 'epochs', 'mean_<metric>', 'std_<metric>'.
    """
    epochs = [h["epoch"] for h in run_histories[0]]
    num_epochs = len(epochs)
    num_runs = len(run_histories)
    
    agg_results = {"epochs": epochs}
    for metric in metrics:
        # Stack metric values: shape (num_runs, num_epochs)
        values = np.array([[h[metric] for h in history] for history in run_histories])
        agg_results[f"mean_{metric}"] = np.mean(values, axis=0)
        agg_results[f"std_{metric}"] = np.std(values, axis=0)
    
    return agg_results

def plot_training_metrics(
    run_histories,
    model_name,
    output_path="results/figures/training_metrics.png",
    metrics=["loss", "hr", "ndcg"],
    metric_labels={"loss": "Loss", "hr": "HR@10", "ndcg": "NDCG@10"}
):
    """Plot mean and std of training metrics over epochs for a single model.

    Args:
        run_histories (list): List of histories from multiple runs.
        model_name (str): Name of the model (e.g., 'GMF').
        output_path (str or Path): Path to save the plot.
        metrics (list): Metrics to plot.
        metric_labels (dict): Mapping of metric keys to display labels.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    agg_results = aggregate_runs(run_histories, metrics)
    epochs = agg_results["epochs"]
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    colors = ['blue', 'green', 'purple']
    
    for idx, metric in enumerate(metrics):
        mean = agg_results[f"mean_{metric}"]
        std = agg_results[f"std_{metric}"]
        axes[idx].plot(epochs, mean, color=colors[idx % len(colors)], label=model_name)
        axes[idx].fill_between(epochs, mean - std, mean + std, color=colors[idx % len(colors)], alpha=0.2)
        axes[idx].set_title(metric_labels.get(metric, metric.capitalize()))
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel(metric_labels.get(metric, metric.capitalize()))
        axes[idx].grid(True)
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Training metrics plot saved to {output_path}")

def plot_comparative_metrics(
    run_histories_list,
    model_names,
    output_path="results/figures/comparative_metrics.png",
    metrics=["loss", "hr", "ndcg"],
    metric_labels={"loss": "Loss", "hr": "HR@10", "ndcg": "NDCG@10"}
):
    """Plot comparative mean and std metrics for multiple models.

    Args:
        run_histories_list (list): List of run_histories for each model.
        model_names (list): List of model names.
        output_path (str or Path): Path to save the plot.
        metrics (list): Metrics to plot.
        metric_labels (dict): Mapping of metric keys to display labels.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    agg_results_list = [aggregate_runs(run_histories, metrics) for run_histories in run_histories_list]
    epochs = agg_results_list[0]["epochs"]
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    colors = ['blue', 'green', 'purple', 'orange']
    
    for idx, metric in enumerate(metrics):
        for agg_results, model_name, color in zip(agg_results_list, model_names, colors):
            mean = agg_results[f"mean_{metric}"]
            std = agg_results[f"std_{metric}"]
            axes[idx].plot(epochs, mean, color=color, label=model_name)
            axes[idx].fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2)
        axes[idx].set_title(metric_labels.get(metric, metric.capitalize()))
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel(metric_labels.get(metric, metric.capitalize()))
        axes[idx].grid(True)
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Comparative metrics plot saved to {output_path}")

def save_run_history(history, output_path):
    """Save history of a single run to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Run history saved to {output_path}")

def load_run_histories(run_dir, num_runs=10):
    """Load histories from multiple runs."""
    run_dir = Path(run_dir)
    histories = []
    for i in range(1, num_runs + 1):
        run_path = run_dir / f"run_{i}.json"
        if run_path.exists():
            with open(run_path, 'r') as f:
                histories.append(json.load(f))
    return histories