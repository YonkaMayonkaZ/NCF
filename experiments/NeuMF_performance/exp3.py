import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess

# Ensure the script runs inside "training_metrics" folder
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Experiment settings
k_values = range(1, 11)  # K from 1 to 10
neg_values = range(1, 11)  # Num Negatives from 1 to 10
epochs = 15
num_experiments = 5  # Reduce for faster runs

# Storage for results
hr_k_results = {k: [] for k in k_values}
ndcg_k_results = {k: [] for k in k_values}
hr_neg_results = {neg: [] for neg in neg_values}
ndcg_neg_results = {neg: [] for neg in neg_values}

# Function to extract HR@K and NDCG@K from training output
def extract_metrics_from_output(output, k_values):
    hr_k_values = {k: None for k in k_values}
    ndcg_k_values = {k: None for k in k_values}

    for line in output:
        if "HR =" in line and "NDCG =" in line:
            parts = line.strip().split(",")
            try:
                hr_value = float(parts[0].split("HR =")[1].strip())
                ndcg_value = float(parts[1].split("NDCG =")[1].strip())

                k = len(hr_k_values)  # Assume results are printed in order
                if k in hr_k_values:
                    hr_k_values[k] = hr_value
                if k in ndcg_k_values:
                    ndcg_k_values[k] = ndcg_value
            except (IndexError, ValueError):
                print(f"⚠️ Skipping malformed line: {line}")

    return hr_k_values, ndcg_k_values

# Run experiments for HR@K and NDCG@K (Varying K)
for k in k_values:
    print(f"\n🔷 Running NeuMF with K = {k} 🔷")
    for _ in range(num_experiments):
        command = (
            f"python /app/main.py --batch_size=256 --lr=0.0005 --factor_num=16 "
            f"--num_layers=3 --num_ng=4 --dropout=0.05 --epochs={epochs} --model NeuMF-end --top_k={k}"
        )
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        output = process.stdout.readlines()
        process.wait()

        hr_k_values, ndcg_k_values = extract_metrics_from_output(output, k_values)
        hr_k_results[k].append(hr_k_values[k])
        ndcg_k_results[k].append(ndcg_k_values[k])

# Run experiments for HR@10 and NDCG@10 (Varying Num Negatives)
for neg in neg_values:
    print(f"\n🔷 Running NeuMF with num_negatives = {neg} 🔷")
    for _ in range(num_experiments):
        command = (
            f"python /app/main.py --batch_size=256 --lr=0.0005 --factor_num=16 "
            f"--num_layers=3 --num_ng={neg} --dropout=0.05 --epochs={epochs} --model NeuMF-end"
        )
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        output = process.stdout.readlines()
        process.wait()

        hr_k_values, ndcg_k_values = extract_metrics_from_output(output, [10])
        hr_neg_results[neg].append(hr_k_values[10])
        ndcg_neg_results[neg].append(ndcg_k_values[10])

# Compute mean and std deviation
hr_k_mean = {k: np.mean(hr_k_results[k]) for k in k_values}
hr_k_std = {k: np.std(hr_k_results[k]) for k in k_values}
ndcg_k_mean = {k: np.mean(ndcg_k_results[k]) for k in k_values}
ndcg_k_std = {k: np.std(ndcg_k_results[k]) for k in k_values}
hr_neg_mean = {neg: np.mean(hr_neg_results[neg]) for neg in neg_values}
hr_neg_std = {neg: np.std(hr_neg_results[neg]) for neg in neg_values}
ndcg_neg_mean = {neg: np.mean(ndcg_neg_results[neg]) for neg in neg_values}
ndcg_neg_std = {neg: np.std(ndcg_neg_results[neg]) for neg in neg_values}

# Function to plot results
def plot_metric(x_values, metric_mean, metric_std, xlabel, ylabel, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, list(metric_mean.values()), label=title, color="blue")
    plt.fill_between(x_values, np.array(list(metric_mean.values())) - np.array(list(metric_std.values())),
                     np.array(list(metric_mean.values())) + np.array(list(metric_std.values())), alpha=0.2, color="blue")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

# Generate the four required plots
plot_metric(k_values, hr_k_mean, hr_k_std, "K", "HR@K", "HR@K vs. K", "hr_vs_k.png")
plot_metric(k_values, ndcg_k_mean, ndcg_k_std, "K", "NDCG@K", "NDCG@K vs. K", "ndcg_vs_k.png")
plot_metric(neg_values, hr_neg_mean, hr_neg_std, "Num Negatives", "HR@10", "HR@10 vs. Num Negatives", "hr_vs_negatives.png")
plot_metric(neg_values, ndcg_neg_mean, ndcg_neg_std, "Num Negatives", "NDCG@10", "NDCG@10 vs. Num Negatives", "ndcg_vs_negatives.png")
