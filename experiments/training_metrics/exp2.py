import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure the script runs inside "training_metrics" folder
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# File containing the training logs
output_file = "output.txt"

# Experiment settings
mlp_layers = [1, 2, 3]
epochs = 20
num_experiments = 10  # Ensure this matches your actual number of experiments

# Storage for training loss, HR@10, and NDCG@10
loss_data = {layers: np.zeros((num_experiments, epochs)) for layers in mlp_layers}
hr_data = {layers: np.zeros((num_experiments, epochs)) for layers in mlp_layers}
ndcg_data = {layers: np.zeros((num_experiments, epochs)) for layers in mlp_layers}

# Read the output file and extract metrics
with open(output_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

current_layer = None
experiment_counter = {layer: 0 for layer in mlp_layers}  # Track experiments per MLP layer

for line in lines:
    line = line.strip()

    # Detect the start of a new experiment
    if "🔷 Running NeuMF training with" in line:
        for layer in mlp_layers:
            if f"{layer} MLP layers" in line:
                current_layer = layer
                experiment_counter[current_layer] += 1
                epoch_idx = 0  # Reset epoch counter for each new experiment
                break

    # Extract loss, HR@10, and NDCG@10
    elif "Loss:" in line and current_layer is not None:
        try:
            parts = line.split(",")
            loss_value = float(parts[0].split("Loss:")[1].strip())
            hr_value = float(parts[1].strip())
            ndcg_value = float(parts[2].strip())

            # Ensure experiment index is within array bounds
            exp_idx = experiment_counter[current_layer] - 1  # Zero-based index
            if exp_idx < num_experiments and epoch_idx < epochs:
                loss_data[current_layer][exp_idx, epoch_idx] = loss_value
                hr_data[current_layer][exp_idx, epoch_idx] = hr_value
                ndcg_data[current_layer][exp_idx, epoch_idx] = ndcg_value

            epoch_idx += 1  # Increment epoch index only when data is correctly stored
        except (IndexError, ValueError) as e:
            print(f"⚠️ Skipping malformed line: {line}. Error: {e}")

# Compute mean and standard deviation
loss_means = {layer: np.mean(loss_data[layer], axis=0) for layer in mlp_layers}
loss_stds = {layer: np.std(loss_data[layer], axis=0) for layer in mlp_layers}
hr_means = {layer: np.mean(hr_data[layer], axis=0) for layer in mlp_layers}
hr_stds = {layer: np.std(hr_data[layer], axis=0) for layer in mlp_layers}
ndcg_means = {layer: np.mean(ndcg_data[layer], axis=0) for layer in mlp_layers}
ndcg_stds = {layer: np.std(ndcg_data[layer], axis=0) for layer in mlp_layers}

# Function to plot results for each MLP layer
def plot_metric(layer, metric_means, metric_stds, ylabel, title, filename):
    plt.figure(figsize=(8, 5))
    epochs_range = range(1, epochs + 1)

    mean_values = metric_means[layer]
    std_values = metric_stds[layer]

    plt.plot(epochs_range, mean_values, label=f"MLP Layers: {layer}", color="blue")
    plt.fill_between(epochs_range, mean_values - std_values, mean_values + std_values, alpha=0.2, color="blue")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{title} - MLP Layers: {layer}")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

# Generate separate plots for each MLP layer
for layer in mlp_layers:
    plot_metric(layer, loss_means, loss_stds, "Training Loss", "Training Loss per Epoch", f"training_loss_layer_{layer}.png")
    plot_metric(layer, hr_means, hr_stds, "HR@10", "HR@10 per Epoch", f"hr10_per_epoch_layer_{layer}.png")
    plot_metric(layer, ndcg_means, ndcg_stds, "NDCG@10", "NDCG@10 per Epoch", f"ndcg10_per_epoch_layer_{layer}.png")

