import re
import numpy as np
import matplotlib.pyplot as plt

log_file = "log.txt"

# Containers
metrics = {}
current_model = None
current_layer = None
current_times = []
current_epochs = []
current_losses = []

with open(log_file, "r") as f:
    for line in f:
        # Detect model type and MLP layer
        if "Running NeuMF" in line:
            if "NeuMF-end" in line:
                current_model = "NeuMF-end"
            elif "NeuMF-pre" in line:
                current_model = "NeuMF-pre"
            match = re.search(r"(\d+) layers", line)
            if match:
                current_layer = int(match.group(1))
            key = (current_model, current_layer)
            metrics.setdefault(key, {"hr": [], "ndcg": [], "best_epoch": [], "time": [], "losses": []})

        # Extract per-epoch line
        match = re.match(r"\d+ - Loss: ([0-9.]+), ([0-9.]+), ([0-9.]+), ([0-9:]+)", line)
        if match:
            loss, hr, ndcg, t = match.groups()
            current_losses.append(float(loss))
            current_times.append(t)

        # End-of-run line
        match = re.search(r"End\. Best epoch (\d+): HR = ([0-9.]+), NDCG = ([0-9.]+)", line)
        if match:
            best_epoch, hr, ndcg = match.groups()
            key = (current_model, current_layer)
            metrics[key]["hr"].append(float(hr))
            metrics[key]["ndcg"].append(float(ndcg))
            metrics[key]["best_epoch"].append(int(best_epoch))
            metrics[key]["losses"].append(current_losses)
            # Convert time strings to seconds
            secs = [int(x.split(":")[-1]) + int(x.split(":")[-2]) * 60 for x in current_times]
            metrics[key]["time"].append(sum(secs)/len(secs))
            current_losses, current_times = [], []

# --- HR@10 Plot ---
layers = [1, 2, 3]
hr_means_end = [np.mean(metrics[("NeuMF-end", l)]["hr"]) for l in layers]
hr_stds_end = [np.std(metrics[("NeuMF-end", l)]["hr"]) for l in layers]
hr_means_pre = [np.mean(metrics[("NeuMF-pre", l)]["hr"]) for l in layers]
hr_stds_pre = [np.std(metrics[("NeuMF-pre", l)]["hr"]) for l in layers]

plt.figure(figsize=(8, 5))
plt.errorbar(layers, hr_means_end, yerr=hr_stds_end, fmt="-o", label="NeuMF-end", capsize=5)
plt.errorbar(layers, hr_means_pre, yerr=hr_stds_pre, fmt="-s", label="NeuMF-pre", capsize=5)
plt.title("HR@10 vs MLP Layers")
plt.xlabel("MLP Layers")
plt.ylabel("HR@10")
plt.legend()
plt.grid(True)
plt.savefig("hr_vs_layers.png")
plt.show()

# --- Loss Plots ---
for model in ["NeuMF-end", "NeuMF-pre"]:
    plt.figure(figsize=(8, 5))
    for layer in layers:
        key = (model, layer)
        if key in metrics:
            losses = metrics[key]["losses"]
            avg_loss = np.mean(losses, axis=0)
            plt.plot(avg_loss, label=f"{model} L{layer}")
    plt.title(f"Loss Curve: {model}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"loss_curve_{model.replace('-', '_')}.png")
    plt.show()