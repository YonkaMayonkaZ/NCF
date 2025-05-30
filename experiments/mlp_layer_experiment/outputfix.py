import re
import numpy as np

# Adjust this if your log file has a different name
log_file = "log.txt"

# Key structure: (model_type, num_layers) -> list of HRs
results = {}

with open(log_file, "r") as f:
    current_model = None
    current_layer = None
    for line in f:
        # Detect new experiment block
        if "Running NeuMF" in line:
            if "NeuMF-end" in line:
                current_model = "NeuMF-end"
            elif "NeuMF-pre" in line:
                current_model = "NeuMF-pre"
            match = re.search(r"(\d+) layers", line)
            if match:
                current_layer = int(match.group(1))

        # Extract HR value from summary line
        match = re.search(r"HR = ([0-9.]+)", line)
        if match and current_model is not None and current_layer is not None:
            hr = float(match.group(1))
            key = (current_model, current_layer)
            results.setdefault(key, []).append(hr)

# Compute mean and std, and save to file
output_file = "experiment_mlp_layers_results.txt"
with open(output_file, "w") as f:
    for (model, layer), hr_list in sorted(results.items()):
        mean = np.mean(hr_list)
        std = np.std(hr_list)
        f.write(f"{model} | MLP Layers: {layer} | HR@10 = {mean:.3f} ± {std:.3f}\n")

print(f"✅ Results written to {output_file}")