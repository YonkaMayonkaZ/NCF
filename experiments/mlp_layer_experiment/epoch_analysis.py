import re
import matplotlib.pyplot as plt
from collections import defaultdict

# Parse log.txt
log_file = "log.txt"

# Structure: {(model, layer, run): [metrics per epoch]}
metrics = defaultdict(lambda: {"loss": [], "hr": [], "ndcg": []})
best_epochs = defaultdict(int)  # Store best epoch for each (model, layer, run)
current_model = None
current_layer = None
run_idx = 0

def time_to_seconds(t):
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s

with open(log_file, "r") as f:
    for line in f:
        line = line.strip()  # Remove whitespace
        
        # Detect new experiment - only reset run_idx when we see a new model/layer combo
        if "Running NeuMF-end" in line or "Running NeuMF-pre" in line:
            # Extract model type
            new_model = "NeuMF-end" if "NeuMF-end" in line else "NeuMF-pre"
            
            # Extract layer count
            layer_match = re.search(r"(\d+) layers", line)
            new_layer = int(layer_match.group(1)) if layer_match else None
            
            # Reset run index only if we're starting a new model/layer combination
            if (new_model, new_layer) != (current_model, current_layer):
                run_idx = 0
                current_model = new_model
                current_layer = new_layer
        
        # Parse epoch data - Updated regex to handle the actual format
        if re.match(r"^\d{3} - Loss:", line):
            # More flexible regex pattern to match your actual format
            match = re.search(r"(\d{3}) - Loss: ([0-9.]+), ([0-9.]+), ([0-9.]+), (\d{2}:\d{2}:\d{2})", line)
            if match and current_model and current_layer is not None:
                epoch, loss, hr, ndcg, time_str = match.groups()
                key = (current_model, current_layer, run_idx)
                metrics[key]["loss"].append(float(loss))
                metrics[key]["hr"].append(float(hr))
                metrics[key]["ndcg"].append(float(ndcg))
                
                # Debug print to verify parsing
                if run_idx == 0 and int(epoch) < 3:  # Only print first few epochs of first run
                    print(f"Parsed: {key} - Epoch {epoch}: Loss={loss}, HR={hr}, NDCG={ndcg}")
        
        # Increment run when we see the end of an experiment
        if "End. Best epoch" in line:
            # Extract best epoch number
            best_epoch_match = re.search(r"Best epoch (\d+):", line)
            if best_epoch_match and current_model and current_layer is not None:
                best_epoch = int(best_epoch_match.group(1))
                key = (current_model, current_layer, run_idx)
                best_epochs[key] = best_epoch
                print(f"Best epoch for {key}: {best_epoch}")
            
            run_idx += 1
            print(f"Completed run {run_idx-1} for {current_model} {current_layer} layers")

# Debug: Print what we parsed
print(f"\nParsed data summary:")
for key in sorted(metrics.keys()):
    model, layer, run = key
    num_epochs = len(metrics[key]["loss"])
    print(f"{model} - {layer} layers - Run {run}: {num_epochs} epochs")

# Plotting grouped by model type
for model_type in ["NeuMF-end", "NeuMF-pre"]:
    print(f"\nGenerating plots for {model_type}...")
    
    plt.figure(figsize=(12, 4))
    for layer in [1, 2, 3]:
        loss_curves = []
        for run in range(10):
            key = (model_type, layer, run)
            if key in metrics and len(metrics[key]["loss"]) > 0:
                loss_curves.append(metrics[key]["loss"])
        
        if loss_curves:
            # Calculate average across runs
            max_epochs = max(len(curve) for curve in loss_curves)
            avg_loss = []
            for epoch in range(max_epochs):
                epoch_values = [curve[epoch] for curve in loss_curves if epoch < len(curve)]
                avg_loss.append(sum(epoch_values) / len(epoch_values))
            
            plt.plot(range(1, len(avg_loss) + 1), avg_loss, label=f"MLP {layer} layers ({len(loss_curves)} runs)")
            
            # Add stars for best epochs
            best_epoch_positions = []
            for run in range(10):
                key = (model_type, layer, run)
                if key in best_epochs and key in metrics:
                    best_epoch = best_epochs[key]
                    if best_epoch < len(avg_loss):
                        best_epoch_positions.append(best_epoch + 1)  # Add 1 to match display numbering
            
            # Plot stars at average positions of best epochs
            if best_epoch_positions:
                for pos in set(best_epoch_positions):  # Remove duplicates
                    count = best_epoch_positions.count(pos)
                    actual_index = pos - 1  # Convert back to 0-based for data access
                    if actual_index < len(avg_loss):
                        plt.scatter(pos, avg_loss[actual_index], marker='*', s=100, 
                                  color='red', zorder=5, alpha=0.8)
                        # Add text annotation showing how many runs had best epoch here
                        if count > 1:
                            plt.annotate(f'{count}', (pos, avg_loss[actual_index]), 
                                       xytext=(5, 5), textcoords='offset points', 
                                       fontsize=8, color='red')
            
            print(f"  {layer} layers: {len(loss_curves)} runs, {len(avg_loss)} epochs average")
    
    plt.title(f"{model_type}: Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(range(1, 21))  # Show epochs 1 to 20
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"loss_curves_{model_type}.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(12, 4))
    for layer in [1, 2, 3]:
        hr_curves = []
        for run in range(10):
            key = (model_type, layer, run)
            if key in metrics and len(metrics[key]["hr"]) > 0:
                hr_curves.append(metrics[key]["hr"])
        
        if hr_curves:
            max_epochs = max(len(curve) for curve in hr_curves)
            avg_hr = []
            for epoch in range(max_epochs):
                epoch_values = [curve[epoch] for curve in hr_curves if epoch < len(curve)]
                avg_hr.append(sum(epoch_values) / len(epoch_values))
            
            plt.plot(range(1, len(avg_hr) + 1), avg_hr, label=f"MLP {layer} layers ({len(hr_curves)} runs)")
            
            # Add stars for best epochs
            best_epoch_positions = []
            for run in range(10):
                key = (model_type, layer, run)
                if key in best_epochs and key in metrics:
                    best_epoch = best_epochs[key]
                    if best_epoch < len(avg_hr):
                        best_epoch_positions.append(best_epoch + 1)  # Add 1 to match display numbering
            
            # Plot stars at average positions of best epochs
            if best_epoch_positions:
                for pos in set(best_epoch_positions):  # Remove duplicates
                    count = best_epoch_positions.count(pos)
                    actual_index = pos - 1  # Convert back to 0-based for data access
                    if actual_index < len(avg_hr):
                        plt.scatter(pos, avg_hr[actual_index], marker='*', s=100, 
                                  color='red', zorder=5, alpha=0.8)
                        # Add text annotation showing how many runs had best epoch here
                        if count > 1:
                            plt.annotate(f'{count}', (pos, avg_hr[actual_index]), 
                                       xytext=(5, 5), textcoords='offset points', 
                                       fontsize=8, color='red')
    
    plt.title(f"{model_type}: HR@10 per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("HR@10")
    plt.xticks(range(1, 21))  # Show epochs 1 to 20
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"hr_curves_{model_type}.png", dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(12, 4))
    for layer in [1, 2, 3]:
        ndcg_curves = []
        for run in range(10):
            key = (model_type, layer, run)
            if key in metrics and len(metrics[key]["ndcg"]) > 0:
                ndcg_curves.append(metrics[key]["ndcg"])
        
        if ndcg_curves:
            max_epochs = max(len(curve) for curve in ndcg_curves)
            avg_ndcg = []
            for epoch in range(max_epochs):
                epoch_values = [curve[epoch] for curve in ndcg_curves if epoch < len(curve)]
                avg_ndcg.append(sum(epoch_values) / len(epoch_values))
            
            plt.plot(range(1, len(avg_ndcg) + 1), avg_ndcg, label=f"MLP {layer} layers ({len(ndcg_curves)} runs)")
            
            # Add stars for best epochs
            best_epoch_positions = []
            for run in range(10):
                key = (model_type, layer, run)
                if key in best_epochs and key in metrics:
                    best_epoch = best_epochs[key]
                    if best_epoch < len(avg_ndcg):
                        best_epoch_positions.append(best_epoch + 1)  # Add 1 to match display numbering
            
            # Plot stars at average positions of best epochs
            if best_epoch_positions:
                for pos in set(best_epoch_positions):  # Remove duplicates
                    count = best_epoch_positions.count(pos)
                    actual_index = pos - 1  # Convert back to 0-based for data access
                    if actual_index < len(avg_ndcg):
                        plt.scatter(pos, avg_ndcg[actual_index], marker='*', s=100, 
                                  color='red', zorder=5, alpha=0.8)
                        # Add text annotation showing how many runs had best epoch here
                        if count > 1:
                            plt.annotate(f'{count}', (pos, avg_ndcg[actual_index]), 
                                       xytext=(5, 5), textcoords='offset points', 
                                       fontsize=8, color='red')
    
    plt.title(f"{model_type}: NDCG per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("NDCG")
    plt.xticks(range(1, 21))  # Show epochs 1 to 20
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"ndcg_curves_{model_type}.png", dpi=300, bbox_inches='tight')
    plt.show()

print("✅ Grouped plots saved for each metric and model type.")