import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess

# Experiment Settings
num_experiments = 10  # Run each setting 10 times
mlp_layers = [1, 2, 3]  # Different MLP layers to test allaksto
factor_num = 16  # Number of predictive factors
num_ng = 8  # Number of negative samples per positive sample
lr = 0.0005  # Learning rate
dropout = 0.05  # Dropout rate


# Step 1: Pretrain GMF Model if not already saved
gmf_model_path = f"/app/models/GMF_{factor_num}.pth"
if not os.path.exists(gmf_model_path):
    print(f"⚠️ GMF model not found. Training GMF model...")
    subprocess.run(f"python /app/main.py --batch_size=256 --lr={lr} --factor_num={factor_num} --num_layers=1 --num_ng={num_ng} --dropout={dropout} --model GMF", shell=True)

# Step 2: Pretrain MLP Models for Each MLP Layer Setting
for layers in mlp_layers:
    mlp_model_path = f"/app/models/MLP_{factor_num}_l{layers}.pth"
    if not os.path.exists(mlp_model_path):
        print(f"🟡 Training MLP model with {layers} layers...")
        subprocess.run(f"python /app/main.py --batch_size=256 --lr={lr} --factor_num={factor_num} --num_layers={layers} --num_ng={num_ng} --dropout={dropout} --model MLP", shell=True)

# Function to run NeuMF model training
def run_neumf_training(model_type, layers):
    hr_scores = []  # Store HR@10 scores for multiple runs
    mlp_model_path = f"/app/models/MLP_{factor_num}_l{layers}.pth"

    # Ensure MLP model exists before training NeuMF
    if not os.path.exists(mlp_model_path):
        print(f"⚠️ Error: MLP model {mlp_model_path} not found. Skipping NeuMF training for {model_type} with {layers} layers.")
        return

    for exp in range(num_experiments):
        command = (
            f"python /app/main.py --batch_size=256 --lr={lr} --factor_num={factor_num} "
            f"--num_layers={layers} --num_ng={num_ng} --dropout={dropout} --model {model_type} "
            )

        # Run the command
        process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        # Debugging outputs
        print(f"🔹 Running command: {command}")
        print("🔹 Command Output:\n", process.stdout)
        print("🔹 Command Error:\n", process.stderr)

        output = process.stdout

# Step 3: Train & Evaluate NeuMF Models Separately

for layers in mlp_layers:
   print(f"\n🔷 Running NeuMF-end for {layers} layers 🔷")
   run_neumf_training("NeuMF-end", layers)

for layers in mlp_layers:
    print(f"\n🔷 Running NeuMF-pre for {layers} layers 🔷")
    run_neumf_training("NeuMF-pre", layers)

