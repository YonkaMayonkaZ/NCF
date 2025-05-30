import subprocess

# Config
model = "NeuMF-end"
factor_num = 16
num_layers = 3
batch_size = 256
lr = 0.0005
dropout = 0.05
top_k = 10  # fixed

log_file = "log_negatives.txt"

with open(log_file, "w") as f:
    for num_ng in range(1, 11):
        for i in range(10):
            print(f"▶️ Running num_ng={num_ng}, Run={i+1}")
            result = subprocess.run(
                f"python main.py --batch_size={batch_size} --lr={lr} --factor_num={factor_num} "
                f"--num_layers={num_layers} --dropout={dropout} --num_ng={num_ng} --top_k={top_k} --model={model}",
                shell=True, capture_output=True, text=True
            )
            f.write(f"\nnum_ng={num_ng} Run={i+1}\n")
            f.write(result.stdout)
            f.write(result.stderr)
            f.write("\n")
