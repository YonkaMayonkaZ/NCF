import subprocess

# Config
model = "NeuMF-end"
factor_num = 16
num_layers = 3
batch_size = 256
lr = 0.0005
dropout = 0.05
num_ng = 8  # fixed

log_file = "log_k.txt"

with open(log_file, "w") as f:
    for k in range(1, 11):
        print(f"▶️ Running K={k}")
        result = subprocess.run(
            f"python /app/main.py --batch_size={batch_size} --lr={lr} --factor_num={factor_num} "
            f"--num_layers={num_layers} --dropout={dropout} --num_ng={num_ng} --top_k={k} --model={model}",
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )

        f.write(f"\nK={k}")
        f.write(result.stdout)
        f.write(result.stderr)
        f.write("\n")
