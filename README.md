# Neural Collaborative Filtering (NCF) — Reproduction & Extension

This project reproduces and extends the results of the Neural Collaborative Filtering (NCF) model as proposed by He et al. (2017), using the MovieLens 100K dataset (`u.data`) and a full Dockerized pipeline.

## 📁 Project Structure

```
NCF/
├── main.py                 # Core training/evaluation script
├── model.py                # GMF, MLP, and NeuMF model definitions
├── evaluate.py             # Evaluation metrics (HR@K, NDCG@K)
├── data_utils.py           # Dataset loading and negative sampling
├── config.py               # Argument parser for training options
├── Dockerfile              # Environment setup for Docker
├── experiments/
│   ├── experiment1.py              # HR@10 vs MLP Layers (NeuMF-end & NeuMF-pre)
│   ├── mlp_layers_exp.py           # Runs NeuMF-pre and NeuMF-end for MLP layers 1–3
│   ├── train_eval_by_k.py          # HR@K and NDCG@K for K = 1 to 10
│   ├── train_eval_by_negatives.py # HR@10 and NDCG@10 vs num_negatives
│   ├── plot_eval_k_negatives.py   # Parses logs and plots final evaluations
├── models/                # Saved model checkpoints (*.pth)
├── log.txt                # Collected training logs
```

## 🚀 How to Run

### Step 1: Docker Build & Run

```bash
docker build -t ncf-docker .
docker run --rm -it -v "$PWD":/app ncf-docker /bin/bash
```

### Step 2: Train Base Models

```bash
python main.py --model GMF
python main.py --model MLP --num_layers=3
python main.py --model NeuMF-end --num_layers=3
```

### Step 3: Run Layer Sensitivity Experiment

```bash
python experiments/mlp_layers_exp.py | tee log.txt
python experiments/plot_metrics_per_epoch.py
```

### Step 4: Run Final Evaluation Experiments

```bash
python experiments/train_eval_by_k.py
python experiments/train_eval_by_negatives.py
python experiments/plot_eval_k_negatives.py
```

## 📊 Experimental Goals

### 1. HR\@10 vs MLP Layer Depth

Compare NeuMF-pre vs NeuMF-end for MLP depths 1–3.

### 2. HR\@K & NDCG\@K vs K (Top-K)

Measure recommendation quality for K from 1 to 10.

### 3. HR\@10 & NDCG\@10 vs Number of Negatives

Study impact of negative sampling volume on performance.

### 4. Parameter Count

Analyze model complexity as MLP depth increases.

## 📈 Output Plots

* HR\@K vs K → `hr_vs_k.png`
* NDCG\@K vs K → `ndcg_vs_k.png`
* HR\@10 vs negatives → `hr_vs_negatives.png`
* NDCG\@10 vs negatives → `ndcg_vs_negatives.png`
* Loss / HR / NDCG per epoch (for each model and depth)

## 🧠 Credits

* Based on the [Neural Collaborative Filtering (NCF) paper](https://arxiv.org/abs/1708.05031)
* Implemented and extended for reproducibility and evaluation analysis by **Δανιήλ Μαυρουδής**, AEM 2572
