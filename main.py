import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import model
import config
import evaluate
import data_utils


# ------------------ Argument Parsing ------------------ #
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
parser.add_argument("--epochs", type=int, default=20, help="training epochs")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--factor_num", type=int, default=32, help="predictive factors numbers in the model")
parser.add_argument("--num_layers", type=int, default=3, help="number of layers in MLP model")
parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
parser.add_argument("--test_num_ng", type=int, default=99, help="sample part of negative items for testing")
parser.add_argument("--out", default=True, help="save model or not")
parser.add_argument("--gpu", type=str, default="0", help="gpu card ID")
parser.add_argument("--model",
    type=str,
    default="NeuMF-end",
    choices=["GMF", "MLP", "NeuMF-end", "NeuMF-pre"],
    help="Choose the model to train: GMF, MLP, NeuMF-end, NeuMF-pre")
args = parser.parse_args()

# ------------------ Set GPU/CPU ------------------ #
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("🚀 Using GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️ GPU not available. Using CPU.")

# ------------------ Load Dataset ------------------ #
train_data, test_data, user_num, item_num, train_mat = data_utils.load_all()

train_dataset = data_utils.NCFData(train_data, item_num, train_mat, args.num_ng, True)
test_dataset = data_utils.NCFData(test_data, item_num, train_mat, 0, False)

train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_dataset, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0)

# ------------------ Create Model ------------------ #
GMF_model, MLP_model = None, None

if args.model == 'NeuMF-pre':
    GMF_model_path = os.path.join(config.model_path, f"GMF_{args.factor_num}.pth")
    MLP_model_path = os.path.join(config.model_path, f"MLP_{args.factor_num}_l{args.num_layers}.pth")

    assert os.path.exists(GMF_model_path), f"Lack of GMF model: {GMF_model_path}"
    assert os.path.exists(MLP_model_path), f"Lack of MLP model: {MLP_model_path}"

    GMF_model_state = torch.load(GMF_model_path, map_location=device)
    MLP_model_state = torch.load(MLP_model_path, map_location=device)

    GMF_model = model.NCF(user_num, item_num, args.factor_num, 1, args.dropout, 'GMF', None, None)
    GMF_model.load_state_dict(GMF_model_state)

    MLP_model = model.NCF(user_num, item_num, args.factor_num, args.num_layers, args.dropout, 'MLP', None, None)
    MLP_model.load_state_dict(MLP_model_state)

ncf_model = model.NCF(user_num, item_num, args.factor_num, args.num_layers, args.dropout, args.model, GMF_model, MLP_model)
ncf_model.to(device)

loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(ncf_model.parameters(), lr=args.lr) if args.model == 'NeuMF-pre' else optim.Adam(ncf_model.parameters(), lr=args.lr)

# ------------------ Training ------------------ #
count, best_hr, best_ndcg, best_epoch = 0, 0, 0, 0

for epoch in range(args.epochs):
    ncf_model.train()
    start_time = time.time()
    train_loader.dataset.ng_sample()

    for user, item, label in train_loader:
        user = user.to(device)
        item = item.to(device)
        label = label.float().to(device)

        ncf_model.zero_grad()
        prediction = ncf_model(user, item)
        loss = loss_function(prediction, label)
        loss.backward()
        optimizer.step()
        count += 1

    ncf_model.eval()
    HR, NDCG = evaluate.metrics(ncf_model, test_loader, args.top_k)
    avg_loss = loss.item()
    elapsed_time = time.time() - start_time
    print(f"{epoch:03d} - Loss: {avg_loss:.6}, {np.mean(HR):.3f}, {np.mean(NDCG):.3f}, {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")

    # Save Best Model
    if HR > best_hr:
        best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
        if args.out:
            if not os.path.exists(config.model_path):
                os.makedirs(config.model_path, exist_ok=True)
                # Ensure MLP models are saved with unique filenames
            if args.model == "MLP":
                model_save_path = os.path.join(config.model_path, f"MLP_{args.factor_num}_l{args.num_layers}.pth")
            else:
                model_save_path = os.path.join(config.model_path, f"{args.model}_{args.factor_num}.pth")
        torch.save(ncf_model.state_dict(), model_save_path)


print(f"End. Best epoch {best_epoch:03d}: HR = {best_hr:.3f}, NDCG = {best_ndcg:.3f}")

