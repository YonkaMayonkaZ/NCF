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
from src.ncf.models import NCF
from src.utils.config import config
from src.training.metrics import metrics
from src.data.datasets import load_all, NCFData
from src.distillation import (
    ResponseDistillation,
    FeatureDistillation,
    AttentionDistillation,
    UnifiedDistillation,
)

# ------------------ Argument Parsing ------------------ #
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=config.lr, help="learning rate")
parser.add_argument("--dropout", type=float, default=config.dropout, help="dropout rate")
parser.add_argument("--batch_size", type=int, default=config.batch_size, help="batch size for training")
parser.add_argument("--epochs", type=int, default=config.epochs, help="training epochs")
parser.add_argument("--top_k", type=int, default=config.top_k, help="compute metrics@top_k")
parser.add_argument("--factor_num", type=int, default=config.factor_num // 2, help="predictive factors numbers in the student model")
parser.add_argument("--num_layers", type=int, default=config.num_layers - 1, help="number of layers in MLP for student")
parser.add_argument("--num_ng", type=int, default=config.num_ng, help="sample negative items for training")
parser.add_argument("--test_num_ng", type=int, default=config.test_num_ng, help="sample part of negative items for testing")
parser.add_argument("--out", action='store_true', default=True, help="save model or not")
parser.add_argument("--gpu", type=str, default="0", help="gpu card ID")
parser.add_argument("--teacher_model", type=str, default=config.model_type, 
                    choices=["GMF", "MLP", "NeuMF-end", "NeuMF-pre"],
                    help="Teacher model type")
parser.add_argument("--student_model", type=str, default="NeuMF-end",
                    choices=["GMF", "MLP", "NeuMF-end"],
                    help="Student model type")
parser.add_argument("--temperature", type=float, default=config.temperature, help="Temperature for distillation")
parser.add_argument("--alpha", type=float, default=config.alpha, help="Weight for BCE vs KD loss")
parser.add_argument(
    "--beta",
    type=float,
    default=0.3,
    help="Weight for feature distillation loss",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.2,
    help="Weight for attention distillation loss",
)
parser.add_argument(
    "--distillation",
    type=str,
    default="response",
    choices=["response", "feature", "attention", "unified"],
    help="Distillation strategy to use",
)
args = parser.parse_args()

# ------------------ Set GPU/CPU ------------------ #
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(" Using GPU:", torch.cuda.get_device_name(0))
else:
    print(" GPU not available. Using CPU.")

# ------------------ Load Dataset ------------------ #
train_data, test_data, user_num, item_num, train_mat = load_all()

train_dataset = NCFData(train_data, item_num, train_mat, args.num_ng, True)
test_dataset = NCFData(test_data, item_num, train_mat, 0, False)

train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_dataset, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0)

# ------------------ Load Teacher Model ------------------ #
teacher_path = config.model_dir / f"teacher_{args.teacher_model}_best.pth"
assert os.path.exists(teacher_path), f"Lack of teacher model: {teacher_path}"
teacher_model = NCF(user_num, item_num, args.factor_num * 2, args.num_layers + 1, args.dropout, args.teacher_model)
teacher_model.load_state_dict(torch.load(teacher_path, map_location=device))
teacher_model.to(device)
teacher_model.eval()

# ------------------ Create Student Model ------------------ #
student_model = NCF(user_num, item_num, args.factor_num, args.num_layers, args.dropout, args.student_model)
student_model.to(device)

# ------------------ Distillation Setup ------------------ #
if args.distillation == "response":
    distillation = ResponseDistillation(
        teacher_model,
        student_model,
        temperature=args.temperature,
        alpha=args.alpha,
    )
elif args.distillation == "feature":
    distillation = FeatureDistillation(
        teacher_model,
        student_model,
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
    )
elif args.distillation == "attention":
    distillation = AttentionDistillation(
        teacher_model,
        student_model,
        temperature=args.temperature,
        alpha=args.alpha,
        gamma=args.gamma,
    )
else:
    distillation = UnifiedDistillation(
        teacher_model,
        student_model,
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )

distillation.to(device)

optimizer = optim.Adam(student_model.parameters(), lr=args.lr)

# ------------------ TensorBoard Setup ------------------ #
writer = SummaryWriter(
    log_dir=config.log_dir
    / f"student_{args.student_model}_{args.distillation}_{time.strftime('%Y%m%d_%H%M%S')}"
)

# ------------------ Training ------------------ #
count, best_hr, best_ndcg, best_epoch = 0, 0, 0, 0
for epoch in range(args.epochs):
    distillation.train()
    start_time = time.time()
    train_loader.dataset.ng_sample()
    total_loss = 0
    num_batches = 0

    for user, item, label in train_loader:
        user = user.to(device)
        item = item.to(device)
        label = label.float().to(device)

        optimizer.zero_grad()
        loss = distillation(user, item, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        count += 1

    avg_loss = total_loss / num_batches

    student_model.eval()
    with torch.no_grad():
        HR, NDCG = metrics(student_model, test_loader, args.top_k)
    hr, ndcg = np.mean(HR), np.mean(NDCG)
    elapsed_time = time.time() - start_time
    print(f"{epoch:03d} - Loss: {avg_loss:.6f}, HR: {hr:.3f}, NDCG: {ndcg:.3f}, Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")

    # Log to TensorBoard
    writer.add_scalar("Loss/Train", avg_loss, epoch)
    writer.add_scalar(f"HR@{args.top_k}", hr, epoch)
    writer.add_scalar(f"NDCG@{args.top_k}", ndcg, epoch)

    # Save Best Model
    if hr > best_hr:
        best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
        if args.out:
            model_save_path = config.model_dir / f"student_{args.student_model}_best.pth"
            torch.save(student_model.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path}")

print(f"End. Best epoch {best_epoch:03d}: HR = {best_hr:.3f}, NDCG = {best_ndcg:.3f}")
writer.close()