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
from src.data.datasets import NCFData, load_all
from src.utils.config import config
from src.training.metrics import metrics
from src.utils.visualization import plot_training_metrics

def train_teacher(model_type, user_num, item_num, train_mat, device, args):
    # Load dataset
    train_data, test_data, _, _, _ = load_all()
    train_dataset = NCFData(train_data, item_num, train_mat, args.num_ng, True)
    test_dataset = NCFData(test_data, item_num, train_mat, 0, False)
    
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0)

    # Initialize model
    model = NCF(user_num, item_num, args.factor_num, args.num_layers, args.dropout, model_type)
    
    # Load pretrained weights for NeuMF-pre
    if model_type == "NeuMF-pre":
        gmf_path = config.model_dir / "GMF_best.pth"
        mlp_path = config.model_dir / "MLP_best.pth"
        if gmf_path.exists() and mlp_path.exists():
            gmf_state = torch.load(gmf_path)
            mlp_state = torch.load(mlp_path)
            # Map GMF and MLP weights to NeuMF
            model.load_pretrain_weights(gmf_state, mlp_state)
            print(f"Loaded pretrained GMF and MLP weights for NeuMF-pre")
        else:
            raise FileNotFoundError("Pretrained GMF or MLP weights not found")

    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # TensorBoard setup
    writer = SummaryWriter(log_dir=config.log_dir / f"teacher_{model_type}_{time.strftime('%Y%m%d_%H%M%S')}")

    # Training loop
    count, best_hr = 0, 0
    best_loss, best_ndcg, best_epoch = 0, 0, 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()
        train_loader.dataset.ng_sample()
        total_loss = 0
        num_batches = 0

        for user, item, label in train_loader:
            user = user.to(device)
            item = item.to(device)
            label = label.float().to(device)

            optimizer.zero_grad()
            prediction = model(user, item)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            count += 1

        avg_loss = total_loss / num_batches

        # Evaluation
        model.eval()
        with torch.no_grad():
            HR, NDCG = metrics(model, test_loader, args.top_k)
        hr, ndcg = np.mean(HR), np.mean(NDCG)
        elapsed_time = time.time() - start_time

        # Log to TensorBoard
        writer.add_scalar("Loss/Train", avg_loss, epoch)
        writer.add_scalar(f"HR@{args.top_k}", hr, epoch)
        writer.add_scalar(f"NDCG@{args.top_k}", ndcg, epoch)

        print(f"Epoch {epoch+1:03d}: Loss={avg_loss:.4f}, HR={hr:.3f}, NDCG={ndcg:.3f}, Time={time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")

        # Track history for plotting
        history.append({"epoch": epoch + 1, "loss": avg_loss, "hr": hr, "ndcg": ndcg})

        # Save best model
        if hr > best_hr:
            best_hr, best_ndcg, best_loss, best_epoch = hr, ndcg, avg_loss, epoch
            if args.out:
                model_save_path = config.model_dir / f"teacher_{model_type}_best.pth"
                torch.save(model.state_dict(), model_save_path)
                print(f"Saved best model to {model_save_path}")

    # Plot training metrics
    plot_training_metrics(
        run_histories=[history],  # Single run
        model_name=f"Teacher_{model_type}",
        output_path=config.figure_dir / f"teacher_{model_type}_metrics.png"
    )

    writer.close()
    return best_loss, best_hr, best_ndcg, best_epoch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=config.lr, help="learning rate")
    parser.add_argument("--dropout", type=float, default=config.dropout, help="dropout rate")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="batch size")
    parser.add_argument("--epochs", type=int, default=config.epochs, help="training epochs")
    parser.add_argument("--top_k", type=int, default=config.top_k, help="compute metrics@top_k")
    parser.add_argument("--factor_num", type=int, default=config.factor_num, help="predictive factors")
    parser.add_argument("--num_layers", type=int, default=config.num_layers, help="number of layers in MLP")
    parser.add_argument("--num_ng", type=int, default=config.num_ng, help="sample negative items for training")
    parser.add_argument("--test_num_ng", type=int, default=config.test_num_ng, help="sample negative items for testing")
    parser.add_argument("--out", action="store_true", default=True, help="save model")
    parser.add_argument("--gpu", type=str, default="0", help="gpu card ID")
    parser.add_argument("--model", type=str, default="NeuMF-end", choices=["NeuMF-end", "NeuMF-pre"], help="model type")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    best_loss, best_hr, best_ndcg, best_epoch = train_teacher(
        model_type=args.model,
        user_num=config.user_num,
        item_num=config.item_num,
        train_mat=load_all()[4],
        device=device,
        args=args
    )

    print(f"Best Epoch {best_epoch:03d}: Loss={best_loss:.4f}, HR={best_hr:.3f}, NDCG={best_ndcg:.3f}")

if __name__ == "__main__":
    main()