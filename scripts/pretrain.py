#!/usr/bin/env python3
"""
Clean pretrain script for GMF and MLP models.
Usage: python scripts/pretrain.py --model GMF --epochs 20
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

# Add the project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ncf.models import NCF
from src.data.datasets import NCFData, load_all
from src.utils.config import config
from src.training.metrics import metrics

def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model_type, args, device):
    """Train a single model (GMF or MLP)."""
    print(f"\nTraining {model_type} model...")
    
    # Load dataset
    train_data, test_data, user_num, item_num, train_mat = load_all()
    train_dataset = NCFData(train_data, item_num, train_mat, args.num_ng, True)
    test_dataset = NCFData(test_data, item_num, train_mat, 0, False)
    
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0)

    # Initialize model
    model = NCF(user_num, item_num, args.factor_num, args.num_layers, args.dropout, model_type)
    model.to(device)
    
    # Print model info
    param_count = count_parameters(model)
    print(f"Model: {model_type}")
    print(f"Parameters: {param_count:,}")
    print(f"Factor num: {args.factor_num}")
    print(f"Layers: {args.num_layers}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_hr = 0
    best_loss, best_ndcg, best_epoch = 0, 0, 0
    
    print(f"Training for {args.epochs} epochs...")
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

        avg_loss = total_loss / num_batches

        # Evaluation
        model.eval()
        with torch.no_grad():
            HR, NDCG = metrics(model, test_loader, args.top_k)
        hr, ndcg = np.mean(HR), np.mean(NDCG)
        elapsed_time = time.time() - start_time

        print(f"Epoch {epoch+1:03d}: Loss={avg_loss:.4f}, HR={hr:.3f}, NDCG={ndcg:.3f}, Time={elapsed_time:.1f}s")

        # Save best model
        if hr > best_hr:
            best_hr, best_ndcg, best_loss, best_epoch = hr, ndcg, avg_loss, epoch
            if args.save:
                # Create descriptive filename with key parameters
                if model_type == "MLP":
                    # Format: MLP_3l_32f_best.pth (3 layers, 32 factors)
                    model_filename = f"{model_type}_{args.num_layers}l_{args.factor_num}f_best.pth"
                else:
                    # Format: GMF_32f_best.pth (32 factors)
                    model_filename = f"{model_type}_{args.factor_num}f_best.pth"
                
                model_save_path = config.model_dir / model_filename
                torch.save(model.state_dict(), model_save_path)
                print(f"Saved best model to {model_save_path}")

    print(f"\nTraining completed!")
    print(f"Best HR@{args.top_k}: {best_hr:.4f} at epoch {best_epoch+1}")
    return best_hr, best_ndcg, param_count

def main():
    parser = argparse.ArgumentParser(description='Train GMF or MLP model')
    parser.add_argument("--model", type=str, required=True, choices=["GMF", "MLP"], 
                       help="Model type to train")
    parser.add_argument("--epochs", type=int, default=config.epochs, 
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=config.lr, 
                       help="Learning rate")
    parser.add_argument("--dropout", type=float, default=config.dropout, 
                       help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, 
                       help="Batch size")
    parser.add_argument("--top_k", type=int, default=config.top_k, 
                       help="Top-K for evaluation")
    parser.add_argument("--factor_num", type=int, default=config.factor_num, 
                       help="Number of latent factors")
    parser.add_argument("--num_layers", type=int, default=config.num_layers, 
                       help="Number of MLP layers")
    parser.add_argument("--num_ng", type=int, default=config.num_ng, 
                       help="Number of negative samples for training")
    parser.add_argument("--test_num_ng", type=int, default=config.test_num_ng, 
                       help="Number of negative samples for testing")
    parser.add_argument("--save", action="store_true", default=True, 
                       help="Save the best model")
    parser.add_argument("--gpu", type=str, default="0", 
                       help="GPU device ID")
    
    args = parser.parse_args()

    # Set GPU/CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # Load data info
    _, _, user_num, item_num, train_mat = load_all()
    print(f"\nDataset Info:")
    print(f"Users: {user_num}, Items: {item_num}")
    print(f"Training interactions: {len(train_mat.nonzero()[0])}")

    # Train model
    best_hr, best_ndcg, param_count = train_model(args.model, args, device)

    print(f"\nFinal Results:")
    print(f"Model: {args.model}")
    print(f"HR@{args.top_k}: {best_hr:.4f}")
    print(f"NDCG@{args.top_k}: {best_ndcg:.4f}")
    print(f"Parameters: {param_count:,}")

if __name__ == "__main__":
    main()