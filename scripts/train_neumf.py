#!/usr/bin/env python3
"""
Clean train NeuMF script.
Usage: python scripts/train_neumf.py --model NeuMF-end --num_layers 3
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

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ncf.models import NCF
from src.data.datasets import NCFData, load_all
from src.utils.config import config
from src.training.metrics import metrics

def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def find_pretrained_model(model_type, num_layers, factor_num):
    """Find pretrained model file based on naming convention."""
    if model_type == "GMF":
        filename = f"GMF_{factor_num}f_best.pth"
    elif model_type == "MLP":
        filename = f"MLP_{num_layers}l_{factor_num}f_best.pth"
    else:
        return None
    
    model_path = config.model_dir / filename
    return model_path if model_path.exists() else None

def train_neumf(args, device):
    """Train a single NeuMF model."""
    print(f"\nTraining {args.model} with {args.num_layers} layers...")
    print(f"Pretraining: {'Yes' if args.pretraining else 'No'}")
    
    # Load data
    train_data, test_data, user_num, item_num, train_mat = load_all()
    print(f"Dataset: {user_num} users, {item_num} items")
    
    # Create datasets
    train_dataset = NCFData(train_data, item_num, train_mat, args.num_ng, True)
    test_dataset = NCFData(test_data, item_num, train_mat, 0, False)
    
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0)
    
    # Create model
    model = NCF(user_num, item_num, args.factor_num, args.num_layers, args.dropout, args.model)
    
    # Load pretrained weights if requested
    if args.pretraining:
        gmf_path = find_pretrained_model("GMF", None, args.factor_num)
        mlp_path = find_pretrained_model("MLP", args.num_layers, args.factor_num)
        
        if gmf_path and mlp_path and gmf_path.exists() and mlp_path.exists():
            print("Loading pretrained weights...")
            gmf_state = torch.load(gmf_path, map_location=device)
            mlp_state = torch.load(mlp_path, map_location=device)
            model.load_pretrain_weights(gmf_state, mlp_state)
            print("Pretrained weights loaded successfully")
        else:
            print(f"Warning: Pretrained weights not found!")
            print(f"GMF path: {gmf_path} (exists: {gmf_path and gmf_path.exists()})")
            print(f"MLP path: {mlp_path} (exists: {mlp_path and mlp_path.exists()})")
            print("Training without pretraining...")
            args.pretraining = False
    
    model.to(device)
    
    # Count parameters
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,}")
    
    # Setup training (use SGD for pretrained as in paper)
    criterion = nn.BCEWithLogitsLoss()
    if args.pretraining:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_hr = 0
    best_ndcg = 0
    best_epoch = 0
    
    # Training loop
    print(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # Training phase
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
        
        # Evaluation phase
        model.eval()
        with torch.no_grad():
            HR, NDCG = metrics(model, test_loader, args.top_k)
        
        hr = np.mean(HR)
        ndcg = np.mean(NDCG)
        elapsed_time = time.time() - start_time
        
        print(f"Epoch {epoch+1:03d}: Loss={avg_loss:.4f}, HR={hr:.3f}, NDCG={ndcg:.3f}, Time={elapsed_time:.1f}s")
        
        # Save best model
        if hr > best_hr:
            best_hr = hr
            best_ndcg = ndcg
            best_epoch = epoch + 1
            if args.save:
                pretrain_suffix = "pre" if args.pretraining else "end"
                # Format: NeuMF_end_3l_32f_best.pth (end/pre, 3 layers, 32 factors)
                model_filename = f"NeuMF_{pretrain_suffix}_{args.num_layers}l_{args.factor_num}f_best.pth"
                model_path = config.model_dir / model_filename
                torch.save(model.state_dict(), model_path)
                print(f"Saved best model: {model_filename}")
    
    print(f"\nTraining completed!")
    print(f"Best Result: Epoch {best_epoch:03d}: HR={best_hr:.3f}, NDCG={best_ndcg:.3f}")
    
    # Output results in parseable format for experiments
    print(f"\n--- RESULTS ---")
    print(f"Model: {args.model}")
    print(f"Layers: {args.num_layers}")
    print(f"Pretraining: {args.pretraining}")
    print(f"HR@{args.top_k}: {best_hr}")
    print(f"NDCG@{args.top_k}: {best_ndcg}")
    print(f"Parameters: {param_count}")
    print(f"--- END RESULTS ---")
    
    return {
        'best_hr': best_hr,
        'best_ndcg': best_ndcg,
        'best_epoch': best_epoch,
        'parameters': param_count,
        'num_layers': args.num_layers,
        'pretraining': args.pretraining,
        'model_type': args.model
    }

def main():
    parser = argparse.ArgumentParser(description='Train NeuMF model')
    parser.add_argument('--model', type=str, default='NeuMF-end', 
                       choices=['NeuMF-end', 'NeuMF-pre'], help='Model type')
    parser.add_argument('--epochs', type=int, default=config.epochs, 
                       help='Number of training epochs')
    parser.add_argument('--factor_num', type=int, default=config.factor_num, 
                       help='Number of latent factors')
    parser.add_argument('--num_layers', type=int, default=config.num_layers, 
                       help='Number of MLP layers')
    parser.add_argument('--pretraining', action='store_true', 
                       help='Use pretrained weights (loads GMF and MLP models)')
    parser.add_argument('--lr', type=float, default=config.lr, 
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, 
                       help='Batch size')
    parser.add_argument('--dropout', type=float, default=config.dropout, 
                       help='Dropout rate')
    parser.add_argument('--num_ng', type=int, default=config.num_ng, 
                       help='Number of negative samples')
    parser.add_argument('--test_num_ng', type=int, default=config.test_num_ng, 
                       help='Number of negative samples for testing')
    parser.add_argument('--top_k', type=int, default=config.top_k, 
                       help='Top-K for evaluation')
    parser.add_argument('--save', action='store_true', default=True, 
                       help='Save best model')
    parser.add_argument('--gpu', type=str, default="0", 
                       help='GPU device ID')
    
    args = parser.parse_args()

    # Set GPU/CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # Train model
    result = train_neumf(args, device)
    
    print(f"\nFinal Results:")
    print(f"Model: {result['model_type']}")
    print(f"Layers: {result['num_layers']}")
    print(f"Pretraining: {result['pretraining']}")
    print(f"HR@{args.top_k}: {result['best_hr']:.4f}")
    print(f"NDCG@{args.top_k}: {result['best_ndcg']:.4f}")
    print(f"Parameters: {result['parameters']:,}")

if __name__ == "__main__":
    main()