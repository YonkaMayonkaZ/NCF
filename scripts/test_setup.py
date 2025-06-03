#!/usr/bin/env python3
"""
Test script to verify the NCF setup is working correctly.
Run this script to check if all dependencies and modules are properly installed.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.utils.config import config
        print("[OK] Configuration loaded successfully")
        print(f"   - Users: {config.user_num}")
        print(f"   - Items: {config.item_num}")
        print(f"   - Factors: {config.factor_num}")
    except Exception as e:
        print(f"[ERROR] Config import failed: {e}")
        return False
    
    try:
        from src.ncf.models import NCF
        print("[OK] NCF model imported successfully")
    except Exception as e:
        print(f"[ERROR] NCF model import failed: {e}")
        return False
    
    try:
        from src.data.datasets import load_all, NCFData
        print("[OK] Dataset modules imported successfully")
    except Exception as e:
        print(f"[ERROR] Dataset import failed: {e}")
        return False
    
    try:
        from src.training.metrics import metrics
        print("[OK] Metrics module imported successfully")
    except Exception as e:
        print(f"[ERROR] Metrics import failed: {e}")
        return False
    
    try:
        from src.distillation import ResponseDistillation, FeatureDistillation, AttentionDistillation
        print("[OK] Distillation modules imported successfully")
    except Exception as e:
        print(f"[ERROR] Distillation import failed: {e}")
        return False
    
    try:
        from src.ncf.nmf_model import NMFRecommender
        print("[OK] NMF model imported successfully")
    except Exception as e:
        print(f"[ERROR] NMF model import failed: {e}")
        return False
    
    return True

def test_data_loading():
    """Test if data can be loaded properly."""
    print("\n Testing data loading...")
    
    try:
        from src.data.datasets import load_all
        train_data, test_data, user_num, item_num, train_mat = load_all()
        
        print(f" Data loaded successfully:")
        print(f"   - Training samples: {len(train_data)}")
        print(f"   - Test samples: {len(test_data)}")
        print(f"   - Users: {user_num}")
        print(f"   - Items: {item_num}")
        print(f"   - Training matrix shape: {train_mat.shape}")
        print(f"   - Training matrix density: {train_mat.nnz / (train_mat.shape[0] * train_mat.shape[1]):.4f}")
        
        return True
    except Exception as e:
        print(f" Data loading failed: {e}")
        return False

def test_model_creation():
    """Test if models can be created and run."""
    print("\n Testing model creation...")
    
    try:
        from src.ncf.models import NCF
        
        # Test GMF
        gmf_model = NCF(943, 1682, 32, 3, 0.0, "GMF")
        print(f" GMF model created: {sum(p.numel() for p in gmf_model.parameters()):,} parameters")
        
        # Test MLP
        mlp_model = NCF(943, 1682, 32, 3, 0.0, "MLP")
        print(f" MLP model created: {sum(p.numel() for p in mlp_model.parameters()):,} parameters")
        
        # Test NeuMF-end
        neumf_model = NCF(943, 1682, 32, 3, 0.0, "NeuMF-end")
        print(f" NeuMF-end model created: {sum(p.numel() for p in neumf_model.parameters()):,} parameters")
        
        # Test forward pass
        user_tensor = torch.LongTensor([0, 1, 2])
        item_tensor = torch.LongTensor([0, 1, 2])
        
        with torch.no_grad():
            gmf_output = gmf_model(user_tensor, item_tensor)
            mlp_output = mlp_model(user_tensor, item_tensor)
            neumf_output = neumf_model(user_tensor, item_tensor)
        
        print(f" Forward pass successful:")
        print(f"   - GMF output shape: {gmf_output.shape}")
        print(f"   - MLP output shape: {mlp_output.shape}")
        print(f"   - NeuMF output shape: {neumf_output.shape}")
        
        return True
    except Exception as e:
        print(f" Model creation failed: {e}")
        return False

def test_distillation():
    """Test if distillation modules work."""
    print("\n Testing distillation modules...")
    
    try:
        from src.ncf.models import NCF
        from src.distillation import ResponseDistillation, FeatureDistillation, AttentionDistillation
        
        # Create teacher and student models
        teacher = NCF(943, 1682, 32, 3, 0.0, "NeuMF-end")
        student = NCF(943, 1682, 16, 2, 0.0, "NeuMF-end")
        
        # Test different distillation methods
        response_distill = ResponseDistillation(teacher, student)
        feature_distill = FeatureDistillation(teacher, student)
        attention_distill = AttentionDistillation(teacher, student)
        
        print(f" Distillation modules created successfully:")
        print(f"   - Teacher parameters: {sum(p.numel() for p in teacher.parameters()):,}")
        print(f"   - Student parameters: {sum(p.numel() for p in student.parameters()):,}")
        
        # Test forward pass
        user_tensor = torch.LongTensor([0, 1, 2])
        item_tensor = torch.LongTensor([0, 1, 2])
        label_tensor = torch.FloatTensor([1.0, 0.0, 1.0])
        
        with torch.no_grad():
            response_loss = response_distill(user_tensor, item_tensor, label_tensor)
            feature_loss = feature_distill(user_tensor, item_tensor, label_tensor)
            attention_loss = attention_distill(user_tensor, item_tensor, label_tensor)
        
        print(f" Distillation forward pass successful:")
        print(f"   - Response loss: {response_loss.item():.4f}")
        print(f"   - Feature loss: {feature_loss.item():.4f}")
        print(f"   - Attention loss: {attention_loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f" Distillation test failed: {e}")
        return False

def test_directory_structure():
    """Test if all required directories exist."""
    print("\n Testing directory structure...")
    
    required_dirs = [
        "data/raw",
        "data/processed", 
        "results/figures",
        "results/logs",
        "results/models",
        "results/reports",
        "src/ncf",
        "src/data",
        "src/distillation",
        "src/training",
        "src/utils"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
        else:
            print(f" {dir_path}")
    
    if missing_dirs:
        print(f"\n Missing directories: {missing_dirs}")
        print("Creating missing directories...")
        for dir_path in missing_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f" Created {dir_path}")
    
    return True

def check_environment():
    """Check the Python environment."""
    print(" Checking environment...")
    print(f" Python version: {sys.version}")
    print(f" PyTorch version: {torch.__version__}")
    print(f" CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f" CUDA device: {torch.cuda.get_device_name(0)}")
    print(f" NumPy version: {np.__version__}")

def main():
    """Run all tests."""
    print(" Starting NCF setup verification...\n")
    
    check_environment()
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation),
        ("Distillation", test_distillation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f" {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*50)
    print(" TEST SUMMARY:")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = " PASSED" if result else " FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n All tests passed! Your setup is ready for training.")
        print("\nNext steps:")
        print("1. python scripts/pretrain.py --model GMF --epochs 5")
        print("2. python scripts/pretrain.py --model MLP --epochs 5")
        print("3. python scripts/train_neumf.py --model NeuMF-end --pretraining")
    else:
        print(f"\n  {len(results) - passed} tests failed. Please fix the issues before proceeding.")

if __name__ == "__main__":
    main()