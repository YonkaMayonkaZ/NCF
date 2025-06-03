import os
import sys
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ncf.models import NCF
from src.ncf.nmf_model import NMFRecommender, NMFEvaluator, run_nmf_experiment
from src.data.datasets import NCFData, load_all
from src.utils.config import config
from src.training.metrics import metrics

def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_top_k_performance(model, test_loader, max_k=10):
    """Evaluate model performance for different values of K."""
    hr_at_k = {}
    ndcg_at_k = {}
    
    for k in range(1, max_k + 1):
        HR, NDCG = metrics(model, test_loader, k)
        hr_at_k[k] = np.mean(HR)
        ndcg_at_k[k] = np.mean(NDCG)
    
    return hr_at_k, ndcg_at_k

def evaluate_negative_sampling(model, train_data, test_data, user_num, item_num, train_mat, device, max_negatives=10):
    """Evaluate model performance with different numbers of negative samples."""
    hr_results = {}
    ndcg_results = {}
    
    for num_neg in range(1, max_negatives + 1):
        print(f"Evaluating with {num_neg} negative samples...")
        
        # Create test dataset with current number of negatives
        test_dataset = NCFData(test_data, item_num, train_mat, 0, False)
        test_loader = data.DataLoader(test_dataset, batch_size=num_neg + 1, shuffle=False, num_workers=0)
        
        # Evaluate
        HR, NDCG = metrics(model, test_loader, 10)
        hr_results[num_neg] = np.mean(HR)
        ndcg_results[num_neg] = np.mean(NDCG)
    
    return hr_results, ndcg_results

def load_best_model(model_path, model_type, user_num, item_num, factor_num, num_layers, dropout, device):
    """Load the best trained model."""
    model = NCF(user_num, item_num, factor_num, num_layers, dropout, model_type)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f" Loaded model from {model_path}")
        return model
    else:
        print(f" Model not found at {model_path}")
        return None

def create_comparison_plots():
    """Create plots for comparing different aspects of the models."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_data, test_data, user_num, item_num, train_mat = load_all()
    test_dataset = NCFData(test_data, item_num, train_mat, 0, False)
    test_loader = data.DataLoader(test_dataset, batch_size=config.test_num_ng + 1, shuffle=False, num_workers=0)
    
    # Load best NeuMF model
    neumf_path = config.model_dir / "NeuMF-end_pretrain_best.pth"
    neumf_model = load_best_model(
        neumf_path, "NeuMF-end", user_num, item_num, 
        config.factor_num, config.num_layers, config.dropout, device
    )
    
    if neumf_model is None:
        print(" Cannot find trained NeuMF model. Please train the model first.")
        return
    
    # 1. Top-K Performance Evaluation
    print("\n Evaluating Top-K performance...")
    hr_at_k, ndcg_at_k = evaluate_top_k_performance(neumf_model, test_loader, max_k=10)
    
    # Plot Top-K results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    k_values = list(hr_at_k.keys())
    hr_values = list(hr_at_k.values())
    ndcg_values = list(ndcg_at_k.values())
    
    ax1.plot(k_values, hr_values, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('K')
    ax1.set_ylabel('HR@K')
    ax1.set_title('Hit Ratio @ K')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)
    
    ax2.plot(k_values, ndcg_values, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('K')
    ax2.set_ylabel('NDCG@K')
    ax2.set_title('NDCG @ K')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_values)
    
    plt.tight_layout()
    plt.savefig(config.figure_dir / 'neumf_top_k_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved Top-K performance plot")
    
    # 2. Negative Sampling Effect
    print("\n Evaluating negative sampling effect...")
    hr_neg, ndcg_neg = evaluate_negative_sampling(
        neumf_model, train_data, test_data, user_num, item_num, train_mat, device, max_negatives=10
    )
    
    # Plot negative sampling results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    neg_values = list(hr_neg.keys())
    hr_neg_values = list(hr_neg.values())
    ndcg_neg_values = list(ndcg_neg.values())
    
    ax1.plot(neg_values, hr_neg_values, 'go-', linewidth=2, markersize=6)
    ax1.set_xlabel('Number of Negatives')
    ax1.set_ylabel('HR@10')
    ax1.set_title('HR@10 vs Number of Negative Samples')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(neg_values)
    
    ax2.plot(neg_values, ndcg_neg_values, 'mo-', linewidth=2, markersize=6)
    ax2.set_xlabel('Number of Negatives')
    ax2.set_ylabel('NDCG@10')
    ax2.set_title('NDCG@10 vs Number of Negative Samples')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(neg_values)
    
    plt.tight_layout()
    plt.savefig(config.figure_dir / 'neumf_negative_sampling.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved negative sampling effect plot")
    
    # 3. NMF Experiments
    print("\n Running NMF experiments...")
    n_components_list = list(range(1, 31, 5))  # 1, 6, 11, 16, 21, 26
    nmf_results = run_nmf_experiment(train_mat, test_data, n_components_list, num_runs=10)
    
    # Plot NMF results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    components = list(nmf_results.keys())
    nmf_ndcg_means = [nmf_results[c]['ndcg_mean'] for c in components]
    nmf_ndcg_stds = [nmf_results[c]['ndcg_std'] for c in components]
    nmf_params = [nmf_results[c]['parameters'] for c in components]
    
    # NDCG vs Components
    ax1.errorbar(components, nmf_ndcg_means, yerr=nmf_ndcg_stds, 
                fmt='co-', linewidth=2, markersize=6, capsize=5)
    ax1.set_xlabel('Number of Latent Factors')
    ax1.set_ylabel('NDCG@10')
    ax1.set_title('NMF: NDCG@10 vs Latent Factors')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(components)
    
    # Parameters vs Components
    ax2.plot(components, nmf_params, 'co-', linewidth=2, markersize=6)
    ax2.set_xlabel('Number of Latent Factors')
    ax2.set_ylabel('Number of Parameters')
    ax2.set_title('NMF: Parameters vs Latent Factors')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(components)
    
    plt.tight_layout()
    plt.savefig(config.figure_dir / 'nmf_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved NMF analysis plot")
    
    # 4. Model Comparison
    print("\n Comparing NeuMF vs NMF...")
    
    # Get best NMF result
    best_nmf_comp = max(nmf_results.keys(), key=lambda x: nmf_results[x]['ndcg_mean'])
    best_nmf_result = nmf_results[best_nmf_comp]
    
    # Get NeuMF results
    neumf_hr = hr_at_k[10]  # HR@10
    neumf_ndcg = ndcg_at_k[10]  # NDCG@10
    neumf_params = count_parameters(neumf_model)
    
    print(f"\n Model Comparison:")
    print(f"NeuMF:")
    print(f"  HR@10: {neumf_hr:.4f}")
    print(f"  NDCG@10: {neumf_ndcg:.4f}")
    print(f"  Parameters: {neumf_params:,}")
    
    print(f"\nBest NMF ({best_nmf_comp} factors):")
    print(f"  HR@10: {best_nmf_result['hr_mean']:.4f} ± {best_nmf_result['hr_std']:.4f}")
    print(f"  NDCG@10: {best_nmf_result['ndcg_mean']:.4f} ± {best_nmf_result['ndcg_std']:.4f}")
    print(f"  Parameters: {best_nmf_result['parameters']:,}")
    
    # Create comparison plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    models = ['NeuMF', 'NMF']
    hr_values = [neumf_hr, best_nmf_result['hr_mean']]
    ndcg_values = [neumf_ndcg, best_nmf_result['ndcg_mean']]
    param_values = [neumf_params, best_nmf_result['parameters']]
    
    # HR comparison
    bars1 = ax1.bar(models, hr_values, color=['skyblue', 'lightcoral'])
    ax1.set_ylabel('HR@10')
    ax1.set_title('Hit Ratio Comparison')
    ax1.set_ylim(0, max(hr_values) * 1.1)
    for i, v in enumerate(hr_values):
        ax1.text(i, v + max(hr_values) * 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    # NDCG comparison
    bars2 = ax2.bar(models, ndcg_values, color=['skyblue', 'lightcoral'])
    ax2.set_ylabel('NDCG@10')
    ax2.set_title('NDCG Comparison')
    ax2.set_ylim(0, max(ndcg_values) * 1.1)
    for i, v in enumerate(ndcg_values):
        ax2.text(i, v + max(ndcg_values) * 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    # Parameters comparison (log scale)
    bars3 = ax3.bar(models, param_values, color=['skyblue', 'lightcoral'])
    ax3.set_ylabel('Number of Parameters')
    ax3.set_title('Model Complexity Comparison')
    ax3.set_yscale('log')
    for i, v in enumerate(param_values):
        ax3.text(i, v * 1.1, f'{v:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(config.figure_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved model comparison plot")
    
    return {
        'neumf': {'hr': neumf_hr, 'ndcg': neumf_ndcg, 'parameters': neumf_params},
        'nmf': best_nmf_result,
        'top_k_results': {'hr': hr_at_k, 'ndcg': ndcg_at_k},
        'negative_sampling': {'hr': hr_neg, 'ndcg': ndcg_neg},
        'nmf_detailed': nmf_results
    }

def evaluate_layer_effects():
    """Evaluate the effect of different numbers of MLP layers."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, test_data, user_num, item_num, train_mat = load_all()
    test_dataset = NCFData(test_data, item_num, train_mat, 0, False)
    test_loader = data.DataLoader(test_dataset, batch_size=config.test_num_ng + 1, shuffle=False, num_workers=0)
    
    layer_results = {}
    
    for num_layers in [1, 2, 3]:
        print(f"\n Evaluating {num_layers} layers...")
        
        # Check for pretrained models
        pretrain_path = config.model_dir / f"NeuMF-end_pretrain_layers{num_layers}_best.pth"
        scratch_path = config.model_dir / f"NeuMF-end_scratch_layers{num_layers}_best.pth"
        
        layer_results[num_layers] = {}
        
        # Evaluate with pretraining
        if pretrain_path.exists():
            model = load_best_model(
                pretrain_path, "NeuMF-end", user_num, item_num,
                config.factor_num, num_layers, config.dropout, device
            )
            if model:
                HR, NDCG = metrics(model, test_loader, 10)
                layer_results[num_layers]['pretrain'] = {
                    'hr': np.mean(HR),
                    'ndcg': np.mean(NDCG),
                    'parameters': count_parameters(model)
                }
        
        # Evaluate without pretraining
        if scratch_path.exists():
            model = load_best_model(
                scratch_path, "NeuMF-end", user_num, item_num,
                config.factor_num, num_layers, config.dropout, device
            )
            if model:
                HR, NDCG = metrics(model, test_loader, 10)
                layer_results[num_layers]['scratch'] = {
                    'hr': np.mean(HR),
                    'ndcg': np.mean(NDCG),
                    'parameters': count_parameters(model)
                }
    
    # Plot layer effects
    if layer_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        layers = list(layer_results.keys())
        pretrain_hrs = [layer_results[l].get('pretrain', {}).get('hr', 0) for l in layers]
        scratch_hrs = [layer_results[l].get('scratch', {}).get('hr', 0) for l in layers]
        pretrain_params = [layer_results[l].get('pretrain', {}).get('parameters', 0) for l in layers]
        scratch_params = [layer_results[l].get('scratch', {}).get('parameters', 0) for l in layers]
        
        # HR comparison
        x = np.arange(len(layers))
        width = 0.35
        
        ax1.bar(x - width/2, pretrain_hrs, width, label='With Pretraining', color='skyblue')
        ax1.bar(x + width/2, scratch_hrs, width, label='Without Pretraining', color='lightcoral')
        ax1.set_xlabel('Number of MLP Layers')
        ax1.set_ylabel('HR@10')
        ax1.set_title('HR@10 vs Number of MLP Layers')
        ax1.set_xticks(x)
        ax1.set_xticklabels(layers)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Parameters comparison
        ax2.bar(x - width/2, pretrain_params, width, label='With Pretraining', color='skyblue')
        ax2.bar(x + width/2, scratch_params, width, label='Without Pretraining', color='lightcoral')
        ax2.set_xlabel('Number of MLP Layers')
        ax2.set_ylabel('Number of Parameters')
        ax2.set_title('Parameters vs Number of MLP Layers')
        ax2.set_xticks(x)
        ax2.set_xticklabels(layers)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(config.figure_dir / 'layer_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(" Saved layer effects plot")
    
    return layer_results

def main():
    print(" Starting comprehensive model evaluation...")
    
    # Create all evaluation plots
    results = create_comparison_plots()
    
    # Evaluate layer effects if models exist
    layer_results = evaluate_layer_effects()
    
    print("\n Evaluation completed! Check the figures directory for plots.")
    print(f" Figures saved in: {config.figure_dir}")

if __name__ == "__main__":
    main()