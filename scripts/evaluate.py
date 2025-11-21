#!/usr/bin/env python3
"""
Evaluate trained model on validation set
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import create_dataloaders
from src.models.hybrid import HybridDetector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def detect_model_architecture(checkpoint):
    """Detect model architecture from checkpoint state_dict"""
    state_dict = checkpoint['model_state_dict']
    
    # Get fusion dimension from classifier first layer
    classifier_keys = [k for k in state_dict.keys() if 'classifier.0.weight' in k]
    if classifier_keys:
        fusion_dim = state_dict[classifier_keys[0]].shape[1]
        cnn_out_dim = fusion_dim - 128  # fusion = cnn + mlp(128)
        
        # Map CNN output dim to architecture
        arch_map = {
            576: 'mobilenet_v3_small',
            960: 'mobilenet_v3_large',
            1280: 'mobilenet_v2',  # or efficientnet, we'll try mobilenet first
            512: 'resnet18',
            2048: 'resnet50'
        }
        
        if cnn_out_dim in arch_map:
            return arch_map[cnn_out_dim]
    
    # Default fallback
    return 'mobilenet_v3_small'

def load_model_and_data(checkpoint_path, data_root, dct_dir, batch_size=128, model_name=None):
    """Load model and validation data"""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Auto-detect architecture if not specified
    if model_name is None:
        model_name = detect_model_architecture(checkpoint)
        print(f"🔍 Auto-detected architecture: {model_name}")
    
    # Load model with correct architecture
    model = HybridDetector(
        num_classes=2, 
        pretrained=False,
        model_name=model_name
    ).to(DEVICE)
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded successfully from checkpoint")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print(f"   Checkpoint might be from different architecture")
        print(f"   Try specifying correct --model argument")
        raise
    
    model.eval()
    
    # Load data
    _, val_loader = create_dataloaders(
        root_dir=data_root,
        dct_dir=dct_dir if dct_dir.exists() else None,
        batch_size=batch_size,
        num_workers=0,
        train_ratio=0.8,
        seed=42
    )
    
    return model, val_loader, checkpoint

def run_inference(model, val_loader):
    """Run inference and collect predictions"""
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for img_masked, dct_feat, labels in tqdm(val_loader, desc="Evaluating"):
            img_masked = img_masked.to(DEVICE)
            dct_feat = dct_feat.to(DEVICE)
            
            outputs = model(img_masked, dct_feat)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def compute_metrics(all_labels, all_preds, all_probs):
    """Compute all evaluation metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs)
    }
    
    return metrics

def plot_confusion_matrix(all_labels, all_preds, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(all_labels, all_preds)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'],
                ax=ax)
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, cm

def plot_roc_curve(all_labels, all_probs, save_path=None):
    """Plot ROC curve"""
    auc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def evaluate_model(checkpoint_path=None, data_root=None, dct_dir=None, model_name=None):
    """Main evaluation function"""
    if checkpoint_path is None:
        checkpoint_path = PROJECT_ROOT / "models" / "checkpoints" / "hybrid_imaginet_best.pth"
    if data_root is None:
        data_root = PROJECT_ROOT / "data" / "raw" / "imaginet" / "subset"
    if dct_dir is None:
        dct_dir = PROJECT_ROOT / "data" / "processed" / "imaginet" / "dct_features"
    
    print("="*60)
    print("🎯 MODEL EVALUATION")
    print("="*60)
    
    # Load (model_name=None will auto-detect)
    model, val_loader, checkpoint = load_model_and_data(
        checkpoint_path, data_root, dct_dir, model_name=model_name
    )
    
    print(f"\n📂 Checkpoint: epoch {checkpoint['epoch']+1}, val_acc: {checkpoint['val_acc']:.2f}%")
    
    # Inference
    all_preds, all_labels, all_probs = run_inference(model, val_loader)
    
    # Metrics
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    
    print("\n📊 METRICS")
    print("="*60)
    for k, v in metrics.items():
        print(f"{k.capitalize():12s}: {v*100:.2f}%" if k != 'auc' else f"{k.upper():12s}: {v:.4f}")
    
    # Classification report
    print("\n" + classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))
    
    # Plots
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    
    plot_confusion_matrix(all_labels, all_preds, results_dir / "confusion_matrix.png")
    plot_roc_curve(all_labels, all_probs, results_dir / "roc_curve.png")
    
    print(f"\n💾 Results saved to: {results_dir}")
    
    return {
        'metrics': metrics,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

if __name__ == "__main__":
    evaluate_model()