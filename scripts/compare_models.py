#!/usr/bin/env python3
"""
Compare different model architectures for parameter count and memory usage.

Usage:
    python scripts/compare_models.py
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models.hybrid import HybridDetector

def test_model(model_name, freeze_cnn=False):
    """Test model initialization and get stats."""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}")
    
    try:
        model = HybridDetector(
            num_classes=2,
            pretrained=False,  # Faster for testing
            freeze_cnn=freeze_cnn,
            model_name=model_name
        )
        
        # Get model info
        info = model.get_model_info()
        
        print(f"✅ Successfully loaded!")
        print(f"\n📊 Model Statistics:")
        print(f"   Backbone: {info['backbone']}")
        print(f"   CNN output dim: {info['cnn_output_dim']}")
        print(f"   Total params: {info['total_params']:,}")
        print(f"   Trainable params: {info['trainable_params']:,}")
        if info['frozen_params'] > 0:
            print(f"   Frozen params: {info['frozen_params']:,}")
        
        # Test forward pass
        dummy_img = torch.randn(2, 3, 224, 224)
        dummy_dct = torch.randn(2, 1024)
        
        with torch.no_grad():
            output = model(dummy_img, dummy_dct)
        
        print(f"\n✅ Forward pass successful!")
        print(f"   Input: img(2,3,224,224) + dct(2,1024)")
        print(f"   Output: {tuple(output.shape)}")
        
        # Calculate memory estimate (rough)
        param_mem = info['total_params'] * 4 / (1024**2)  # 4 bytes per float32
        print(f"\n💾 Memory Estimate:")
        print(f"   Model size: ~{param_mem:.2f} MB")
        
        return True, info
        
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return False, None

def main():
    print("\n" + "="*70)
    print("🔍 MODEL ARCHITECTURE COMPARISON")
    print("="*70)
    
    models = [
        'mobilenetv3_small_100',  # Lightest
        'mobilenetv3_large_100',  # Balanced
        'efficientnet_lite0',      # Medium
        'efficientnet_b0'          # Original (heaviest)
    ]
    
    results = []
    
    for model_name in models:
        success, info = test_model(model_name, freeze_cnn=False)
        if success:
            results.append((model_name, info))
    
    # Summary comparison
    if results:
        print("\n" + "="*70)
        print("📊 COMPARISON SUMMARY")
        print("="*70)
        print(f"\n{'Model':<30} {'Params':>15} {'CNN Dim':>10} {'Memory':>12}")
        print("-" * 70)
        
        for model_name, info in results:
            param_mem = info['total_params'] * 4 / (1024**2)
            print(f"{model_name:<30} {info['total_params']:>15,} "
                  f"{info['cnn_output_dim']:>10} {param_mem:>11.2f} MB")
        
        print("\n" + "="*70)
        print("💡 RECOMMENDATIONS")
        print("="*70)
        
        print("\n🚀 For fastest training & lowest memory:")
        print("   python scripts/train.py --model mobilenetv3_small_100 --batch-size 16")
        
        print("\n⚖️  For balanced performance:")
        print("   python scripts/train.py --model mobilenetv3_large_100 --batch-size 12")
        
        print("\n🎯 For best accuracy (if memory allows):")
        print("   python scripts/train.py --model efficientnet_b0 --batch-size 8")
        
        print("\n💡 TIP: Lighter models = less risk of NaN, faster training!")
        print("="*70 + "\n")

if __name__ == "__main__":
    main()
