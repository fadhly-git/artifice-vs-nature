#!/usr/bin/env python3
"""
Analyze training log to detect NaN and show what went wrong.

Usage:
    python scripts/analyze_nan.py results/logs/train_20251117_153249.txt
"""

import sys
import re
from pathlib import Path

def parse_log(log_path):
    """Parse training log and extract key metrics."""
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    epochs = []
    for line in lines:
        # Match epoch summary lines
        match = re.search(r'📊 Epoch (\d+)/\d+ Summary:', line)
        if match:
            epoch_num = int(match.group(1))
            epochs.append({'epoch': epoch_num, 'line': line})
        
        # Match train/val metrics
        if 'Train Loss:' in line:
            match = re.search(r'Train Loss: ([\d.]+|nan) \| Train Acc: ([\d.]+)%', line)
            if match and epochs:
                epochs[-1]['train_loss'] = match.group(1)
                epochs[-1]['train_acc'] = float(match.group(2))
        
        if 'Val Loss:' in line:
            match = re.search(r'Val Loss:\s+([\d.]+|nan) \| Val Acc:\s+([\d.]+)%', line)
            if match and epochs:
                epochs[-1]['val_loss'] = match.group(1)
                epochs[-1]['val_acc'] = float(match.group(2))
        
        # Match LR
        if 'LR:' in line and 'CNN LR' not in line:
            match = re.search(r'LR: ([\d.e-]+)', line)
            if match and epochs:
                epochs[-1]['lr'] = match.group(1)
        
        # Match unfreezing events
        if 'Unfreezing' in line:
            if epochs:
                epochs[-1]['event'] = line.strip()
    
    return epochs

def analyze_nan(epochs):
    """Detect when NaN first appeared and what caused it."""
    print("\n" + "="*80)
    print("🔍 NaN LOSS ANALYSIS")
    print("="*80)
    
    # Find first NaN
    nan_epoch = None
    for e in epochs:
        if e.get('train_loss') == 'nan' or e.get('val_loss') == 'nan':
            nan_epoch = e['epoch']
            break
    
    if not nan_epoch:
        print("\n✅ No NaN detected in training log!")
        return
    
    print(f"\n❌ First NaN detected at Epoch {nan_epoch}")
    
    # Show last 5 good epochs
    print("\n📊 Last 5 Good Epochs Before NaN:")
    print("-" * 80)
    for e in epochs[max(0, nan_epoch-6):nan_epoch-1]:
        event = f" [{e.get('event', '').split('🔓')[1].split('at')[0].strip()}]" if 'event' in e else ""
        print(f"  Epoch {e['epoch']:2d}: Train {e.get('train_loss', 'N/A'):>6s} | "
              f"Val {e.get('val_loss', 'N/A'):>6s} | "
              f"Val Acc {e.get('val_acc', 0):5.2f}% | "
              f"LR {e.get('lr', 'N/A'):>9s}{event}")
    
    # Show NaN epoch
    print("\n🔥 NaN Epoch:")
    print("-" * 80)
    e = epochs[nan_epoch - 1]
    event = f"\n     ⚠️  {e.get('event', '')}" if 'event' in e else ""
    print(f"  Epoch {e['epoch']:2d}: Train {e.get('train_loss', 'N/A'):>6s} | "
          f"Val {e.get('val_loss', 'N/A'):>6s} | "
          f"Val Acc {e.get('val_acc', 0):5.2f}% | "
          f"LR {e.get('lr', 'N/A'):>9s}{event}")
    
    # Diagnosis
    print("\n" + "="*80)
    print("💡 DIAGNOSIS")
    print("="*80)
    
    prev_epoch = epochs[nan_epoch - 2] if nan_epoch > 1 else None
    curr_epoch = epochs[nan_epoch - 1]
    
    if 'event' in curr_epoch and 'Unfreezing ALL' in curr_epoch.get('event', ''):
        print("\n🎯 ROOT CAUSE: CNN Full Unfreeze at Epoch", nan_epoch)
        print("   Problem: All CNN layers unfroze with learning rate too high")
        print("   → Early CNN layers (blocks.0-5) got large gradients")
        print("   → Weight explosion → NaN loss")
        
        print("\n📉 Accuracy Drop:")
        if prev_epoch:
            prev_acc = prev_epoch.get('val_acc', 0)
            curr_acc = curr_epoch.get('val_acc', 0)
            drop = prev_acc - curr_acc
            print(f"   Before: {prev_acc:.2f}%")
            print(f"   After:  {curr_acc:.2f}%")
            print(f"   Drop:   {drop:.2f}% ⬇️")
        
        print("\n🔧 SOLUTIONS:")
        print("   1. Add gradient clipping: --grad-clip 1.0")
        print("   2. Use smaller CNN LR: multiply by 0.1 instead of 2")
        print("   3. Unfreeze earlier (epoch 15-25) when LR is higher")
        print("   4. Or disable full unfreeze: --unfreeze-all-epoch 999")
        
        print("\n💊 RECOVERY:")
        print("   python scripts/recover_from_nan.py")
        print("   python scripts/train.py --resume --lr 1e-5 --grad-clip 0.5")
    else:
        print("\n⚠️  NaN appeared but no obvious unfreezing event detected")
        print("   Check for other causes:")
        print("   - Learning rate too high")
        print("   - Batch normalization issues")
        print("   - Data corruption")
        print("   - Mixed precision overflow")
    
    print("\n" + "="*80)

def show_best_performance(epochs):
    """Show best validation accuracy before NaN."""
    print("\n" + "="*80)
    print("🏆 BEST PERFORMANCE (Before NaN)")
    print("="*80)
    
    best = max([e for e in epochs if e.get('val_loss') != 'nan'], 
               key=lambda x: x.get('val_acc', 0))
    
    print(f"\n  Epoch {best['epoch']:2d}:")
    print(f"    Train Loss: {best.get('train_loss', 'N/A')}")
    print(f"    Val Loss:   {best.get('val_loss', 'N/A')}")
    print(f"    Val Acc:    {best.get('val_acc', 0):.2f}% ⭐")
    print(f"    LR:         {best.get('lr', 'N/A')}")
    
    print("\n  💡 This is the checkpoint you should resume from!")
    print("     models/checkpoints/hybrid_imagenet_best.pth")
    print("\n" + "="*80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_nan.py <log_file>")
        print("\nExample:")
        print("  python scripts/analyze_nan.py results/logs/train_20251117_153249.txt")
        sys.exit(1)
    
    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"❌ Log file not found: {log_path}")
        sys.exit(1)
    
    epochs = parse_log(log_path)
    
    if not epochs:
        print("❌ No epoch data found in log file")
        sys.exit(1)
    
    analyze_nan(epochs)
    show_best_performance(epochs)
    
    print()
