#!/usr/bin/env python3
"""
Recovery script untuk melanjutkan training dari checkpoint terbaik
ketika NaN loss terdeteksi.

Usage:
    python scripts/recover_from_nan.py --resume --lr 1e-5 --grad-clip 0.5
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import argparse

parser = argparse.ArgumentParser(description='Recover training from best checkpoint')
parser.add_argument('--lr', type=float, default=1e-5, help='Lower learning rate for recovery')
parser.add_argument('--grad-clip', type=float, default=0.5, help='Stricter gradient clipping')
parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
parser.add_argument('--epochs', type=int, default=50, help='Total epochs')
args = parser.parse_args()

print("\n" + "="*60)
print("🔧 RECOVERY MODE: Loading Best Checkpoint")
print("="*60)

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
BEST_CHECKPOINT = CHECKPOINT_DIR / "hybrid_imaginet_best.pth"

if not BEST_CHECKPOINT.exists():
    print(f"\n❌ Best checkpoint not found: {BEST_CHECKPOINT}")
    print("   Nothing to recover from!")
    sys.exit(1)

# Load best checkpoint to check epoch
checkpoint = torch.load(BEST_CHECKPOINT, map_location='cpu')
best_epoch = checkpoint.get('epoch', 0)
best_acc = checkpoint.get('val_acc', 0.0)

print(f"\n✅ Found best checkpoint:")
print(f"   Epoch: {best_epoch}")
print(f"   Val Acc: {best_acc:.2f}%")
print(f"\n🔄 Will resume training with:")
print(f"   Learning Rate: {args.lr:.2e} (conservative)")
print(f"   Gradient Clip: {args.grad_clip} (strict)")
print(f"   Batch Size: {args.batch_size}")

# Copy best to latest for resuming
LATEST_CHECKPOINT = CHECKPOINT_DIR / "hybrid_imagenet_latest.pth"
import shutil
shutil.copy(BEST_CHECKPOINT, LATEST_CHECKPOINT)
print(f"\n📋 Copied best → latest checkpoint")

print(f"\n🚀 Now run training with --resume flag:")
print(f"   python scripts/train.py --resume --lr {args.lr} --grad-clip {args.grad_clip} --batch-size {args.batch_size}")
print(f"\n   Or disable full CNN unfreezing to prevent future NaN:")
print(f"   python scripts/train.py --resume --lr {args.lr} --grad-clip {args.grad_clip} --unfreeze-all-epoch 999")

print("\n" + "="*60)
