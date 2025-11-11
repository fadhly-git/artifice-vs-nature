#!/usr/bin/env python3
"""
Quick test untuk debug training speed
"""

import torch
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import HybridDataset
from src.models.hybrid import HybridDetector
from torch.utils.data import DataLoader, ConcatDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FAKE_DIR = PROJECT_ROOT / "data" / "processed" / "imaginet" / "subset_pt" / "fake"
REAL_DIR = PROJECT_ROOT / "data" / "processed" / "imaginet" / "subset_pt" / "real"

print("="*60)
print("🧪 TRAINING SPEED TEST")
print("="*60)

# Load small subset
print("\n📁 Loading datasets...")
fake_dataset = HybridDataset(FAKE_DIR)
real_dataset = HybridDataset(REAL_DIR)

# Use only 100 samples for quick test
from torch.utils.data import Subset
import random

fake_indices = random.sample(range(len(fake_dataset)), min(50, len(fake_dataset)))
real_indices = random.sample(range(len(real_dataset)), min(50, len(real_dataset)))

fake_subset = Subset(fake_dataset, fake_indices)
real_subset = Subset(real_dataset, real_indices)
full_dataset = ConcatDataset([real_subset, fake_subset])

print(f"✅ Test dataset size: {len(full_dataset)} samples")

# Test different batch sizes
for batch_size in [1, 2, 4, 8]:
    print(f"\n{'='*60}")
    print(f"Testing batch_size = {batch_size}")
    print(f"{'='*60}")
    
    dataloader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )
    
    model = HybridDetector(num_classes=2, pretrained=False).to(DEVICE)
    model.eval()
    
    print(f"Batches: {len(dataloader)}")
    
    # Test forward pass speed
    start = time.time()
    
    with torch.no_grad():
        for i, (img_masked, dct_feat, labels) in enumerate(dataloader):
            img_masked = img_masked.to(DEVICE)
            dct_feat = dct_feat.to(DEVICE)
            
            outputs = model(img_masked, dct_feat)
            
            if i == 0:
                print(f"First batch shapes:")
                print(f"  img_masked: {img_masked.shape}")
                print(f"  dct_feat: {dct_feat.shape}")
                print(f"  outputs: {outputs.shape}")
    
    end = time.time()
    
    total_time = end - start
    time_per_batch = total_time / len(dataloader)
    samples_per_sec = len(full_dataset) / total_time
    
    print(f"\n📊 Results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Time per batch: {time_per_batch*1000:.2f}ms")
    print(f"  Samples/sec: {samples_per_sec:.2f}")
    print(f"  Est. time for 33750 samples: {33750/samples_per_sec/60:.1f} minutes")
    
    del model
    torch.cuda.empty_cache()

print("\n" + "="*60)
print("✅ SPEED TEST COMPLETE")
print("="*60)
print("\n💡 Recommendations:")
print("1. Use batch_size >= 4 for faster training")
print("2. Enable --amp for mixed precision")
print("3. Use --subset 0.1 for quick experiments (10% of data)")
print("4. Increase --num-workers to 2 or 4 for faster data loading")
