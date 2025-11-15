#!/usr/bin/env python3
import sys
import time
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

print("="*60)
print("DATASET LOADING TEST")
print("="*60)

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "imaginet" / "subset"
DCT_DIR = PROJECT_ROOT / "data" / "processed" / "imaginet" / "dct_features"

# Step 1: Check if data exists
print("\n[1/5] Checking data directories...")
sys.stdout.flush()

if not DATA_ROOT.exists():
    print(f"❌ DATA_ROOT not found: {DATA_ROOT}")
    sys.exit(1)

real_dir = DATA_ROOT / "real"
fake_dir = DATA_ROOT / "fake"

if real_dir.exists():
    real_count = len(list(real_dir.rglob("*.jpg"))) + len(list(real_dir.rglob("*.png")))
    print(f"✅ Real images: {real_count}")
else:
    print(f"⚠️  No real directory: {real_dir}")

if fake_dir.exists():
    fake_count = len(list(fake_dir.rglob("*.jpg"))) + len(list(fake_dir.rglob("*.png")))
    print(f"✅ Fake images: {fake_count}")
else:
    print(f"⚠️  No fake directory: {fake_dir}")

dct_exists = DCT_DIR.exists()
print(f"{'✅' if dct_exists else '⚠️ '} DCT directory: {dct_exists}")

time.sleep(1)

# Step 2: Import dataset module
print("\n[2/5] Importing dataset module...")
sys.stdout.flush()

try:
    from src.data.dataset import HybridDataset
    print("✅ Dataset module imported")
except Exception as e:
    print(f"❌ Failed to import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

time.sleep(1)

# Step 3: Initialize dataset
print("\n[3/5] Initializing dataset...")
print("    (This may take a while if scanning many files)")
sys.stdout.flush()

try:
    dataset = HybridDataset(
        DATA_ROOT,
        dct_dir=DCT_DIR if dct_exists else None,
        is_training=False,  # No augmentation
        current_epoch=0,
        max_epochs=1
    )
    print(f"✅ Dataset initialized: {len(dataset)} samples")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

time.sleep(1)

# Step 4: Load single sample (without DataLoader)
print("\n[4/5] Loading single sample directly...")
sys.stdout.flush()

try:
    img, dct, label = dataset[0]
    print(f"✅ Sample loaded:")
    print(f"    Image shape: {img.shape}")
    print(f"    DCT shape: {dct.shape}")
    print(f"    Label: {label.item()}")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

time.sleep(1)

# Step 5: Test DataLoader with batch_size=1
print("\n[5/5] Testing DataLoader (batch_size=1, num_workers=0)...")
print("    (This is where training usually hangs)")
sys.stdout.flush()

try:
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # CRITICAL
        pin_memory=False,
        drop_last=False
    )
    
    print(f"✅ DataLoader created: {len(loader)} batches")
    
    # Try to iterate
    print("\n    Iterating first 3 batches...")
    for i, (img, dct, label) in enumerate(loader):
        if i >= 3:
            break
        print(f"    Batch {i+1}: img={img.shape}, dct={dct.shape}, label={label.shape}")
        sys.stdout.flush()
    
    print("✅ DataLoader iteration successful")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL DATASET TESTS PASSED!")
print("="*60)
print("\nNext: Test model + training loop")
