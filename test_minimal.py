import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

print("="*60)
print("MINIMAL TRAINING TEST")
print("="*60)

# Test 1: GPU access
print("\n1. Testing GPU...")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    # Simple GPU operation
    a = torch.randn(100, 100, device='cuda')
    b = torch.randn(100, 100, device='cuda')
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    print("✅ GPU computation working")
else:
    print("❌ GPU not available")
    sys.exit(1)

# Test 2: Load data
print("\n2. Testing dataset...")
from src.data.dataset import HybridDataset

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "imaginet" / "subset"
DCT_DIR = PROJECT_ROOT / "data" / "processed" / "imaginet" / "dct_features"

try:
    dataset = HybridDataset(
        DATA_ROOT,
        dct_dir=DCT_DIR if DCT_DIR.exists() else None,
        is_training=True
    )
    print(f"✅ Dataset loaded: {len(dataset)} samples")
except Exception as e:
    print(f"❌ Dataset error: {e}")
    sys.exit(1)

# Test 3: Single batch
print("\n3. Testing single batch...")
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

try:
    img, dct, label = next(iter(loader))
    print(f"✅ Batch loaded:")
    print(f"   Image: {img.shape}")
    print(f"   DCT: {dct.shape}")
    print(f"   Label: {label}")
except Exception as e:
    print(f"❌ DataLoader error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Model forward pass
print("\n4. Testing model...")
from src.models.hybrid import HybridDetector

try:
    model = HybridDetector(num_classes=2, pretrained=False).cuda()
    
    img = img.cuda()
    dct = dct.cuda()
    
    print("   Running forward pass...")
    output = model(img, dct)
    torch.cuda.synchronize()
    
    print(f"✅ Model working: output shape {output.shape}")
except Exception as e:
    print(f"❌ Model error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Full iteration
print("\n5. Testing 5 iterations...")
import time

model.train()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for i, (img, dct, label) in enumerate(loader):
    if i >= 5:
        break
    
    start = time.time()
    
    img = img.cuda()
    dct = dct.cuda()
    label = label.cuda()
    
    optimizer.zero_grad()
    output = model(img, dct)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
    
    torch.cuda.synchronize()
    
    elapsed = time.time() - start
    print(f"   Iter {i+1}: loss={loss.item():.4f}, time={elapsed:.3f}s")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)

