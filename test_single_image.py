#!/usr/bin/env python3
import sys
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import numpy as np

print("Testing single image load...")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "imaginet" / "subset"

# Find first image
img_path = None
for p in (DATA_ROOT / "real").rglob("*.jpg"):
    img_path = p
    break

if not img_path:
    for p in (DATA_ROOT / "real").rglob("*.png"):
        img_path = p
        break

if not img_path:
    print("❌ No image found!")
    sys.exit(1)

print(f"Testing image: {img_path.name}")

# Load image
print("1. Loading image...")
img = Image.open(img_path).convert('RGB')
print(f"   ✅ Loaded: {img.size}")

# Transform
print("2. Applying transforms...")
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_tensor = transform(img)
print(f"   ✅ Transformed: {img_tensor.shape}")

# Move to GPU
print("3. Moving to GPU...")
img_tensor = img_tensor.unsqueeze(0).cuda()
print(f"   ✅ On GPU: {img_tensor.device}")

print("\n✅ Single image test PASSED!")
