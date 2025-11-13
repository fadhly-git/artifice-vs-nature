#!/bin/bash
# Test script to verify PyTorch compatibility

echo "=================================="
echo "PyTorch Compatibility Test"
echo "=================================="
echo ""

# Check Python version
echo "1. Python Version:"
python --version
echo ""

# Check PyTorch version
echo "2. PyTorch Info:"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA Version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"
echo ""

# Check AMP API availability
echo "3. AMP API Detection:"
python -c "
try:
    from torch.amp import autocast, GradScaler
    print('  ✅ torch.amp (PyTorch 1.11+) - AVAILABLE')
    amp_version = '1.11+'
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    print('  ✅ torch.cuda.amp (PyTorch 1.7) - AVAILABLE')
    amp_version = '1.7'
print(f'  Using AMP API version: {amp_version}')
"
echo ""

# Check timm
echo "4. Timm (PyTorch Image Models):"
python -c "
import timm
print(f'  Timm version: {timm.__version__}')
print(f'  EfficientNet-B0 available: {\"efficientnet_b0\" in timm.list_models()}')"
echo ""

# Check other dependencies
echo "5. Other Dependencies:"
python -c "
try:
    import numpy as np
    print(f'  NumPy: {np.__version__}')
except ImportError:
    print('  NumPy: NOT INSTALLED')

try:
    import PIL
    print(f'  Pillow: {PIL.__version__}')
except ImportError:
    print('  Pillow: NOT INSTALLED')

try:
    import yaml
    print(f'  PyYAML: Available')
except ImportError:
    print('  PyYAML: NOT INSTALLED')

try:
    import scipy
    print(f'  SciPy: {scipy.__version__}')
except ImportError:
    print('  SciPy: NOT INSTALLED')

try:
    import kornia
    print(f'  Kornia: {kornia.__version__} (GPU augmentation)')
except ImportError:
    print('  Kornia: NOT INSTALLED (optional)')
"
echo ""

# Test model import
echo "6. Model Import Test:"
python -c "
import sys
sys.path.insert(0, '.')
try:
    from src.models.hybrid import HybridDetector
    print('  ✅ HybridDetector imported successfully')
except Exception as e:
    print(f'  ❌ Error importing HybridDetector: {e}')
"
echo ""

echo "=================================="
echo "Test Complete!"
echo "=================================="
echo ""
echo "If all checks passed, you can train with:"
echo "  python scripts/train.py --batch-size 8 --epochs 2 --amp"
echo ""
