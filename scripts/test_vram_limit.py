"""
Test VRAM usage untuk identify crash point
"""

import os, sys
import torch
import torch.nn as nn
from pathlib import Path
import subprocess

# ROCm settings untuk RX 580
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '8.0.3'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:128'

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import create_dataloaders
from src.models.hybrid import HybridDetector

def get_vram_mb():
    try:
        result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'], 
                              capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'Total Used Memory' in line:
                return int(line.split(':')[1].strip()) / (1024**2)
    except:
        pass
    return 0

DEVICE = torch.device("cuda")
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "imaginet" / "subset"

print("🔍 Testing VRAM limits...")
print(f"GPU: {torch.cuda.get_device_name(0)}")

for bs in [1, 2, 4]:
    print(f"\n{'='*50}")
    print(f"BATCH_SIZE = {bs}")
    try:
        torch.cuda.empty_cache()
        train_loader, _ = create_dataloaders(DATA_ROOT, None, bs, 0, 42)
        model = HybridDetector(num_classes=2).to(DEVICE)
        
        batch = next(iter(train_loader))
        img, dct, lbl = [x.to(DEVICE) for x in batch]
        
        out = model(img, dct)
        loss = nn.CrossEntropyLoss()(out, lbl)
        
        vram_before = get_vram_mb()
        loss.backward()
        vram_after = get_vram_mb()
        
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"✅ PASSED - Peak: {peak:.0f}MB, VRAM: {vram_after:.0f}MB")
        
        del model, train_loader
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ CRASH - {str(e)[:50]}")
        break
