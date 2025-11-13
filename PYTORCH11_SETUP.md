# 🚀 PyTorch 1.11+ Setup Guide

Panduan instalasi untuk menjalankan proyek dengan **PyTorch 1.11** atau lebih baru.

## 📋 Prerequisites

- Python 3.8 atau lebih tinggi
- CUDA 11.3+ (untuk GPU NVIDIA) atau ROCm 5.0+ (untuk GPU AMD)
- RAM minimal 8GB
- Storage minimal 10GB untuk dataset

## 🔧 Langkah Instalasi

### 1. Setup Virtual Environment

```bash
# Buat virtual environment
python -m venv venv

# Aktivasi (Linux/Mac)
source venv/bin/activate

# Atau (Windows)
venv\Scripts\activate
```

### 2. Install PyTorch 1.11

Pilih versi sesuai hardware Anda:

#### **Option A: CUDA 11.3 (NVIDIA GPU)**
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

#### **Option B: CUDA 11.5 (NVIDIA GPU - RTX 30/40 series)**
```bash
pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 --extra-index-url https://download.pytorch.org/whl/cu115
```

#### **Option C: ROCm 5.0 (AMD GPU)**
```bash
pip install torch==1.11.0+rocm5.0 torchvision==0.12.0+rocm5.0 --extra-index-url https://download.pytorch.org/whl/rocm5.0
```

#### **Option D: CPU Only**
```bash
pip install torch==1.11.0+cpu torchvision==0.12.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Dependencies

```bash
pip install -r requirements_torch11.txt
```

### 4. Verify Installation

```python
import torch
import torchvision

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

## 🎯 Quick Start

### Setup Dataset
```bash
# Download ImageNet subset (ikuti instruksi di datasets/README.md)
cd datasets
python download_imaginet.py
cd ..
```

### Train Model

**Quick Test (2 epochs):**
```bash
python scripts/train.py --batch-size 8 --epochs 2 --amp
```

**Full Training (50 epochs):**
```bash
python scripts/train.py --batch-size 16 --epochs 50 --amp
```

**With GPU Augmentation (MUCH faster):**
```bash
python scripts/train.py --batch-size 16 --epochs 50 --amp --gpu-aug
```

## 🆕 Perbedaan dari PyTorch 1.7

### 1. **Mixed Precision API**
- **PyTorch 1.7:** `from torch.cuda.amp import autocast`
- **PyTorch 1.11:** `from torch.amp import autocast`
- Parameter baru: `device_type='cuda'` atau `'cpu'`

### 2. **Better Performance**
- **30% lebih cepat** pada training dengan AMP
- **Memory efficiency** lebih baik (bisa pakai batch size lebih besar)
- **Gradient stability** lebih baik (convergence lebih smooth)

### 3. **Modern GPU Support**
- ✅ NVIDIA RTX 30 series (Ampere)
- ✅ NVIDIA RTX 40 series (Ada Lovelace)
- ✅ AMD RX 6000 series (ROCm 5.0+)
- ✅ Apple Silicon M1/M2 (MPS backend)

### 4. **API Changes**
```python
# OLD (PyTorch 1.7)
from torch.cuda.amp import autocast, GradScaler
with autocast(enabled=True):
    output = model(input)

# NEW (PyTorch 1.11+)
from torch.amp import autocast, GradScaler
with autocast(device_type='cuda', enabled=True):
    output = model(input)
```

## ⚙️ Training Options

```bash
# Basic options
--batch-size 16         # Batch size per GPU
--epochs 50             # Number of epochs
--lr 1e-4              # Learning rate
--amp                  # Enable mixed precision

# Advanced options
--gpu-aug              # Use Kornia GPU augmentation (RECOMMENDED!)
--accum-steps 4        # Gradient accumulation (for large effective batch)
--freeze-cnn           # Freeze CNN backbone
--fast-mode            # Minimal augmentation (faster preprocessing)
--num-workers 4        # DataLoader workers (0 for ROCm)

# Resume training
--resume               # Resume from latest checkpoint
```

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python scripts/train.py --batch-size 4 --epochs 50 --amp

# Or use gradient accumulation
python scripts/train.py --batch-size 4 --accum-steps 4 --epochs 50 --amp
# (effective batch size = 4 * 4 = 16)
```

### Slow Data Loading
```bash
# Increase workers (CUDA only, NOT for ROCm!)
python scripts/train.py --num-workers 4 --epochs 50 --amp

# Use GPU augmentation
python scripts/train.py --gpu-aug --epochs 50 --amp
```

### AMP Issues
```bash
# Disable AMP if causing problems
python scripts/train.py --amp False --epochs 50
```

### ImportError: cannot import 'autocast'
Pastikan PyTorch version >= 1.11:
```bash
pip install --upgrade torch torchvision
```

## 📊 Expected Performance

| Hardware | Batch Size | Speed (img/s) | Memory Usage |
|----------|------------|---------------|--------------|
| RTX 3090 | 32 | ~150 | ~18GB |
| RTX 3080 | 16 | ~100 | ~8GB |
| RTX 3070 | 8 | ~60 | ~6GB |
| RX 6800 XT | 16 | ~80 | ~10GB |
| CPU | 4 | ~5 | ~4GB |

*With AMP and GPU augmentation enabled*

## 🔄 Backward Compatibility

Kode ini **backward compatible** dengan PyTorch 1.7! Script secara otomatis mendeteksi versi PyTorch dan menggunakan API yang sesuai:

```python
# Auto-detection in train.py
try:
    from torch.amp import autocast, GradScaler  # PyTorch 1.11+
    TORCH_AMP_AVAILABLE = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler  # PyTorch 1.7
    TORCH_AMP_AVAILABLE = False
```

## 📚 Resources

- [PyTorch 1.11 Release Notes](https://github.com/pytorch/pytorch/releases/tag/v1.11.0)
- [Mixed Precision Training Guide](https://pytorch.org/docs/stable/amp.html)
- [Torchvision 0.12 Documentation](https://pytorch.org/vision/0.12/)
- [Timm Documentation](https://timm.fast.ai/)

## 📞 Support

Jika ada masalah:
1. Check versi PyTorch: `python -c "import torch; print(torch.__version__)"`
2. Check CUDA: `nvidia-smi` (untuk NVIDIA) atau `rocm-smi` (untuk AMD)
3. Buka issue di repository dengan error log lengkap

---

**Happy Training! 🚀**
