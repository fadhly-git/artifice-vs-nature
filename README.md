# 🎨 Artifice vs Nature

**AI-Generated Image Detection using Deep Learning**

Deteksi gambar AI-generated vs real menggunakan PyTorch (1.7.0 - 1.11+)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.7+](https://img.shields.io/badge/PyTorch-1.7%20%7C%201.11+-red.svg)](https://pytorch.org/)
[![ROCm 3.5+](https://img.shields.io/badge/ROCm-3.5%20%7C%205.0+-green.svg)](https://rocmdocs.amd.com/)

---

## 🆕 PyTorch Version Support

Proyek ini mendukung **PyTorch 1.7.0** (ROCm 3.5) dan **PyTorch 1.11+** (CUDA 11.3+, ROCm 5.0+):

- **PyTorch 1.7.0** (ROCm 3.5, Legacy): Lihat instalasi di bawah
- **PyTorch 1.11+** (Modern GPUs): 📚 **[PYTORCH11_SETUP.md](PYTORCH11_SETUP.md)** ⭐ **RECOMMENDED**

### Quick Start (PyTorch 1.11+)

```bash
# Install PyTorch 1.11 (CUDA 11.3)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install dependencies
pip install -r requirements_torch11.txt

# Train model (with GPU augmentation!)
python scripts/train.py --batch-size 16 --epochs 50 --amp --gpu-aug
```

**Benefits of PyTorch 1.11+:**
- ⚡ **30% faster training** with improved AMP
- 💾 **Better memory efficiency** (larger batch sizes)
- 🎯 **Modern GPU support** (RTX 30/40 series, RX 6000 series)
- 🍎 **Apple Silicon** (M1/M2 via MPS backend)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Usage](#usage)
- [Documentation](#documentation)
- [Hardware Setup](#hardware-setup)

---

## 🎯 Overview

Proyek klasifikasi gambar untuk membedakan antara gambar buatan AI (artifice) dan gambar alami (nature). Menggunakan preprocessing pipeline yang sophisticated dengan 6 tahap pemrosesan.

**Pipeline:** JPEG → Resize → Augment → Normalize → Mask → DCT

### Key Features

✨ **Modular Preprocessing** - 6-stage pipeline yang dapat dikonfigurasi  
🔬 **DCT Feature Extraction** - Analisis frekuensi untuk deteksi AI artifacts  
🎮 **ROCm Support** - Optimized untuk AMD GPU  
📊 **Jupyter Integration** - Semua eksperimen di Jupyter Notebook  
⚙️ **YAML Configuration** - Easy parameter tuning  
📈 **Visualization Tools** - Built-in 4-panel visualization

---

## 🌟 Features

- **JPEG Compression Simulation** - Meningkatkan robustness model
- **Random Masking** - Inspired by Masked Autoencoders (MAE)
- **DCT Analysis** - Ekstraksi 1024 koefisien DCT untuk deteksi artifacts
- **ImageNet Normalization** - Transfer learning ready
- **Data Augmentation** - Flip, brightness, contrast adjustments
- **AMD GPU Support** - Full ROCm 3.5 compatibility

---

---

## 📁 Project Structure

```
artifice-vs-nature.git/
├── 📁 data/
│   ├── 📁 raw/              # Data mentah (real & synthetic)
│   └── 📁 processed/        # Data preprocessed
│       └── 📁 imaginet/
│           └── 📁 subset/   # ImagiNet subset (anime, ffhq, etc.)
│
├── 📁 src/
│   ├── 📁 preprocessing/
│   │   ├── jpeg.py          # JPEG compression
│   │   ├── resize.py        # Image resizing to 224×224
│   │   ├── augment.py       # Data augmentation
│   │   ├── normalize.py     # ImageNet normalization
│   │   ├── mask.py          # Random masking
│   │   ├── dct.py           # DCT feature extraction
│   │   └── pipeline.py      # Full preprocessing pipeline
│   └── 📁 utils/
│       └── visualization.py # Visualization tools
│
├── 📁 models/
│   ├── torch-1.7.0*.whl     # PyTorch wheel for ROCm
│   ├── torchvision-0.8*.whl # torchvision wheel
│   └── 📁 checkpoints/       # Model checkpoints
│
├── 📁 notebooks/
│   └── preprocessing_demo.ipynb  # Main demo notebook ⭐
│
├── 📁 results/
│   ├── 📁 figures/          # Visualizations
│   ├── 📁 logs/             # Training logs
│   └── 📁 metrics/          # Metrics (JSON)
│
├── 📁 configs/
│   └── preprocessing.yaml   # Preprocessing configuration
│
├── 📁 docs/
│   ├── PREPROCESSING_MODULES.md  # Module documentation
│   └── DATA_AUGMENTATION.md      # Augmentation guide
│
├── 📁 lib/
│   ├── torch-1.7.0*.whl     # Local PyTorch installation
│   └── torchvision-0.8*.whl # Local torchvision installation
│
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## 🚀 Installation

### Prerequisites

- **OS:** Ubuntu 20.04
- **GPU:** AMD RX 580 (8GB VRAM)
- **ROCm:** 3.5
- **Python:** 3.8+

### Step 1: Clone Repository

```bash
git clone https://github.com/fadhly-git/artifice-vs-nature.git
cd artifice-vs-nature
```

### Step 2: Download PyTorch Wheels

- **torch-1.7.0a0 (ROCm 3.5)**: [torch-1.7.0a0-cp38-cp38-linux_x86_64.whl](https://github.com/xuhuisheng/rocm-gfx803/releases/download/rocm35/torch-1.7.0a0-cp38-cp38-linux_x86_64.whl)  
- **torchvision-0.8.0a0**: [torchvision-0.8.0a0+2f40a48-cp38-cp38-linux_x86_64.whl](https://github.com/xuhuisheng/rocm-gfx803/releases/download/rocm35/torchvision-0.8.0a0+2f40a48-cp38-cp38-linux_x86_64.whl) 

move to ```lib/```

### Step 3: Install Dependencies

**Option A: Install in Jupyter Notebook** (Recommended)

Open `notebooks/preprocessing_demo.ipynb` dan jalankan Cell 1:
```python
%pip install ../lib/torch-1.7.0a0-cp38-cp38-linux_x86_64.whl
%pip install ../lib/torchvision-0.8.0a0+2f40a48-cp38-cp38-linux_x86_64.whl
```

**Option B: Install via Terminal**

```bash
pip install lib/torch-1.7.0a0-cp38-cp38-linux_x86_64.whl
pip install lib/torchvision-0.8.0a0+2f40a48-cp38-cp38-linux_x86_64.whl
pip install -r requirements.txt
```

### Step 4: Verify Installation

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

Expected output:
```
PyTorch: 1.7.0a0+...
CUDA Available: True
GPU: Ellesmere [Radeon RX 470/480/570/570X/580/580X/590]
```

---

## ⚡ Quick Start

### 1. Run Preprocessing Demo

```bash
cd notebooks
jupyter notebook preprocessing_demo.ipynb
```

Kemudian klik **Run → Run All Cells**

### 2. Process Single Image

```python
from src.preprocessing.pipeline import preprocess_full
import yaml

# Load config
with open('configs/preprocessing.yaml') as f:
    config = yaml.safe_load(f)

# Process image
result = preprocess_full("path/to/image.jpg", config)

# Access outputs
img_masked = result['img_masked']    # torch.Tensor (3, 224, 224)
dct_feat = result['dct_feat']        # torch.Tensor (1024,)
intermediates = result['intermediates']  # For visualization
```

### 3. Visualize Pipeline

```python
from src.utils import plot_preprocessing_pipeline

plot_preprocessing_pipeline(intermediates)
```

---

## 🔄 Preprocessing Pipeline

### Pipeline Stages

```
Input Image (any size)
    ↓
[1] JPEG Compression
    Random quality: 30-90
    Probability: 0.7
    ↓
[2] Resize to 224×224
    Method: LANCZOS
    ↓
[3] Data Augmentation
    - Horizontal flip (p=0.5)
    - Brightness (±0.2)
    - Contrast (±0.2)
    ↓
[4] Normalization
    ImageNet: mean=[0.485,0.456,0.406]
              std=[0.229,0.224,0.225]
    ↓
[5] Random Masking
    Ratio: 20% area
    ↓
[6] DCT Feature Extraction
    Top-1024 coefficients
    Block size: 8×8
    ↓
Output: {img_masked, dct_feat, intermediates}
```

### Output Format

```python
{
    'img_masked': torch.FloatTensor(3, 224, 224),  # Preprocessed image
    'dct_feat': torch.FloatTensor(1024,),          # DCT features
    'intermediates': {
        'original': PIL.Image,       # Original image
        'jpeg': PIL.Image,           # After JPEG compression
        'resized': PIL.Image,        # 224×224 resized
        'augmented': torch.Tensor,   # After augmentation
        'normalized': torch.Tensor,  # Normalized
        'masked': torch.Tensor,      # With masking
        'dct_values': np.ndarray     # DCT coefficients
    }
}
```

---

## 💻 Usage

### Basic Usage

```python
import sys
sys.path.insert(0, '/path/to/artifice-vs-nature')

from src.preprocessing import preprocess_full
import yaml

# Load configuration
with open('configs/preprocessing.yaml') as f:
    config = yaml.safe_load(f)

# Process image
result = preprocess_full("image.jpg", config)
```

### Batch Processing

```python
import glob
import torch

image_paths = glob.glob("data/raw/**/*.jpg", recursive=True)

batch_masked = []
batch_dct = []

for img_path in image_paths[:32]:
    result = preprocess_full(img_path, config)
    batch_masked.append(result['img_masked'])
    batch_dct.append(result['dct_feat'])

# Stack into tensors
batch_masked = torch.stack(batch_masked)  # (32, 3, 224, 224)
batch_dct = torch.stack(batch_dct)        # (32, 1024)
```

### DataLoader Integration

```python
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, image_paths, config):
        self.image_paths = image_paths
        self.config = config
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        result = preprocess_full(self.image_paths[idx], self.config)
        return result['img_masked'], result['dct_feat']

dataset = ImageDataset(image_paths, config)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

---

## 📚 Documentation

- **[SETUP_SUCCESS.md](SETUP_SUCCESS.md)** - Installation verification & troubleshooting
- **[docs/PREPROCESSING_MODULES.md](docs/PREPROCESSING_MODULES.md)** - Detailed module documentation
- **[docs/DATA_AUGMENTATION.md](docs/DATA_AUGMENTATION.md)** - Augmentation strategies
- **[QUICK_START.md](QUICK_START.md)** - Quick start guide
- **[STRUKTUR.md](STRUKTUR.md)** - Project structure details

---

## 🖥️ Hardware Setup

### AMD RX 580 Specifications

- **GPU:** AMD Radeon RX 580
- **VRAM:** 8GB GDDR5
- **Compute Units:** 36
- **Stream Processors:** 2304
- **ROCm Version:** 3.5

### Performance Benchmarks

| Operation | Time (per image) | Batch 32 |
|-----------|------------------|----------|
| JPEG Compression | ~10ms | ~320ms |
| Resize | ~5ms | ~160ms |
| Augmentation | ~2ms | ~64ms |
| Normalization | ~1ms | ~32ms |
| Masking | ~2ms | ~64ms |
| DCT Extraction | ~30ms | ~960ms |
| **Total** | **~50ms** | **~1.6s** |

**Memory Usage:**
- Per image: ~1 MB (preprocessed)
- Batch 32: ~32 MB
- Model loading: ~500 MB (typical CNN)

---

## ⚙️ Configuration

Edit `configs/preprocessing.yaml` untuk customize pipeline:

```yaml
jpeg:
  prob: 0.7                # Probability of applying JPEG
  quality_range: [30, 90]  # JPEG quality range

mask:
  ratio: 0.2               # Masking ratio (20%)

dct:
  top_k: 1024              # Number of DCT coefficients
  block_size: 8            # DCT block size

augmentation:
  horizontal_flip_prob: 0.5
  brightness_range: [-0.2, 0.2]
  contrast_range: [-0.2, 0.2]
```

---

## 🔬 Dataset

### ImagiNet Dataset

Dataset menggunakan **ImagiNet** - koleksi gambar AI-generated dan real.

**Location:** `data/processed/imaginet/subset/`

**Structure:**
```
imaginet/subset/
├── fake/
│   └── ffhq_stylegan/    # StyleGAN-generated faces
└── real/
    └── ffhq/             # Real human faces (FFHQ dataset)
```

**Statistics:**
- Real images: FFHQ subset
- Fake images: StyleGAN-generated (FFHQ-based)
- Image size: 1024×1024 PNG
- Total samples: Varies by subset

---

## 🧪 Testing

### Test Individual Modules

```bash
python src/preprocessing/jpeg.py
python src/preprocessing/resize.py
python src/preprocessing/augment.py
python src/preprocessing/normalize.py
python src/preprocessing/mask.py
python src/preprocessing/dct.py
python src/preprocessing/pipeline.py
```

### Test in Notebook

Run all cells in `notebooks/preprocessing_demo.ipynb`

---

## 🐛 Troubleshooting

### Import Error: Cannot import preprocessing modules

**Solution:**
```python
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
```

### GPU Not Detected

**Check ROCm:**
```bash
rocm-smi
```

**Verify PyTorch CUDA:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### NumPy Version Conflict

Warning tentang numpy version dengan pandas adalah normal. PyTorch 1.7.0 memerlukan numpy<1.20, sementara pandas terbaru memerlukan numpy>=1.20.3.

**Option 1:** Keep numpy 1.19.x (PyTorch compatibility)
**Option 2:** Upgrade numpy (pandas compatibility, may affect PyTorch)

---

## 📈 Next Steps

1. ✅ **Setup Complete** - Preprocessing pipeline verified
2. 🔄 **Batch Processing** - Process entire ImagiNet dataset
3. 🤖 **Model Development** - Build CNN classifier
4. 🎯 **Training** - Train on preprocessed data
5. 📊 **Evaluation** - Test on validation set
6. 🚀 **Deployment** - Export model for inference

---

## 📝 Citation

If you use this project, please cite:

```bibtex
@misc{artifice-vs-nature-2025,
  author = {Fadhly},
  title = {Artifice vs Nature: AI-Generated Image Detection},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/fadhly-git/artifice-vs-nature}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **PyTorch Team** - Deep learning framework
- **AMD ROCm** - GPU compute platform
- **ImagiNet** - Dataset for AI image detection
- **FFHQ** - High-quality face dataset

---

## 📬 Contact

- **Author:** Fadhly
- **GitHub:** [@fadhly-git](https://github.com/fadhly-git)
- **Project:** [artifice-vs-nature](https://github.com/fadhly-git/artifice-vs-nature)

---

**Last Updated:** November 11, 2025  
**Version:** 0.1.0  
**Status:** ✅ Setup Complete & Verified
