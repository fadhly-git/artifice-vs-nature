# 🚀 Quick Start - Artifice vs Nature

## 📋 Complete Pipeline

### 🔵 Preprocessing (Done ONCE - Save to Disk)

| No | Tahap | Input → Output | Kode | Paper |
|----|-------|----------------|------|-------|
| 1 | Resize | File → PIL (224×224) | `resize_images.py` | - |
| 2 | **JPEG Compress** | PIL → PIL compressed | `JPEG quality=70` | **Paper 3** |
| 3 | Resize | PIL → PIL (256×256) | `img.resize(256)` | - |
| 4 | Center Crop | PIL → PIL (224×224) | `crop to 224×224` | All |
| 5 | ToTensor | PIL → Tensor (3,224,224) | `T.ToTensor()` | - |
| 6 | **Geometric** | Tensor → 4 variants | flip, rot±15° | Papers 1,2,3,5,8 |
| 7 | **Save** | Tensor → .pt files | `torch.save()` | - |

**Output:** Setiap gambar → 4 file .pt (`_orig`, `_flip`, `_rot15`, `_rot-15`)

---

### 🟢 Training (Load from Disk - FAST!)

| No | Tahap | Input → Output | Kode | Paper |
|----|-------|----------------|------|-------|
| 1 | Load | .pt → Tensor | `torch.load()` | - |
| 2 | **Normalize** | Tensor → z-score | `mean=[0.485,0.456,0.406]` | All |
| 3 | **Masking** | Tensor → masked | `mask 20% pixels` | **Paper 5** |

---

## 🎯 Setup

### Step 1: Resize to 224×224
```bash
python src/resize_images.py \
    --input data/processed/imaginet/subset \
    --output data/processed/imaginet/subset_224 \
    --faces
```

### Step 2: Full Preprocessing (JPEG + Augmentations)
```bash
python src/preprocess_full.py \
    --input data/processed/imaginet/subset_224 \
    --output data/processed/imaginet/preprocessed \
    --faces
```

**Output:**
```
preprocessed/
  real/
    img001_orig.pt
    img001_flip.pt
    img001_rot15.pt
    img001_rot-15.pt
    ...
  fake/
    ...
```

### Step 3: Training (Load Preprocessed Tensors)
```python
from src.preprocessed_dataset import PreprocessedDataset
from src.data_transforms import get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader

# Training (normalize + mask)
train_transform = get_train_transforms()
train_dataset = PreprocessedDataset(
    'data/processed/imaginet/preprocessed',
    transform=train_transform
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Validation (normalize only)
val_transform = get_val_transforms()
val_dataset = PreprocessedDataset(
    'data/processed/imaginet/preprocessed',
    transform=val_transform
)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

---

## 💡 Key Points

### ✅ Preprocessing Saves ALL to Disk
**Keuntungan:**
- Training SANGAT cepat (load tensor vs decode image)
- GPU fully utilized (no preprocessing overhead)
- Reproducible (augmentasi fixed)

### ✅ JPEG Compression (Q=70%)
**Kenapa?** Real-world: Platform sosmed compress gambar sebelum resize

### ✅ Geometric Variants (4x per gambar)
- Original
- Horizontal flip
- Rotate +15°
- Rotate -15°

### ✅ Random Masking (20%) - Training Only
**Kenapa?** Paper 5: Paksa model fokus pada artefak, bukan konten

---

## 📊 Pipeline Flow

### Preprocessing (One-time):
```
image.jpg
  ↓ Load
PIL (224×224)
  ↓ JPEG compress (Q=70)
PIL compressed
  ↓ Resize (256)
PIL (256×256)
  ↓ Crop (224)
PIL (224×224)
  ↓ ToTensor
Tensor (3,224,224)
  ↓ Geometric (4 variants)
  ├─ image_orig.pt
  ├─ image_flip.pt
  ├─ image_rot15.pt
  └─ image_rot-15.pt
```

### Training (Every batch):
```
image_orig.pt
  ↓ Load
Tensor (3,224,224) [0,1]
  ↓ Normalize
Tensor z-scored
  ↓ Random Mask (20%)
Tensor masked
  ↓
Model
```

---

## 📖 More Info

- **Notebook Tutorial:** `notebooks/note.ipynb`
- **Preprocessing Script:** `src/preprocess_full.py`
- **Dataset Loader:** `src/preprocessed_dataset.py`
- **Transforms:** `src/data_transforms.py`

---

**Compatible:** PyTorch 1.7.0a0, TorchVision 0.8.0a0
