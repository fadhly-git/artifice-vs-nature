# Preprocessing Modules Documentation

## Overview

Modul preprocessing untuk deteksi gambar AI-generated vs real. Pipeline terdiri dari 6 tahap yang dieksekusi secara berurutan.

## Architecture

```
Input Image (any size)
    ↓
[1] JPEG Compression (quality: 30-90, prob: 0.7)
    ↓
[2] Resize to 224×224 (LANCZOS)
    ↓
[3] Augmentation (flip, brightness, contrast)
    ↓
[4] Normalization (ImageNet stats)
    ↓
[5] Random Masking (20% area)
    ↓
[6] DCT Feature Extraction (top-1024 coeffs)
    ↓
Output: {img_masked, dct_feat, intermediates}
```

---

## Module Details

### 1. `jpeg.py` - JPEG Compression

**Function:** `apply_jpeg(img_pil, quality_range=(30,90), prob=0.7)`

**Purpose:** Simulate real-world image degradation to improve model robustness

**Parameters:**
- `img_pil` (PIL.Image): Input image
- `quality_range` (tuple): Min/max JPEG quality
- `prob` (float): Probability of applying compression

**Returns:** `(compressed_img, quality_used)`

**Example:**
```python
from PIL import Image
from src.preprocessing.jpeg import apply_jpeg

img = Image.open("sample.jpg")
img_compressed, quality = apply_jpeg(img, quality_range=(30, 90), prob=0.7)
print(f"Applied JPEG quality: {quality}")
```

---

### 2. `resize.py` - Image Resizing

**Function:** `resize_to_224(img_pil)`

**Purpose:** Standardize image dimensions for model input

**Parameters:**
- `img_pil` (PIL.Image): Input image (any size)

**Returns:** PIL.Image (224×224)

**Example:**
```python
from src.preprocessing.resize import resize_to_224

img_resized = resize_to_224(img)
print(img_resized.size)  # (224, 224)
```

---

### 3. `augment.py` - Data Augmentation

**Function:** `apply_augment(tensor)`

**Purpose:** Apply random augmentations for data diversity

**Augmentations:**
- Random horizontal flip (p=0.5)
- Random brightness adjustment (±0.2)
- Random contrast adjustment (±0.2)

**Parameters:**
- `tensor` (torch.Tensor): Image tensor (C, H, W) in [0, 1]

**Returns:** torch.Tensor (C, H, W)

**Example:**
```python
import torch
from src.preprocessing.augment import apply_augment

tensor = torch.rand(3, 224, 224)
tensor_aug = apply_augment(tensor)
```

---

### 4. `normalize.py` - Normalization

**Function:** `normalize_image(tensor)`

**Purpose:** Normalize using ImageNet statistics for transfer learning

**Statistics:**
- Mean: [0.485, 0.456, 0.406]
- Std: [0.229, 0.224, 0.225]

**Parameters:**
- `tensor` (torch.Tensor): Image tensor (C, H, W) in [0, 1]

**Returns:** torch.Tensor (C, H, W) normalized

**Example:**
```python
from src.preprocessing.normalize import normalize_image

tensor_norm = normalize_image(tensor)
print(f"Mean: {tensor_norm.mean():.4f}, Std: {tensor_norm.std():.4f}")
```

---

### 5. `mask.py` - Random Masking

**Function:** `apply_mask(tensor, ratio=0.2)`

**Purpose:** Force model to learn robust features (inspired by MAE)

**Parameters:**
- `tensor` (torch.Tensor): Image tensor (C, H, W)
- `ratio` (float): Fraction of image area to mask (0-1)

**Returns:** torch.Tensor with masked regions set to 0

**Example:**
```python
from src.preprocessing.mask import apply_mask

tensor_masked = apply_mask(tensor, ratio=0.2)
# 20% of image area is randomly masked
```

---

### 6. `dct.py` - DCT Feature Extraction

**Function:** `extract_dct_features(img_np, top_k=1024, block_size=8)`

**Purpose:** Extract frequency-domain features for AI artifact detection

**Method:**
- Divide image into 8×8 blocks
- Apply 2D DCT to each block
- Extract top-k coefficients by magnitude

**Parameters:**
- `img_np` (np.ndarray): Grayscale image (H, W)
- `top_k` (int): Number of top coefficients to extract
- `block_size` (int): DCT block size (default 8)

**Returns:** np.ndarray (top_k,)

**Example:**
```python
import numpy as np
from PIL import Image
from src.preprocessing.dct import extract_dct_features

img = Image.open("sample.jpg").convert('L')
img_np = np.array(img)
dct_feat = extract_dct_features(img_np, top_k=1024)
print(dct_feat.shape)  # (1024,)
```

---

### 7. `pipeline.py` - Full Pipeline

**Function:** `preprocess_full(img_path, config)`

**Purpose:** Orchestrate all preprocessing steps

**Parameters:**
- `img_path` (str): Path to input image
- `config` (dict): Configuration dictionary

**Returns:** Dictionary with keys:
```python
{
    'img_masked': torch.FloatTensor(3, 224, 224),  # Final preprocessed image
    'dct_feat': torch.FloatTensor(1024,),          # DCT features
    'intermediates': {                              # For visualization
        'original': PIL.Image,
        'original_size': tuple,
        'jpeg': PIL.Image,
        'jpeg_quality': int,
        'resized': PIL.Image,
        'augmented': torch.Tensor,
        'normalized': torch.Tensor,
        'masked': torch.Tensor,
        'dct_values': np.ndarray
    }
}
```

**Example:**
```python
from src.preprocessing.pipeline import preprocess_full
import yaml

# Load config
with open('configs/preprocessing.yaml') as f:
    config = yaml.safe_load(f)

# Process image
result = preprocess_full("path/to/image.jpg", config)

# Use outputs
img_masked = result['img_masked']  # For model input
dct_feat = result['dct_feat']      # For feature analysis
intermediates = result['intermediates']  # For visualization
```

---

## Utilities

### `visualization.py` - Pipeline Visualization

**Function:** `plot_preprocessing_pipeline(intermediates, figsize=(14,10))`

**Purpose:** Visualize 4 key preprocessing stages

**Visualization:**
1. **Original Image** - Raw input
2. **After JPEG** - With compression artifacts
3. **Masked Image** - With 20% random masking
4. **DCT Coefficients** - Top 100 frequency components

**Example:**
```python
from src.utils import plot_preprocessing_pipeline

result = preprocess_full("sample.jpg", config)
plot_preprocessing_pipeline(result['intermediates'])
```

---

## Configuration File

### `configs/preprocessing.yaml`

```yaml
jpeg:
  prob: 0.7                # Probability of applying JPEG compression
  quality_range: [30, 90]  # Min/max JPEG quality

mask:
  ratio: 0.2               # Fraction of image to mask

dct:
  top_k: 1024              # Number of DCT coefficients
  block_size: 8            # DCT block size (8×8)

augmentation:
  horizontal_flip_prob: 0.5
  brightness_range: [-0.2, 0.2]
  contrast_range: [-0.2, 0.2]

normalization:
  mean: [0.485, 0.456, 0.406]  # ImageNet mean
  std: [0.229, 0.224, 0.225]   # ImageNet std
```

---

## Batch Processing Example

```python
import glob
import torch
from src.preprocessing.pipeline import preprocess_full
import yaml

# Load config
with open('configs/preprocessing.yaml') as f:
    config = yaml.safe_load(f)

# Get all images
image_paths = glob.glob("data/raw/**/*.jpg", recursive=True)

# Process batch
batch_masked = []
batch_dct = []

for img_path in image_paths[:32]:  # Process 32 images
    result = preprocess_full(img_path, config)
    batch_masked.append(result['img_masked'])
    batch_dct.append(result['dct_feat'])

# Stack into batches
batch_masked = torch.stack(batch_masked)  # (32, 3, 224, 224)
batch_dct = torch.stack(batch_dct)        # (32, 1024)

print(f"Batch shapes: {batch_masked.shape}, {batch_dct.shape}")
```

---

## Integration with DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader
from src.preprocessing.pipeline import preprocess_full
import yaml

class ImageDataset(Dataset):
    def __init__(self, image_paths, config):
        self.image_paths = image_paths
        self.config = config
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        result = preprocess_full(self.image_paths[idx], self.config)
        return result['img_masked'], result['dct_feat']

# Load config
with open('configs/preprocessing.yaml') as f:
    config = yaml.safe_load(f)

# Create dataset and dataloader
image_paths = glob.glob("data/raw/**/*.jpg", recursive=True)
dataset = ImageDataset(image_paths, config)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Training loop
for batch_img, batch_dct in dataloader:
    # batch_img: (32, 3, 224, 224)
    # batch_dct: (32, 1024)
    pass
```

---

## Performance Notes

### AMD RX 580 (ROCm 3.5)
- **Preprocessing time:** ~50-100ms per image (CPU-bound)
- **Batch size 32:** ~2-3 seconds
- **GPU utilization:** <10% (most operations on CPU)
- **Memory:** ~1 MB per processed image

### Optimization Tips
1. Use `num_workers > 0` in DataLoader for parallel preprocessing
2. Cache preprocessed images to disk for repeated use
3. DCT computation is the slowest step (~30-40ms)
4. Consider using GPU for augmentation if available

---

## Testing

Each module includes a test section in `if __name__ == "__main__"`:

```bash
# Test individual modules
python src/preprocessing/jpeg.py
python src/preprocessing/resize.py
python src/preprocessing/augment.py
python src/preprocessing/normalize.py
python src/preprocessing/mask.py
python src/preprocessing/dct.py
python src/preprocessing/pipeline.py
```

---

## References

1. **JPEG Compression:** Data augmentation for deepfake detection
2. **DCT Features:** Frequency analysis for AI-generated image artifacts
3. **Masking:** Inspired by MAE (Masked Autoencoders Are Scalable Vision Learners)
4. **Normalization:** Transfer learning from ImageNet pretrained models

---

**Last Updated:** November 11, 2025  
**Version:** 0.1.0  
**Compatibility:** PyTorch 1.7.0, Python 3.8+
