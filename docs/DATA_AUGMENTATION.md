# Data Augmentation Pipeline

Complete 6-stage preprocessing and augmentation pipeline for AI-generated image detection, based on research papers and best practices.

## 🎯 Pipeline Overview

**⚠️ IMPORTANT: JPEG Compression BEFORE Resize (Real-world scenario)**

```
Input Image (Original, any size)
    ↓
Stage 1: JPEG Compression (Q=30-70) 🔴 [Paper 3] ← FIRST!
    ↓
Stage 2: Resize to 256x256
    ↓
Stage 3: Center Crop to 224x224
    ↓
Stage 4: Geometric Augmentation (Flip, Rotation ±15°) [Papers 1,2,3,5,8]
    ↓
Stage 5: Normalization (ImageNet stats)
    ↓
Stage 6: Random Masking (20%) 🟡 [Paper 5]
    ↓
Output Tensor (3, 224, 224) - Ready for Model
```

## 📋 Features

### 🔴 JPEG Compression Simulation (Stage 1)
- **Purpose**: Simulate real-world compression (social media, messaging apps)
- **Implementation**: Random JPEG quality (30-70) - **APPLIED FIRST, BEFORE RESIZE**
- **Research**: Paper 3 - "Images on social media often use Q<70, detectors fail without this simulation"
- **Impact**: Robust detection on compressed images
- **⚠️ Critical**: Must be done BEFORE resize to simulate real-world scenario correctly

### 🟡 Random Masking (Stage 6)
- **Purpose**: Force model to focus on artifacts, not content
- **Implementation**: Randomly mask 20% of pixels
- **Research**: Paper 5 - "Force focus on artifacts, reduce content bias"
- **Impact**: Better generalization, less overfitting to content

### Other Augmentations
- **Geometric**: RandomHorizontalFlip, RandomRotation(±15°)
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Resize**: 256 → CenterCrop 224 (standard for ResNet, EfficientNet, ViT)

## 🚀 Quick Start

### Basic Usage

```python
from src.data_transforms import get_training_transforms, get_validation_transforms

# Training transforms (with all augmentations)
train_transform = get_training_transforms()

# Validation/Test transforms (no augmentation)
val_transform = get_validation_transforms()

# Apply to image
from PIL import Image
img = Image.open('path/to/image.jpg')
tensor = train_transform(img)  # Returns torch.Tensor (3, 224, 224)
```

### Using Presets

```python
from src.data_transforms import get_preset_transforms

# Full augmentation (recommended for training)
train_transform = get_preset_transforms('full_augmentation')

# No masking (good for initial experiments)
train_transform = get_preset_transforms('no_masking')

# Aggressive JPEG compression (for social media scenarios)
train_transform = get_preset_transforms('aggressive_jpeg')

# Minimal augmentation (for clean datasets)
train_transform = get_preset_transforms('minimal')
```

### With PyTorch DataLoader

```python
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.data_transforms import get_preset_transforms, get_validation_transforms

# Setup transforms
train_transform = get_preset_transforms('full_augmentation')
val_transform = get_validation_transforms()

# Create datasets
train_dataset = ImageFolder('data/train', transform=train_transform)
val_dataset = ImageFolder('data/val', transform=val_transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Training loop
for images, labels in train_loader:
    # images: (batch_size, 3, 224, 224)
    # labels: (batch_size,)
    outputs = model(images)
    loss = criterion(outputs, labels)
    # ... backward pass ...
```

## ⚙️ Configuration Options

### Preset Configurations

| Preset | JPEG Compression | Random Masking | Rotation | Use Case |
|--------|------------------|----------------|----------|----------|
| **full_augmentation** | ✅ Q=30-70 | ✅ 20% | ±15° | **Recommended** - Full pipeline for training |
| **no_masking** | ✅ Q=30-70 | ❌ | ±15° | Initial training without masking |
| **aggressive_jpeg** | ✅ Q=10-50 | ✅ 20% | ±15° | Heavy social media compression |
| **minimal** | ❌ | ❌ | ±10° | Clean datasets with minimal augmentation |

### Custom Configuration

```python
from src.data_transforms import get_training_transforms

transform = get_training_transforms(
    image_size=224,                    # Final image size
    resize_size=256,                   # Size before center crop
    jpeg_quality_range=(30, 70),       # JPEG quality range
    rotation_degrees=15,               # Max rotation angle
    mask_ratio=0.2,                    # 20% masking
    use_jpeg_compression=True,         # Enable JPEG compression
    use_masking=True                   # Enable random masking
)
```

## 🎨 Visualization

Visualize augmentation effects:

```python
from src.visualize_augmentations import visualize_augmentations

# Visualize all stages
visualize_augmentations(
    'path/to/image.jpg',
    save_path='augmentation_stages.png'
)
```

Or from command line:

```bash
# Show augmentation stages
python src/visualize_augmentations.py \
    --image data/sample.jpg \
    --mode stages \
    --output results/augmentation_stages.png

# Show multiple random augmentations
python src/visualize_augmentations.py \
    --image data/sample.jpg \
    --mode multiple \
    --samples 8 \
    --output results/augmentation_multiple.png

# Compare presets
python src/visualize_augmentations.py \
    --image data/sample.jpg \
    --mode presets \
    --output results/augmentation_presets.png
```

## 🔬 Research Background

### Stage 2: JPEG Compression
- **Problem**: Images on social media are often compressed (Q<70)
- **Impact**: Detectors trained on high-quality images fail on compressed images
- **Solution**: Simulate compression during training
- **Source**: Paper 3

### Stage 6: Random Masking
- **Problem**: Models can overfit to image content instead of AI artifacts
- **Impact**: Model learns "what" is in the image, not "how" it was generated
- **Solution**: Mask 20% of pixels to force focus on artifacts
- **Source**: Paper 5

### Geometric Augmentation
- **Purpose**: Improve generalization and robustness
- **Methods**: Horizontal flip, rotation (±15°)
- **Sources**: Papers 1, 2, 3, 5, 8

## 📊 Output Specifications

### Training Transform Output
- **Shape**: `(3, 224, 224)` - Channels, Height, Width
- **Type**: `torch.Tensor`
- **Range**: Normalized values (approximately -2 to +2 after normalization)
- **Mean**: `[0.485, 0.456, 0.406]` (ImageNet)
- **Std**: `[0.229, 0.224, 0.225]` (ImageNet)

### Validation Transform Output
- **Shape**: `(3, 224, 224)`
- **Type**: `torch.Tensor`
- **Range**: Normalized values (same as training)
- **Augmentation**: None (deterministic)

## ✅ Compatibility

- **PyTorch**: 1.7.0a0 or higher
- **TorchVision**: 0.8.0a0 or higher
- **PIL/Pillow**: Any recent version
- **Python**: 3.7+

## 📝 Examples

### Example 1: Simple Training

```python
from src.data_transforms import get_preset_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Get transform
transform = get_preset_transforms('full_augmentation')

# Load dataset
dataset = ImageFolder('data/processed/imaginet/subset_224', transform=transform)

# Create dataloader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Use in training
for images, labels in loader:
    outputs = model(images)
    # ...
```

### Example 2: Custom Augmentation

```python
from src.data_transforms import get_training_transforms

# Custom settings
transform = get_training_transforms(
    image_size=224,
    jpeg_quality_range=(20, 60),  # More aggressive compression
    rotation_degrees=20,          # More rotation
    mask_ratio=0.3,              # 30% masking
    use_jpeg_compression=True,
    use_masking=True
)
```

### Example 3: A/B Testing Presets

```python
from src.data_transforms import get_preset_transforms

# Test different presets
presets = ['full_augmentation', 'no_masking', 'aggressive_jpeg', 'minimal']

for preset_name in presets:
    transform = get_preset_transforms(preset_name)
    # Train model with this preset
    # Compare results
```

## 🛠️ Custom Transforms

You can also use individual custom transforms:

```python
from src.data_transforms import RandomJPEGCompression, RandomMasking
import torchvision.transforms as T

# Build custom pipeline
custom_transform = T.Compose([
    T.Resize(256),
    RandomJPEGCompression(quality_range=(30, 70)),  # Custom JPEG
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    RandomMasking(mask_ratio=0.2)  # Custom masking
])
```

## 📚 API Reference

### Main Functions

- `get_training_transforms(**kwargs)` - Get training pipeline with all augmentations
- `get_validation_transforms(**kwargs)` - Get validation pipeline (no augmentation)
- `get_test_transforms(**kwargs)` - Get test pipeline (with optional JPEG)
- `get_preset_transforms(preset, **kwargs)` - Get preset configuration

### Custom Transforms

- `RandomJPEGCompression(quality_range=(30, 70))` - Random JPEG compression
- `RandomMasking(mask_ratio=0.2, mask_value=0)` - Random pixel masking

## 🎯 Best Practices

1. **Training**: Use `get_preset_transforms('full_augmentation')`
2. **Validation**: Use `get_validation_transforms()` (no augmentation)
3. **Testing**: Use `get_test_transforms()` (with JPEG for real-world simulation)
4. **Experiments**: Try different presets and compare results
5. **Custom Models**: Adjust `image_size` and `resize_size` if needed

## 🔍 Troubleshooting

### Issue: "Import matplotlib could not be resolved"
**Solution**: Install matplotlib (optional, only for visualization)
```bash
pip install matplotlib
```

### Issue: Images look wrong/distorted
**Solution**: Make sure to denormalize before visualization
```python
from src.visualize_augmentations import denormalize
img_denorm = denormalize(tensor)
```

### Issue: Slow data loading
**Solution**: Increase `num_workers` in DataLoader
```python
loader = DataLoader(dataset, batch_size=32, num_workers=4)
```

## 📖 References

1. Paper 3: JPEG compression simulation for robust detection
2. Paper 5: Random masking to reduce content bias
3. Papers 1, 2, 3, 5, 8: Geometric augmentation for generalization

## 📄 License

Part of the Artifice vs Nature project.
