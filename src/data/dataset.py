import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import numpy as np
import random
import io
from src.preprocessing.jpeg import apply_jpeg

class ProgressiveJpegTransform:
    """
    Progressive JPEG compression augmentation.
    Starts with high quality, gradually decreases as training progresses.
    """
    def __init__(self, current_epoch=0, max_epochs=50, prob=0.5):
        self.current_epoch = current_epoch
        self.max_epochs = max_epochs
        self.prob = prob
    
    def __call__(self, img):
        if random.random() > self.prob:
            return img
        
        # Progressive quality range based on epoch
        # Early epochs: quality 70-95 (mild compression)
        # Later epochs: quality 30-90 (aggressive compression)
        progress = min(1.0, self.current_epoch / (self.max_epochs * 0.6))
        
        min_quality = int(70 - 40 * progress)  # 70 -> 30
        max_quality = int(95 - 5 * progress)   # 95 -> 90
        
        quality = random.randint(min_quality, max_quality)
        
        # Apply JPEG compression
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')

class HybridDataset(Dataset):
    """
    Dataset untuk AI-generated image detection dengan preprocessing on-the-fly.
    
    Args:
        root_dir: Path ke folder utama yang berisi subfolder 'real' dan 'fake'
        dct_dir: Path ke folder berisi file DCT features (.npy)
        transform: Transformasi untuk training/validation
        is_training: Boolean untuk menentukan apakah mode training (dengan augmentasi)
        current_epoch: Current training epoch for progressive augmentation
        max_epochs: Total training epochs
    
    Notes:
        - DCT features loaded with memory-mapped I/O for efficiency
        - Arrays are copied to writable format for PyTorch compatibility (v1.11+)
        - Supports both 'real' (label=0) and 'fake' (label=1) images
    """
    def __init__(self, root_dir, dct_dir=None, transform=None, is_training=True, 
                 current_epoch=0, max_epochs=50):
        self.root_dir = Path(root_dir)
        self.dct_dir = Path(dct_dir) if dct_dir else None
        self.transform = transform
        self.is_training = is_training
        self.current_epoch = current_epoch
        self.max_epochs = max_epochs
        
        # Collect all image files
        self.samples = []
        self.labels = []
        
        # Scan real images (label=0)
        real_dir = self.root_dir / "real"
        if real_dir.exists():
            for img_path in real_dir.rglob("*.jpg"):
                self.samples.append(img_path)
                self.labels.append(0)
            for img_path in real_dir.rglob("*.png"):
                self.samples.append(img_path)
                self.labels.append(0)
        
        # Scan fake images (label=1)
        fake_dir = self.root_dir / "fake"
        if fake_dir.exists():
            for img_path in fake_dir.rglob("*.jpg"):
                self.samples.append(img_path)
                self.labels.append(1)
            for img_path in fake_dir.rglob("*.png"):
                self.samples.append(img_path)
                self.labels.append(1)
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        print(f"{'[TRAIN]' if is_training else '[VAL]'} Loaded {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            img_tensor = self.transform(img)
        else:
            # Default: just convert to tensor
            img_tensor = T.ToTensor()(img)
        
        # Load DCT features if directory provided
        if self.dct_dir:
            dct_path = self.dct_dir / f"{img_path.stem}.npy"
            if dct_path.exists():
                # Load with mmap and copy to writable array (fixes PyTorch warning)
                dct_array = np.load(dct_path, mmap_mode='r')
                dct_feat = torch.from_numpy(np.array(dct_array, copy=True)).float()
            else:
                # Fallback: zero tensor
                dct_feat = torch.zeros(1024)
        else:
            # No DCT features: return dummy tensor
            dct_feat = torch.zeros(1024)
        
        return img_tensor, dct_feat, torch.tensor(label, dtype=torch.long)


def get_transforms(is_training=True, current_epoch=0, max_epochs=50):
    """
    Get appropriate transforms for training or validation.
    
    Args:
        is_training: If True, include augmentations
        current_epoch: Current training epoch for progressive JPEG
        max_epochs: Total epochs for progressive JPEG
    
    Returns:
        torchvision.transforms.Compose object
    """
    if is_training:
        # Training: with progressive augmentation
        return T.Compose([
            ProgressiveJpegTransform(current_epoch=current_epoch, max_epochs=max_epochs, prob=0.5),
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation: no augmentation
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(root_dir, dct_dir=None, batch_size=4, num_workers=0, 
                       train_ratio=0.8, seed=42, current_epoch=0, max_epochs=50):
    """
    Create train and validation dataloaders with automatic 80/20 split.
    
    Args:
        root_dir: Path to dataset root (contains 'real' and 'fake' folders)
        dct_dir: Path to DCT features directory (optional)
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading (use 0 for ROCm 3.5)
        train_ratio: Ratio of training data (default 0.8 for 80/20 split)
        seed: Random seed for reproducibility
        current_epoch: Current epoch for progressive augmentation
        max_epochs: Total epochs for progressive augmentation
    
    Returns:
        train_loader, val_loader
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Create full dataset with progressive transforms
    full_dataset = HybridDataset(
        root_dir, 
        dct_dir=dct_dir, 
        transform=get_transforms(is_training=True, current_epoch=current_epoch, max_epochs=max_epochs),
        is_training=True,
        current_epoch=current_epoch,
        max_epochs=max_epochs
    )
    
    # Split into train and validation
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create separate dataset for validation with different transform (no augmentation)
    val_dataset_obj = HybridDataset(
        root_dir, 
        dct_dir=dct_dir, 
        transform=get_transforms(is_training=False), 
        is_training=False,
        current_epoch=current_epoch,
        max_epochs=max_epochs
    )
    
    # Use the same indices as val_dataset
    val_samples = [train_dataset.dataset.samples[i] for i in val_dataset.indices]
    val_labels = [train_dataset.dataset.labels[i] for i in val_dataset.indices]
    val_dataset_obj.samples = val_samples
    val_dataset_obj.labels = val_labels
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset_obj,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"   Total: {len(full_dataset)} images")
    print(f"   Train: {train_size} images ({train_ratio*100:.0f}%)")
    print(f"   Val: {val_size} images ({(1-train_ratio)*100:.0f}%)")
    
    return train_loader, val_loader