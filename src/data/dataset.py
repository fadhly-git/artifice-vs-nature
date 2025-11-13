import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import numpy as np
import random

class HybridDataset(Dataset):
    """
    Dataset untuk AI-generated image detection dengan preprocessing on-the-fly.
    
    Args:
        root_dir: Path ke folder utama yang berisi subfolder 'real' dan 'fake'
        dct_dir: Path ke folder berisi file DCT features (.npy)
        transform: Transformasi untuk training/validation
        is_training: Boolean untuk menentukan apakah mode training (dengan augmentasi)
    
    Notes:
        - DCT features loaded with memory-mapped I/O for efficiency
        - Arrays are copied to writable format for PyTorch compatibility (v1.11+)
        - Supports both 'real' (label=0) and 'fake' (label=1) images
    """
    def __init__(self, root_dir, dct_dir=None, transform=None, is_training=True):
        self.root_dir = Path(root_dir)
        self.dct_dir = Path(dct_dir) if dct_dir else None
        self.transform = transform
        self.is_training = is_training
        
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
        
        # Load image with optimized PIL settings
        with Image.open(img_path) as img:
            # Convert to RGB immediately (lazy loading optimization)
            img = img.convert('RGB')
            
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


def get_transforms(is_training=True, fast_mode=False, use_gpu=False):
    """
    Get appropriate transforms for training or validation.
    
    Args:
        is_training: If True, include augmentations
        fast_mode: If True, use minimal augmentation (faster preprocessing)
                   If 'balanced', use moderate augmentation (good speed + accuracy)
        use_gpu: If True, return minimal CPU transforms (augmentation done on GPU via Kornia)
    
    Returns:
        torchvision.transforms.Compose object
    """
    if use_gpu:
        # GPU mode: minimal CPU transforms (just resize + to tensor)
        # Augmentation will be done on GPU using Kornia
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            # Note: Normalization will be done on GPU too
        ])
    
    if is_training:
        if fast_mode == 'balanced':
            # Balanced: moderate augmentation (ColorJitter only, no rotation/crop)
            return T.Compose([
                T.Resize((224, 224)),  # Direct resize (faster)
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.15, contrast=0.15),  # Lighter color jitter
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif fast_mode:
            # Fast training: minimal augmentation
            return T.Compose([
                T.Resize((224, 224)),  # Direct resize (faster than crop)
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Full training: with heavy augmentation
            return T.Compose([
                T.Resize(256),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    else:
        # Validation: no augmentation
        return T.Compose([
            T.Resize((224, 224)),  # Direct resize for validation
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(root_dir, dct_dir=None, batch_size=4, num_workers=0, 
                       train_ratio=0.8, seed=42, fast_mode=False, use_gpu_aug=False):
    """
    Create train and validation dataloaders with automatic 80/20 split.
    
    Args:
        root_dir: Path to dataset root (contains 'real' and 'fake' folders)
        dct_dir: Path to DCT features directory (optional)
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading (use 0 for ROCm 3.5)
        train_ratio: Ratio of training data (default 0.8 for 80/20 split)
        seed: Random seed for reproducibility
        fast_mode: If True, use minimal augmentation (faster preprocessing)
        use_gpu_aug: If True, use Kornia for GPU-based augmentation (MUCH faster!)
    
    Returns:
        train_loader, val_loader
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Create full dataset (no transform yet)
    full_dataset = HybridDataset(root_dir, dct_dir=dct_dir, transform=None, is_training=True)
    
    # Split into train and validation
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Apply transforms
    train_dataset.dataset.transform = get_transforms(is_training=True, fast_mode=fast_mode, use_gpu=use_gpu_aug)
    train_dataset.dataset.is_training = True
    
    # Create separate dataset for validation with different transform
    val_dataset_obj = HybridDataset(
        root_dir, 
        dct_dir=dct_dir, 
        transform=get_transforms(is_training=False, fast_mode=fast_mode, use_gpu=use_gpu_aug), 
        is_training=False
    )
    
    # Use the same indices as val_dataset
    val_samples = [train_dataset.dataset.samples[i] for i in val_dataset.indices]
    val_labels = [train_dataset.dataset.labels[i] for i in val_dataset.indices]
    val_dataset_obj.samples = val_samples
    val_dataset_obj.labels = val_labels
    
    # Create dataloaders
    # Use prefetch_factor to pre-load more batches (overlaps GPU computation with data loading)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
        persistent_workers=False,  # Don't keep workers alive (saves memory)
        prefetch_factor=2 if num_workers > 0 else 2  # Pre-load 2 batches per worker
    )
    
    val_loader = DataLoader(
        val_dataset_obj,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"   Total: {len(full_dataset)} images")
    print(f"   Train: {train_size} images ({train_ratio*100:.0f}%)")
    print(f"   Val: {val_size} images ({(1-train_ratio)*100:.0f}%)")
    print(f"   Fast mode: {fast_mode}")
    print(f"   GPU augmentation: {use_gpu_aug}")
    
    return train_loader, val_loader


class PyTorchGPUAugment(nn.Module):
    """
    Pure PyTorch GPU augmentation (NO MAGMA required - ROCm compatible!)
    Input: x tensor shape (B, C, H, W), values in [0..1] float
    """
    def __init__(self, is_training=True, device='cuda'):
        super().__init__()
        self.is_training = is_training
        self.device = device
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def random_horizontal_flip(self, x, p=0.5):
        if self.is_training and torch.rand(1, device=x.device) < p:
            return torch.flip(x, dims=[-1])
        return x

    def color_jitter(self, x, brightness=0.2, contrast=0.2, saturation=0.2):
        if not self.is_training:
            return x
        # brightness
        if brightness > 0:
            b = 1.0 + (torch.rand(1, device=x.device) * 2 - 1) * brightness
            x = x * b
        # contrast
        if contrast > 0:
            c = 1.0 + (torch.rand(1, device=x.device) * 2 - 1) * contrast
            mean = x.mean(dim=[2,3], keepdim=True)
            x = (x - mean) * c + mean
        # saturation
        if saturation > 0:
            s = 1.0 + (torch.rand(1, device=x.device) * 2 - 1) * saturation
            gray = (0.2989*x[:,0:1,:,:] + 0.5870*x[:,1:2,:,:] + 0.1140*x[:,2:3,:,:])
            x = (x - gray) * s + gray
        return x.clamp(0.0, 1.0)

    def gaussian_blur(self, x, kernel_size=3, sigma=1.0, p=0.3):
        if not self.is_training or torch.rand(1, device=x.device) >= p:
            return x
        half = kernel_size // 2
        coords = torch.arange(-half, half+1, device=x.device, dtype=torch.float32)
        g = torch.exp(-(coords**2) / (2 * (sigma**2)))
        g = g / g.sum()
        kernel_2d = g[:,None] * g[None,:]
        kernel_2d = kernel_2d.expand(x.shape[1], 1, kernel_size, kernel_size)
        import torch.nn.functional as F
        x = F.conv2d(x, kernel_2d, padding=half, groups=x.shape[1])
        return x

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, x):
        x = self.random_horizontal_flip(x, p=0.5)
        x = self.color_jitter(x, brightness=0.15, contrast=0.15, saturation=0.15)
        x = self.gaussian_blur(x, kernel_size=3, sigma=1.0, p=0.3)
        x = self.normalize(x)
        return x


def get_gpu_augmentation(is_training=True, device='cuda'):
    """
    Get pure PyTorch GPU augmentation (NO Kornia, NO MAGMA!)
    Works with ROCm 3.5 and older PyTorch versions.
    """
    return PyTorchGPUAugment(is_training=is_training, device=device).to(device)