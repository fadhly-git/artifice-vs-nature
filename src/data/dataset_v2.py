# File: src/data/dataset.py

import time
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import numpy as np
import random
import scipy.fftpack as fftpack  # For DCT on-the-fly
import io

import tqdm

class JPEGCompression(object):
    def __init__(self, quality_range=(70, 100)):
        self.quality_range = quality_range

    def __call__(self, img):
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

class HybridDataset(Dataset):
    def __init__(self, root_dir, dct_dir=None, transform=None, is_training=True, debug=False):
        self.debug = debug
        self.pbar = None
        if self.debug:
            print(f"[DEBUG] Dataset init: {root_dir}, training={is_training}")
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
        try:
            img_path = self.samples[idx]
            label = self.labels[idx]
            
            # Load image dengan error handling
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"❌ Error loading image {img_path}: {e}")
                # Return dummy data jika corrupt
                img = Image.new('RGB', (224, 224), color=(128, 128, 128))
            
            if self.transform:
                img = self.transform(img)
            
            dct_feat = self._get_dct_feat(img_path, img)
            
            return img, dct_feat, torch.tensor(label, dtype=torch.long)
        
        except Exception as e:
            print(f"❌ Error in __getitem__ at index {idx}: {e}")
            # Return dummy batch
            return (
                torch.randn(3, 224, 224),
                torch.randn(1024),
                torch.tensor(0, dtype=torch.long)
            )
    
    def _get_dct_feat(self, img_path, img_tensor):
        try:
            # If DCT dir exists, load .npy
            if self.dct_dir:
                dct_path = self.dct_dir / f"{img_path.stem}.npy"
                if dct_path.exists():
                    dct_array = np.load(dct_path, mmap_mode='r')
                    dct_feat = torch.from_numpy(np.array(dct_array, copy=True)).float()
                    
                    # 🔧 VALIDASI: Pastikan tidak ada NaN/Inf
                    if torch.isnan(dct_feat).any() or torch.isinf(dct_feat).any():
                        print(f"⚠️  Warning: NaN/Inf in DCT file {dct_path}, regenerating...")
                    else:
                        return dct_feat
            
            # Fallback: Compute DCT on-the-fly
            img_np = np.array(img_tensor.permute(1, 2, 0) * 255).astype(np.uint8)
            gray = np.mean(img_np, axis=2)
            gray_resized = np.array(Image.fromarray(gray.astype(np.uint8)).resize((32, 32)))
            
            # 🔧 VALIDASI: Check grayscale
            if np.isnan(gray_resized).any() or np.isinf(gray_resized).any():
                print(f"⚠️  Warning: NaN/Inf in grayscale conversion for {img_path}")
                gray_resized = np.zeros((32, 32))
            
            dct = fftpack.dct(fftpack.dct(gray_resized.T, norm='ortho').T, norm='ortho')
            dct_feat = torch.from_numpy(dct.flatten()[:1024]).float()
            
            return dct_feat
        
        except Exception as e:
            print(f"❌ Error computing DCT for {img_path}: {e}")
            # Return zero vector if error
            return torch.zeros(1024, dtype=torch.float32)

def get_transforms(is_training=True):
    """
    Get appropriate transforms for training or validation.
    
    Args:
        is_training: If True, include augmentations
    
    Returns:
        torchvision.transforms.Compose object
    """
    if is_training:
        # Training: with augmentation
        return T.Compose([
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
                       train_ratio=0.8, seed=42, debug=False):
    full_dataset = HybridDataset(root_dir, dct_dir=dct_dir, transform=None, is_training=True, debug=debug)
    
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_dataset.dataset.debug = debug
    
    # Apply transforms
    train_dataset.dataset.transform = get_transforms(is_training=True)
    train_dataset.dataset.is_training = True
    
    val_dataset_obj = HybridDataset(
        root_dir, 
        dct_dir=dct_dir, 
        transform=get_transforms(is_training=False),
        is_training=False
    )
    
    val_samples = [full_dataset.samples[i] for i in val_dataset.indices]
    val_labels = [full_dataset.labels[i] for i in val_dataset.indices]
    val_dataset_obj.samples = val_samples
    val_dataset_obj.labels = val_labels
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False if torch.cuda.is_available() else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset_obj,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False if torch.cuda.is_available() else False,
        drop_last=False
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"   Total: {len(full_dataset)} images")
    print(f"   Train: {train_size} images ({train_ratio*100:.0f}%)")
    print(f"   Val: {val_size} images ({(1-train_ratio)*100:.0f}%)")
    
    return train_loader, val_loader