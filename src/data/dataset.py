import torch
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
        
        # Deterministic order
        if len(self.samples) > 0:
            paired = list(zip(self.samples, self.labels))
            paired.sort(key=lambda x: str(x[0]))
            self.samples, self.labels = zip(*paired)
            self.samples = list(self.samples)
            self.labels = list(self.labels)

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
                dct_feat = torch.from_numpy(np.load(dct_path)).float()
                # Normalize DCT features to prevent gradient explosion
                # Using approximate mean and std from DCT features (mean~1600, std~200)
                dct_feat = (dct_feat - 1600.0) / 200.0
            else:
                # Fallback: zero tensor
                dct_feat = torch.zeros(1024)
        else:
            # No DCT features: return dummy tensor
            dct_feat = torch.zeros(1024)
        
        return img_tensor, dct_feat, torch.tensor(label, dtype=torch.long)


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
                       train_ratio=0.8, seed=42, split_file=None, save_split=False):
    """
    Create train and validation dataloaders with automatic 80/20 split.
    
    Args:
        root_dir: Path to dataset root (contains 'real' and 'fake' folders)
        dct_dir: Path to DCT features directory (optional)
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading (use 0 for ROCm 3.5)
        train_ratio: Ratio of training data (default 0.8 for 80/20 split)
        seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Create full dataset (no transform yet)
    full_dataset = HybridDataset(root_dir, dct_dir=dct_dir, transform=None, is_training=True)

    # Build or load split indices
    total_len = len(full_dataset)
    train_size = int(train_ratio * total_len)
    val_size = total_len - train_size

    if split_file and Path(split_file).exists():
        idx = torch.load(split_file, map_location='cpu')
        train_indices = list(map(int, idx['train_idx']))
        val_indices = list(map(int, idx['val_idx']))
    else:
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(total_len, generator=gen).tolist()
        train_indices = perm[:train_size]
        val_indices = perm[train_size:]
        if split_file and save_split:
            Path(split_file).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'train_idx': train_indices,
                'val_idx': val_indices
            }, split_file)

    # Build train dataset with training transforms
    train_dataset = HybridDataset(root_dir, dct_dir=dct_dir,
                                  transform=get_transforms(is_training=True), is_training=True)
    train_dataset.samples = [full_dataset.samples[i] for i in train_indices]
    train_dataset.labels = [full_dataset.labels[i] for i in train_indices]

    # Build val dataset with validation transforms
    val_dataset_obj = HybridDataset(root_dir, dct_dir=dct_dir,
                                    transform=get_transforms(is_training=False), is_training=False)
    val_dataset_obj.samples = [full_dataset.samples[i] for i in val_indices]
    val_dataset_obj.labels = [full_dataset.labels[i] for i in val_indices]
    
    # Create dataloaders
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
    print(f"   Train: {len(train_indices)} images ({train_ratio*100:.0f}%)")
    print(f"   Val: {len(val_indices)} images ({(1-train_ratio)*100:.0f}%)")

    return train_loader, val_loader