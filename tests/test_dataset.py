"""
Tests for HybridDataset and related functions
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
import shutil

from src.data.dataset import HybridDataset, get_transforms, create_dataloaders


@pytest.fixture
def temp_dataset_dir():
    """Create temporary dataset directory structure"""
    temp_dir = tempfile.mkdtemp()
    
    # Create real and fake directories
    real_dir = Path(temp_dir) / "real"
    fake_dir = Path(temp_dir) / "fake"
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)
    
    # Create dummy images
    for i in range(5):
        img = Image.new('RGB', (224, 224), color=(i*50, i*50, i*50))
        img.save(real_dir / f"real_{i}.jpg")
        img.save(fake_dir / f"fake_{i}.png")
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_dct_dir():
    """Create temporary DCT features directory"""
    temp_dir = tempfile.mkdtemp()
    
    # Create dummy DCT features
    for i in range(5):
        dct_feat = np.random.randn(1024).astype(np.float32)
        np.save(Path(temp_dir) / f"real_{i}.npy", dct_feat)
        np.save(Path(temp_dir) / f"fake_{i}.npy", dct_feat)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestHybridDataset:
    """Test cases for HybridDataset class"""
    
    def test_dataset_initialization(self, temp_dataset_dir):
        """Test dataset loads correctly"""
        dataset = HybridDataset(temp_dataset_dir, is_training=True)
        
        assert len(dataset) == 10  # 5 real + 5 fake
        assert len(dataset.samples) == 10
        assert len(dataset.labels) == 10
    
    def test_dataset_labels(self, temp_dataset_dir):
        """Test labels are assigned correctly"""
        dataset = HybridDataset(temp_dataset_dir)
        
        # Count real (0) and fake (1) labels
        real_count = sum(1 for label in dataset.labels if label == 0)
        fake_count = sum(1 for label in dataset.labels if label == 1)
        
        assert real_count == 5
        assert fake_count == 5
    
    def test_dataset_getitem_without_dct(self, temp_dataset_dir):
        """Test __getitem__ returns correct format without DCT"""
        dataset = HybridDataset(temp_dataset_dir)
        
        img_tensor, dct_feat, label = dataset[0]
        
        assert isinstance(img_tensor, torch.Tensor)
        assert isinstance(dct_feat, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert img_tensor.shape[0] == 3  # RGB channels
        assert dct_feat.shape == torch.Size([1024])
        assert label.dtype == torch.long
    
    def test_dataset_getitem_with_dct(self, temp_dataset_dir, temp_dct_dir):
        """Test __getitem__ loads DCT features correctly"""
        dataset = HybridDataset(temp_dataset_dir, dct_dir=temp_dct_dir)
        
        img_tensor, dct_feat, label = dataset[0]
        
        assert dct_feat.shape == torch.Size([1024])
        # Should not be all zeros since we loaded actual features
        assert not torch.all(dct_feat == 0)
    
    def test_dataset_with_transform(self, temp_dataset_dir):
        """Test dataset applies transforms correctly"""
        transform = get_transforms(is_training=False)
        dataset = HybridDataset(temp_dataset_dir, transform=transform)
        
        img_tensor, _, _ = dataset[0]
        
        # After transform, image should be normalized and 224x224
        assert img_tensor.shape == torch.Size([3, 224, 224])
    
    def test_empty_dataset_raises_error(self):
        """Test that empty directory raises ValueError"""
        with tempfile.TemporaryDirectory() as temp_dir:
            real_dir = Path(temp_dir) / "real"
            fake_dir = Path(temp_dir) / "fake"
            real_dir.mkdir()
            fake_dir.mkdir()
            
            with pytest.raises(ValueError, match="No images found"):
                HybridDataset(temp_dir)


class TestGetTransforms:
    """Test cases for get_transforms function"""
    
    def test_training_transforms(self):
        """Test training transforms include augmentations"""
        transform = get_transforms(is_training=True)
        
        # Create dummy image
        img = Image.new('RGB', (256, 256), color=(128, 128, 128))
        
        # Apply transform
        result = transform(img)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([3, 224, 224])
    
    def test_validation_transforms(self):
        """Test validation transforms have no augmentation"""
        transform = get_transforms(is_training=False)
        
        # Create dummy image
        img = Image.new('RGB', (256, 256), color=(128, 128, 128))
        
        # Apply transform
        result = transform(img)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([3, 224, 224])
    
    def test_transforms_normalize(self):
        """Test that transforms apply normalization"""
        transform = get_transforms(is_training=False)
        
        img = Image.new('RGB', (256, 256), color=(255, 255, 255))
        result = transform(img)
        
        # Normalized values should not be in [0, 1] range
        assert result.max() > 1.0 or result.min() < 0.0


class TestCreateDataloaders:
    """Test cases for create_dataloaders function"""
    
    def test_dataloader_creation(self, temp_dataset_dir):
        """Test dataloaders are created with correct split"""
        train_loader, val_loader = create_dataloaders(
            temp_dataset_dir,
            batch_size=2,
            train_ratio=0.8,
            seed=42
        )
        
        assert train_loader is not None
        assert val_loader is not None
    
    def test_dataloader_split_ratio(self, temp_dataset_dir):
        """Test train/val split ratio is correct"""
        train_loader, val_loader = create_dataloaders(
            temp_dataset_dir,
            batch_size=2,
            train_ratio=0.8,
            seed=42
        )
        
        # Total should be 10 images (5 real + 5 fake)
        # With drop_last=True on train, we might lose some samples
        train_samples = len(train_loader.dataset)
        val_samples = len(val_loader.dataset)
        
        assert train_samples == 8  # 80% of 10
        assert val_samples == 2    # 20% of 10
    
    def test_dataloader_batch_iteration(self, temp_dataset_dir):
        """Test iterating through dataloader batches"""
        train_loader, _ = create_dataloaders(
            temp_dataset_dir,
            batch_size=2,
            seed=42
        )
        
        batch = next(iter(train_loader))
        img_tensor, dct_feat, labels = batch
        
        assert img_tensor.shape[0] == 2  # batch_size
        assert dct_feat.shape[0] == 2
        assert labels.shape[0] == 2
    
    def test_dataloader_with_dct(self, temp_dataset_dir, temp_dct_dir):
        """Test dataloaders work with DCT features"""
        train_loader, val_loader = create_dataloaders(
            temp_dataset_dir,
            dct_dir=temp_dct_dir,
            batch_size=2,
            seed=42
        )
        
        batch = next(iter(train_loader))
        _, dct_feat, _ = batch
        
        # Should load actual DCT features, not zeros
        assert not torch.all(dct_feat == 0)
    
    def test_dataloader_reproducibility(self, temp_dataset_dir):
        """Test same seed produces same split"""
        train_loader1, val_loader1 = create_dataloaders(
            temp_dataset_dir,
            batch_size=2,
            seed=42
        )
        
        train_loader2, val_loader2 = create_dataloaders(
            temp_dataset_dir,
            batch_size=2,
            seed=42
        )
        
        # Check that the split is the same (same number of samples)
        assert len(train_loader1.dataset) == len(train_loader2.dataset)
        assert len(val_loader1.dataset) == len(val_loader2.dataset)
        
        # Check that validation datasets have same samples (no shuffle)
        val_labels1 = [val_loader1.dataset.labels[i] for i in range(len(val_loader1.dataset))]
        val_labels2 = [val_loader2.dataset.labels[i] for i in range(len(val_loader2.dataset))]
        assert val_labels1 == val_labels2


class TestDatasetEdgeCases:
    """Test edge cases and error handling"""
    
    def test_missing_dct_file_fallback(self, temp_dataset_dir):
        """Test dataset handles missing DCT files gracefully"""
        with tempfile.TemporaryDirectory() as dct_dir:
            dataset = HybridDataset(temp_dataset_dir, dct_dir=dct_dir)
            
            _, dct_feat, _ = dataset[0]
            
            # Should return zero tensor as fallback
            assert torch.all(dct_feat == 0)
    
    def test_dataset_length(self, temp_dataset_dir):
        """Test __len__ returns correct count"""
        dataset = HybridDataset(temp_dataset_dir)
        
        assert len(dataset) == 10
        assert len(dataset) == len(dataset.samples)
        assert len(dataset) == len(dataset.labels)
