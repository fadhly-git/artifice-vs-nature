"""
Augmentation Module
Apply random augmentations to tensor images
"""

import torch
import random


def apply_augment(tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply random augmentations to image tensor
    
    Args:
        tensor: Input tensor (C, H, W) with values in [0, 1]
    
    Returns:
        Augmented tensor (C, H, W)
    
    Augmentations:
        - Random horizontal flip (p=0.5)
        - Random brightness adjustment (±0.2)
        - Random contrast adjustment (±0.2)
    
    Example:
        >>> import torch
        >>> tensor = torch.rand(3, 224, 224)
        >>> tensor_aug = apply_augment(tensor)
        >>> print(tensor_aug.shape)  # torch.Size([3, 224, 224])
    """
    # Random horizontal flip
    if random.random() > 0.5:
        tensor = torch.flip(tensor, dims=[2])
    
    # Random brightness
    brightness_factor = 1.0 + random.uniform(-0.2, 0.2)
    tensor = torch.clamp(tensor * brightness_factor, 0, 1)
    
    # Random contrast
    contrast_factor = 1.0 + random.uniform(-0.2, 0.2)
    mean = tensor.mean(dim=[1, 2], keepdim=True)
    tensor = torch.clamp((tensor - mean) * contrast_factor + mean, 0, 1)
    
    return tensor


if __name__ == "__main__":
    print("Augmentation module loaded successfully")
    print("Usage: apply_augment(tensor)")