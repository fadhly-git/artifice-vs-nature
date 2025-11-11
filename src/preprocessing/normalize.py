"""
Normalization Module
Normalize images using ImageNet statistics
"""

import torch


def normalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize image tensor using ImageNet mean and std
    
    Args:
        tensor: Input tensor (C, H, W) with values in [0, 1]
    
    Returns:
        Normalized tensor (C, H, W)
    
    Statistics:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    Example:
        >>> import torch
        >>> tensor = torch.rand(3, 224, 224)
        >>> tensor_norm = normalize_image(tensor)
        >>> print(tensor_norm.mean(), tensor_norm.std())
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    return (tensor - mean) / std


if __name__ == "__main__":
    print("Normalization module loaded successfully")
    print("Usage: normalize_image(tensor)")