"""
Masking Module
Apply random masking to image tensors for robust feature learning
"""

import torch
import random


def apply_mask(tensor: torch.Tensor, ratio: float = 0.2) -> torch.Tensor:
    """
    Apply random rectangular masking to image tensor
    
    Args:
        tensor: Input tensor (C, H, W)
        ratio: Ratio of image area to mask (0-1)
    
    Returns:
        Masked tensor (C, H, W) with masked regions set to 0
    
    Example:
        >>> import torch
        >>> tensor = torch.rand(3, 224, 224)
        >>> tensor_masked = apply_mask(tensor, ratio=0.2)
        >>> print(tensor_masked.shape)  # torch.Size([3, 224, 224])
    """
    C, H, W = tensor.shape
    
    # Calculate mask dimensions
    mask_area = H * W * ratio
    mask_h = int((mask_area * random.uniform(0.5, 2.0)) ** 0.5)
    mask_w = int(mask_area / mask_h)
    
    mask_h = min(mask_h, H)
    mask_w = min(mask_w, W)
    
    # Random position
    top = random.randint(0, H - mask_h)
    left = random.randint(0, W - mask_w)
    
    # Apply mask
    tensor_masked = tensor.clone()
    tensor_masked[:, top:top+mask_h, left:left+mask_w] = 0
    
    return tensor_masked


if __name__ == "__main__":
    print("Masking module loaded successfully")
    print("Usage: apply_mask(tensor, ratio=0.2)")