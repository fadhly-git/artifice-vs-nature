"""
Full Preprocessing Pipeline
Orchestrates all preprocessing steps in correct order
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Any

from .jpeg import apply_jpeg
from .resize import resize_to_224
from .augment import apply_augment
from .normalize import normalize_image
from .mask import apply_mask
from .dct import extract_dct_features


def preprocess_full(img_path: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Full preprocessing pipeline: JPEG → Resize → Augment → Normalize → Mask → DCT
    
    Args:
        img_path: Path to input image
        config: Configuration dictionary with preprocessing parameters
    
    Returns:
        Dictionary containing:
            - img_masked: torch.FloatTensor (3, 224, 224) - final masked image
            - dct_feat: torch.FloatTensor (top_k,) - DCT features
            - intermediates: Dict with intermediate results for visualization
    
    Example:
        >>> config = {
        ...     'jpeg': {'prob': 0.7, 'quality_range': [30, 90]},
        ...     'mask': {'ratio': 0.2},
        ...     'dct': {'top_k': 1024, 'block_size': 8}
        ... }
        >>> result = preprocess_full("data/raw/imaginet/subset/anime/sample.jpg", config)
        >>> print(result['img_masked'].shape)  # torch.Size([3, 224, 224])
        >>> print(result['dct_feat'].shape)    # torch.Size([1024])
    """
    # Default config
    if config is None:
        config = {
            'jpeg': {'prob': 0.7, 'quality_range': [30, 90]},
            'mask': {'ratio': 0.2},
            'dct': {'top_k': 1024, 'block_size': 8}
        }
    
    intermediates = {}
    
    # 1. Load image
    img_original = Image.open(img_path).convert('RGB')
    intermediates['original'] = img_original.copy()
    intermediates['original_size'] = img_original.size
    
    # 2. JPEG compression
    img_jpeg, jpeg_quality = apply_jpeg(
        img_original,
        quality_range=tuple(config['jpeg']['quality_range']),
        prob=config['jpeg']['prob']
    )
    intermediates['jpeg'] = img_jpeg.copy()
    intermediates['jpeg_quality'] = jpeg_quality
    
    # 3. Resize to 224x224
    img_resized = resize_to_224(img_jpeg)
    intermediates['resized'] = img_resized.copy()
    
    # 4. Convert to tensor [0, 1]
    img_tensor = torch.from_numpy(np.array(img_resized)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    
    # 5. Augmentation
    img_augmented = apply_augment(img_tensor)
    intermediates['augmented'] = img_augmented.clone()
    
    # 6. Normalization
    img_normalized = normalize_image(img_augmented)
    intermediates['normalized'] = img_normalized.clone()
    
    # 7. Masking
    img_masked = apply_mask(img_normalized, ratio=config['mask']['ratio'])
    intermediates['masked'] = img_masked.clone()
    
    # 8. DCT feature extraction (on grayscale version before normalization)
    img_gray = img_resized.convert('L')
    img_gray_np = np.array(img_gray)
    dct_features = extract_dct_features(
        img_gray_np,
        top_k=config['dct']['top_k'],
        block_size=config['dct']['block_size']
    )
    intermediates['dct_values'] = dct_features
    
    # Convert to torch tensor
    dct_feat_tensor = torch.from_numpy(dct_features).float()
    
    return {
        'img_masked': img_masked,
        'dct_feat': dct_feat_tensor,
        'intermediates': intermediates
    }


if __name__ == "__main__":
    print("Full preprocessing pipeline loaded successfully")
    print("Pipeline order: JPEG → Resize → Augment → Normalize → Mask → DCT")
    print("Usage: preprocess_full(img_path, config)")