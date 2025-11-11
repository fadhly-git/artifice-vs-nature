"""
JPEG Compression Module
Applies JPEG compression with random quality for data augmentation
"""

import io
import random
from PIL import Image
from typing import Tuple


def apply_jpeg(img_pil: Image.Image, quality_range: Tuple[int, int] = (30, 90), prob: float = 0.7) -> Tuple[Image.Image, int]:
    """
    Apply JPEG compression to PIL Image with random quality
    
    Args:
        img_pil: Input PIL Image
        quality_range: (min_quality, max_quality) for JPEG compression
        prob: Probability of applying JPEG compression
    
    Returns:
        Tuple of (compressed PIL Image, quality used)
    
    Example:
        >>> from PIL import Image
        >>> img = Image.open("data/raw/imaginet/subset/anime/sample.jpg")
        >>> img_compressed, quality = apply_jpeg(img, quality_range=(30, 90), prob=0.7)
        >>> print(f"Applied JPEG quality: {quality}")
    """
    if random.random() > prob:
        return img_pil, 100  # No compression
    
    quality = random.randint(quality_range[0], quality_range[1])
    
    # Convert to RGB if needed
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    
    # Apply JPEG compression
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    img_compressed = Image.open(buffer)
    
    return img_compressed, quality


if __name__ == "__main__":
    # Test example
    print("JPEG compression module loaded successfully")
    print("Usage: apply_jpeg(img_pil, quality_range=(30,90), prob=0.7)")