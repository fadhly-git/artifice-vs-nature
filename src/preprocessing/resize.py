"""
Resize Module
Resize images to standard 224x224 for model input
"""

from PIL import Image


def resize_to_224(img_pil: Image.Image) -> Image.Image:
    """
    Resize PIL Image to 224x224 using LANCZOS resampling
    
    Args:
        img_pil: Input PIL Image
    
    Returns:
        Resized PIL Image (224x224)
    
    Example:
        >>> from PIL import Image
        >>> img = Image.open("data/raw/imaginet/subset/anime/sample.jpg")
        >>> img_resized = resize_to_224(img)
        >>> print(img_resized.size)  # (224, 224)
    """
    return img_pil.resize((224, 224), Image.LANCZOS)


if __name__ == "__main__":
    print("Resize module loaded successfully")
    print("Usage: resize_to_224(img_pil)")