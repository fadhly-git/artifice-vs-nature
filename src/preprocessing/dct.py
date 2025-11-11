"""
DCT Feature Extraction Module
Extract DCT (Discrete Cosine Transform) coefficients for frequency analysis
"""

import numpy as np
from scipy.fftpack import dct


def extract_dct_features(img_np: np.ndarray, top_k: int = 1024, block_size: int = 8) -> np.ndarray:
    """
    Extract top-k DCT coefficients from image
    
    Args:
        img_np: Input grayscale image as numpy array (H, W)
        top_k: Number of top coefficients to extract
        block_size: Size of DCT blocks (default 8x8)
    
    Returns:
        1D numpy array of top-k DCT coefficients sorted by magnitude
    
    Example:
        >>> import numpy as np
        >>> from PIL import Image
        >>> img = Image.open("sample.jpg").convert('L')
        >>> img_np = np.array(img)
        >>> dct_feat = extract_dct_features(img_np, top_k=1024)
        >>> print(dct_feat.shape)  # (1024,)
    """
    H, W = img_np.shape
    
    # Ensure dimensions are multiples of block_size
    H_pad = ((H + block_size - 1) // block_size) * block_size
    W_pad = ((W + block_size - 1) // block_size) * block_size
    
    img_padded = np.zeros((H_pad, W_pad))
    img_padded[:H, :W] = img_np
    
    # Extract DCT coefficients block by block
    dct_coeffs = []
    
    for i in range(0, H_pad, block_size):
        for j in range(0, W_pad, block_size):
            block = img_padded[i:i+block_size, j:j+block_size]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            dct_coeffs.extend(dct_block.flatten())
    
    # Get top-k coefficients by absolute magnitude
    dct_coeffs = np.array(dct_coeffs)
    top_indices = np.argsort(np.abs(dct_coeffs))[-top_k:]
    top_coeffs = dct_coeffs[top_indices]
    
    return top_coeffs


if __name__ == "__main__":
    print("DCT feature extraction module loaded successfully")
    print("Usage: extract_dct_features(img_np, top_k=1024, block_size=8)")