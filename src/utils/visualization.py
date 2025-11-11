"""
Visualization utilities for preprocessing pipeline
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any


def plot_preprocessing_pipeline(intermediates: Dict[str, Any], figsize=(14, 10)):
    """
    Visualize preprocessing pipeline with 4 key stages
    
    Args:
        intermediates: Dictionary containing intermediate results from pipeline
        figsize: Figure size tuple
    
    Example:
        >>> from src.preprocessing.pipeline import preprocess_full
        >>> result = preprocess_full("sample.jpg")
        >>> plot_preprocessing_pipeline(result['intermediates'])
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Original image
    axs[0, 0].imshow(intermediates['original'])
    axs[0, 0].set_title(f"Original ({intermediates['original_size'][0]}×{intermediates['original_size'][1]})")
    axs[0, 0].axis('off')
    
    # 2. After JPEG compression
    axs[0, 1].imshow(intermediates['jpeg'])
    axs[0, 1].set_title(f"After JPEG (Q={intermediates['jpeg_quality']})")
    axs[0, 1].axis('off')
    
    # 3. Masked image (denormalize for visualization)
    masked_tensor = intermediates['masked']
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    masked_denorm = masked_tensor * std + mean
    masked_denorm = torch.clamp(masked_denorm, 0, 1)
    masked_img = masked_denorm.permute(1, 2, 0).numpy()
    
    axs[1, 0].imshow(masked_img)
    axs[1, 0].set_title("Masked 20% (Normalized)")
    axs[1, 0].axis('off')
    
    # 4. Top 100 DCT coefficients
    dct_values = intermediates['dct_values']
    top_100 = np.abs(dct_values[:100])
    
    axs[1, 1].bar(range(len(top_100)), top_100, color='steelblue')
    axs[1, 1].set_title(f"Top 100 DCT Coefficients (from {len(dct_values)} total)")
    axs[1, 1].set_xlabel("Coefficient Index")
    axs[1, 1].set_ylabel("Magnitude")
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Visualization utilities loaded successfully")
    print("Usage: plot_preprocessing_pipeline(intermediates)")