#!/usr/bin/env python3
"""
Precompute DCT Features for all images
Saves DCT features as .npy files for faster training
Uses multi-processing for faster computation on CPU
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dct import extract_dct_features


def process_single_image(img_path, output_dir, top_k, block_size, overwrite):
    """
    Process a single image and save DCT features.
    This function is used for parallel processing.
    
    Returns:
        tuple: (status, message) where status is 'processed', 'skipped', or 'error'
    """
    try:
        # Output path
        output_path = output_dir / f"{img_path.stem}.npy"
        
        # Skip if already exists and not overwriting
        if output_path.exists() and not overwrite:
            return ('skipped', None)
        
        # Load image and convert to grayscale
        img = Image.open(img_path).convert('L')
        img_np = np.array(img)
        
        # Extract DCT features
        dct_features = extract_dct_features(
            img_np, 
            top_k=top_k, 
            block_size=block_size
        )
        
        # Save as .npy
        np.save(output_path, dct_features)
        return ('processed', None)
        
    except Exception as e:
        return ('error', f"{img_path.name}: {str(e)}")


def precompute_dct_for_dataset(
    image_dir: Path, 
    output_dir: Path,
    top_k: int = 1024,
    block_size: int = 8,
    overwrite: bool = False,
    num_workers: int = None
):
    """
    Precompute DCT features for all images in a directory.
    Uses multi-processing for faster computation.
    
    Args:
        image_dir: Directory containing images (with 'real' and 'fake' subfolders)
        output_dir: Directory to save .npy files
        top_k: Number of top DCT coefficients to extract
        block_size: DCT block size
        overwrite: If True, recompute even if .npy exists
        num_workers: Number of parallel workers (default: CPU count - 1)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave 1 core free
    
    # Collect all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(image_dir.rglob(f'*{ext}'))
        image_files.extend(image_dir.rglob(f'*{ext.upper()}'))
    
    if len(image_files) == 0:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"📊 PRECOMPUTING DCT FEATURES (CPU MULTI-PROCESSING)")
    print(f"{'='*60}")
    print(f"Image directory: {image_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total images: {len(image_files)}")
    print(f"DCT parameters: top_k={top_k}, block_size={block_size}")
    print(f"Overwrite existing: {overwrite}")
    print(f"CPU workers: {num_workers}/{cpu_count()}")
    print()
    
    skipped = 0
    processed = 0
    errors = 0
    error_messages = []
    
    # Create partial function with fixed parameters
    process_fn = partial(
        process_single_image,
        output_dir=output_dir,
        top_k=top_k,
        block_size=block_size,
        overwrite=overwrite
    )
    
    # Process images in parallel
    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_fn, image_files),
                total=len(image_files),
                desc="Processing images",
                ncols=80
            ))
    else:
        # Single process mode
        results = [process_fn(img) for img in tqdm(
            image_files, 
            desc="Processing images", 
            ncols=80
        )]
    
    # Count results
    for status, msg in results:
        if status == 'processed':
            processed += 1
        elif status == 'skipped':
            skipped += 1
        elif status == 'error':
            errors += 1
            if msg:
                error_messages.append(msg)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ PRECOMPUTATION COMPLETE")
    print(f"{'='*60}")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    print(f"Total .npy files: {len(list(output_dir.glob('*.npy')))}")
    
    if error_messages:
        print(f"\n⚠️  Error details (showing first 5):")
        for msg in error_messages[:5]:
            print(f"   {msg}")
    
    print(f"\nOutput saved to: {output_dir}")
    print()


def verify_dct_files(image_dir: Path, dct_dir: Path):
    """
    Verify that all images have corresponding DCT files.
    """
    # Collect all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(image_dir.rglob(f'*{ext}'))
        image_files.extend(image_dir.rglob(f'*{ext.upper()}'))
    
    missing = []
    for img_path in image_files:
        dct_path = dct_dir / f"{img_path.stem}.npy"
        if not dct_path.exists():
            missing.append(img_path.name)
    
    if missing:
        print(f"\n⚠️  Missing DCT files for {len(missing)} images:")
        for name in missing[:10]:
            print(f"   - {name}")
        if len(missing) > 10:
            print(f"   ... and {len(missing) - 10} more")
    else:
        print(f"\n✅ All {len(image_files)} images have DCT features!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Precompute DCT features for training dataset'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default=str(PROJECT_ROOT / 'data' / 'processed' / 'imaginet' / 'subset'),
        help='Directory containing images (default: data/processed/imaginet/subset)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(PROJECT_ROOT / 'data' / 'processed' / 'imaginet' / 'dct_features'),
        help='Output directory for .npy files (default: data/processed/imaginet/dct_features)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=1024,
        help='Number of top DCT coefficients (default: 1024)'
    )
    parser.add_argument(
        '--block-size',
        type=int,
        default=8,
        help='DCT block size (default: 8)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing .npy files'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify that all images have DCT files'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of CPU workers for parallel processing (default: CPU count - 1)'
    )
    
    args = parser.parse_args()
    
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    
    if not image_dir.exists():
        print(f"❌ Image directory not found: {image_dir}")
        sys.exit(1)
    
    # Precompute DCT features
    precompute_dct_for_dataset(
        image_dir=image_dir,
        output_dir=output_dir,
        top_k=args.top_k,
        block_size=args.block_size,
        overwrite=args.overwrite,
        num_workers=args.workers
    )
    
    # Verify if requested
    if args.verify:
        verify_dct_files(image_dir, output_dir)
