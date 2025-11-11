#!/usr/bin/env python3
"""
Extract specific categories from ImagiNet dataset
Supports extracting multiple dataset categories like anime, faces, landscapes, etc.
"""

import subprocess
import os
from pathlib import Path
import sys


def check_archive_exists(data_dir):
    """Check if archive files exist"""
    data_dir = Path(data_dir)
    parts = sorted(data_dir.glob("imaginet.7z.*"))
    
    if not parts:
        raise FileNotFoundError(
            f"❌ Archive files not found in {data_dir}\n"
            f"Expected: imaginet.7z.001, imaginet.7z.002, etc."
        )
    
    return parts


def get_archive_structure(archive_path, max_depth=2):
    """Get folder structure from archive"""
    print(f"📊 Analyzing archive structure...")
    
    cmd = f"7z l -slt '{archive_path}' | grep '^Path = ' | sed 's/^Path = //' | grep '/' | cut -d'/' -f1-{max_depth} | sort -u"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    folders = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
    return folders


def count_files_in_category(archive_path, category):
    """Count files for a specific category"""
    cmd = f"7z l -slt '{archive_path}' | grep '^Path = ' | grep -i '{category}/' | wc -l"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return int(result.stdout.strip())


def extract_categories(
    data_dir="/media/fadhly/files/imaginet/data",
    output_dir="/media/fadhly/files/imaginet/processed",
    categories=None,
    dry_run=False
):
    """
    Extract specific categories from ImagiNet archive
    
    Parameters:
    -----------
    data_dir : str
        Directory containing imaginet.7z.* files
    output_dir : str
        Output directory for extracted files
    categories : list or tuple
        Categories to extract. Available categories:
        - 'danbooru2021': Anime artwork (~10-20GB)
        - 'ffhq': Face images (~10-15GB)
        - 'ffhq_stylegan': StyleGAN generated faces
        - 'animaginexl_paintings_fake': AI-generated paintings/landscapes
        - 'dalle3': DALL-E 3 generated images
        - 'imagenet': ImageNet real images (LARGE!)
    dry_run : bool
        If True, only show what would be extracted without actually extracting
    
    Returns:
    --------
    dict : Extraction results
    """
    
    # Default categories if none specified
    if categories is None:
        categories = ('danbooru2021', 'ffhq', 'animaginexl_paintings_fake')
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    print("\n" + "=" * 70)
    print("🚀 IMAGINET CATEGORY EXTRACTOR")
    print("=" * 70)
    
    # Check archive exists
    parts = check_archive_exists(data_dir)
    print(f"\n✅ Found {len(parts)} archive parts")
    
    first_part = parts[0]
    
    # Show categories
    print(f"\n📋 Categories to extract:")
    for cat in categories:
        print(f"   • {cat}")
    
    # Dry run: estimate sizes
    if dry_run:
        print(f"\n🔍 DRY RUN MODE - Counting files...")
        print("=" * 70)
        
        total_files = 0
        for cat in categories:
            count = count_files_in_category(first_part, cat)
            total_files += count
            print(f"   {cat:30s}: {count:>10,} files")
        
        print("=" * 70)
        print(f"   TOTAL ESTIMATED FILES: {total_files:,}")
        print("\n💡 Run with dry_run=False to actually extract")
        
        return {
            'dry_run': True,
            'categories': categories,
            'estimated_files': total_files
        }
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create file lists
    cat_name = "_".join([c.replace('/', '_') for c in categories])
    filelist = data_dir / f"filelist_{cat_name}.txt"
    selected = data_dir / f"selected_{cat_name}.txt"
    
    # Step 1: List all files in archive
    print(f"\n📄 Step 1/3: Listing archive contents...")
    print("⏳ This may take several minutes...")
    
    list_cmd = f"7z l -slt '{first_part}' | grep '^Path = ' | sed 's/^Path = //' > '{filelist}'"
    result = subprocess.run(list_cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to list archive: {result.stderr}")
    
    with open(filelist, 'r') as f:
        total_in_archive = sum(1 for _ in f)
    print(f"✅ Found {total_in_archive:,} total files in archive")
    
    # Step 2: Filter by categories
    print(f"\n🔍 Step 2/3: Filtering categories...")
    
    # Build grep pattern for folders (more precise)
    # Match lines that start with category name followed by /
    grep_patterns = []
    for cat in categories:
        grep_patterns.append(f"^{cat}/")
    
    grep_pattern = "|".join(grep_patterns)
    
    filter_cmd = f"grep -E '({grep_pattern})' '{filelist}' > '{selected}' || touch '{selected}'"
    result = subprocess.run(filter_cmd, shell=True, capture_output=True, text=True)
    
    # Check selection
    with open(selected, 'r') as f:
        selected_count = sum(1 for _ in f)
    
    if selected_count == 0:
        print(f"\n⚠️  WARNING: No files found for categories: {', '.join(categories)}")
        print(f"\n💡 Available categories in archive:")
        
        # Show available categories
        struct = get_archive_structure(first_part, max_depth=1)
        for folder in struct[:20]:
            print(f"   • {folder}")
        
        raise ValueError(f"No files matched the specified categories")
    
    print(f"✅ Selected {selected_count:,} files to extract")
    
    # Estimate size
    avg_file_size = 500 * 1024  # Assume ~500KB per file (conservative)
    estimated_size_gb = (selected_count * avg_file_size) / (1024**3)
    print(f"📊 Estimated extraction size: ~{estimated_size_gb:.1f} GB")
    
    # Step 3: Extract files
    print(f"\n📦 Step 3/3: Extracting files...")
    print(f"⏳ This will take a while (10-30+ minutes depending on size)")
    print(f"📂 Output: {output_dir}\n")
    
    extract_cmd = [
        "7z", "x", str(first_part),
        f"-o{output_dir}",
        f"-i@{selected}",
        "-y"  # Yes to all prompts
    ]
    
    # Run extraction with live output
    process = subprocess.Popen(
        extract_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Show progress
    for line in process.stdout:
        if line.strip():
            # Show extraction progress
            if "Extracting" in line or "%" in line:
                print(f"  {line.strip()}")
    
    process.wait()
    
    if process.returncode != 0:
        raise RuntimeError("Extraction failed!")
    
    # Step 4: Verify extraction
    print(f"\n🧾 Verifying extraction...")
    
    extracted_log = output_dir / f"extracted_{cat_name}.txt"
    verify_cmd = f"find '{output_dir}' -type f > '{extracted_log}'"
    subprocess.run(verify_cmd, shell=True)
    
    with open(extracted_log, 'r') as f:
        extracted_count = sum(1 for _ in f)
    
    # Get actual size
    size_cmd = f"du -sh '{output_dir}'"
    size_result = subprocess.run(size_cmd, shell=True, capture_output=True, text=True)
    actual_size = size_result.stdout.split()[0] if size_result.stdout else "unknown"
    
    print("\n" + "=" * 70)
    print("✅ EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"Categories extracted: {', '.join(categories)}")
    print(f"Total files extracted: {extracted_count:,}")
    print(f"Actual size: {actual_size}")
    print(f"Output directory: {output_dir}")
    print(f"Log file: {extracted_log}")
    print("=" * 70)
    
    return {
        'success': True,
        'categories': categories,
        'output_dir': str(output_dir),
        'total_files': extracted_count,
        'size': actual_size,
        'log_file': str(extracted_log)
    }


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract categories from ImagiNet dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available categories:
  danbooru2021              - Anime artwork (~10-20GB)
  ffhq                      - Face images (~10-15GB)
  ffhq_stylegan             - StyleGAN generated faces
  animaginexl_paintings_fake - AI-generated paintings/landscapes
  dalle3                    - DALL-E 3 generated images
  imagenet                  - ImageNet real images (VERY LARGE!)

Examples:
  # Extract anime and faces (dry run first)
  python extract_imaginet_categories.py --categories danbooru2021 ffhq --dry-run
  
  # Actually extract
  python extract_imaginet_categories.py --categories danbooru2021 ffhq
  
  # Extract to custom directory
  python extract_imaginet_categories.py --categories ffhq --output /path/to/output
        """
    )
    
    parser.add_argument(
        '--data-dir',
        default='/media/fadhly/files/imaginet/data',
        help='Directory containing imaginet.7z.* files'
    )
    parser.add_argument(
        '--output',
        default='/media/fadhly/files/imaginet/processed',
        help='Output directory for extracted files'
    )
    parser.add_argument(
        '--categories',
        nargs='+',
        default=['danbooru2021', 'ffhq', 'animaginexl_paintings_fake'],
        help='Categories to extract (space-separated)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be extracted without actually extracting'
    )
    
    args = parser.parse_args()
    
    try:
        result = extract_categories(
            data_dir=args.data_dir,
            output_dir=args.output,
            categories=args.categories,
            dry_run=args.dry_run
        )
        
        if result.get('success'):
            sys.exit(0)
        elif result.get('dry_run'):
            print("\n💡 Add flag to actually extract: remove --dry-run")
            sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
