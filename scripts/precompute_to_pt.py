import os
import torch
import yaml 
import json
from pathlib import Path
from tqdm import tqdm
import sys
from datetime import datetime

# Get absolute paths based on script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pipeline import preprocess_full

CONFIG_PATH = PROJECT_ROOT / "configs" / "preprocessing.yaml"
PROGRESS_FILE = PROJECT_ROOT / "data" / "processed" / "imaginet" / "progress.json"

# Auto-detect data location (try multiple possible paths)
possible_fake_paths = [
    PROJECT_ROOT / "data" / "processed" / "imaginet" / "subset" / "fake",
    PROJECT_ROOT / "data" / "raw" / "imaginet" / "subset" / "fake",
    PROJECT_ROOT / "data" / "processed" / "imagenet" / "subset" / "fake",
]

possible_real_paths = [
    PROJECT_ROOT / "data" / "processed" / "imaginet" / "subset" / "real",
    PROJECT_ROOT / "data" / "raw" / "imaginet" / "subset" / "real",
    PROJECT_ROOT / "data" / "processed" / "imagenet" / "subset" / "real",
]

# Find first existing path or use default
RAW_FAKE_DIR = next((p for p in possible_fake_paths if p.exists()), possible_fake_paths[0])
RAW_REAL_DIR = next((p for p in possible_real_paths if p.exists()), possible_real_paths[0])

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "imaginet" / "subset_pt"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
    
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "fake").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "real").mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_progress():
    """Load progress dari file JSON"""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not load progress file: {e}")
            return {"processed": [], "failed": [], "last_update": None}
    return {"processed": [], "failed": [], "last_update": None}

def save_progress(progress):
    """Save progress ke file JSON"""
    try:
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        progress["last_update"] = datetime.now().isoformat()
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        print(f"⚠️  Warning: Could not save progress: {e}")

def export_folder(src_dir: Path, dst_dir: Path, label: int, progress: dict):
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    img_files = list(src_path.rglob("*.jpg")) + list(src_path.rglob("*.png"))
    print(f"📁 Found {len(img_files)} images in {src_dir}")
    
    if len(img_files) == 0:
        print(f"⚠️  Warning: No images found in {src_dir}")
        return
    
    # Filter out already processed files
    processed_set = set(progress.get("processed", []))
    failed_set = set(progress.get("failed", []))
    
    # Skip already processed files
    img_files_to_process = [
        img for img in img_files 
        if str(img) not in processed_set
    ]
    
    already_processed = len(img_files) - len(img_files_to_process)
    if already_processed > 0:
        print(f"⏭️  Skipping {already_processed} already processed files")
    
    if len(img_files_to_process) == 0:
        print(f"✅ All files already processed in {src_path.name}")
        return
    
    print(f"🔄 Processing {len(img_files_to_process)} remaining files")
    
    processed_count = 0
    failed_count = 0
    
    for img_path in tqdm(img_files_to_process, desc=f"Processing {src_path.name}"):
        try:
            # Check if output file already exists
            stem = img_path.stem
            save_path = dst_path / f"{stem}.pt"
            
            if save_path.exists() and str(img_path) not in failed_set:
                # File exists and wasn't previously failed, skip
                if str(img_path) not in processed_set:
                    progress["processed"].append(str(img_path))
                    processed_count += 1
                continue
            
            result = preprocess_full(str(img_path), config)
            
            save_data = {
                'img_masked': result['img_masked'].cpu(),
                'dct_feat': result['dct_feat'].cpu(),
                'label': label,  
                'original_path': str(img_path),
                'jpeg_quality': result['intermediates'].get('jpeg_quality', None)
            }
            
            torch.save(save_data, save_path)
            
            # Mark as processed
            progress["processed"].append(str(img_path))
            processed_count += 1
            
            # Remove from failed list if it was there
            if str(img_path) in failed_set:
                progress["failed"].remove(str(img_path))
            
            # Save progress every 10 files
            if processed_count % 10 == 0:
                save_progress(progress)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  KeyboardInterrupt detected! Saving progress...")
            save_progress(progress)
            print(f"✅ Progress saved. Processed {processed_count} files in this run.")
            print(f"📊 Total processed: {len(progress['processed'])} files")
            print(f"❌ Total failed: {len(progress['failed'])} files")
            print("\n💡 Run the script again to resume from where you left off.")
            sys.exit(0)
            
        except Exception as e:
            print(f"\n❌ Error processing {img_path.name}: {e}")
            if str(img_path) not in failed_set:
                progress["failed"].append(str(img_path))
            failed_count += 1
    
    # Final progress save
    save_progress(progress)
    
    print(f"\n📊 Summary for {src_path.name}:")
    print(f"   ✅ Processed: {processed_count} files")
    print(f"   ❌ Failed: {failed_count} files")
            
if __name__ == "__main__":
    print("🚀 Starting preprocessing to .pt format...")
    print(f"📋 Config: {CONFIG_PATH}")
    print(f"💾 Output: {OUTPUT_DIR}")
    print(f"🖥️  Device: {device}")
    
    # Load previous progress
    progress = load_progress()
    
    if progress["processed"] or progress["failed"]:
        print(f"\n📊 Resuming from previous run:")
        print(f"   ✅ Already processed: {len(progress['processed'])} files")
        print(f"   ❌ Previously failed: {len(progress['failed'])} files")
        if progress.get("last_update"):
            print(f"   🕐 Last update: {progress['last_update']}")
    else:
        print(f"\n🆕 Starting fresh preprocessing")
    
    print()
    
    # Check if source directories exist
    if not RAW_FAKE_DIR.exists():
        print(f"⚠️  Warning: Fake directory not found: {RAW_FAKE_DIR}")
        print(f"   Using: {RAW_FAKE_DIR}")
    
    if not RAW_REAL_DIR.exists():
        print(f"⚠️  Warning: Real directory not found: {RAW_REAL_DIR}")
        print(f"   Using: {RAW_REAL_DIR}")
    
    try:
        # Export fake images (label=1)
        print("\n" + "="*60)
        print("📸 Processing FAKE images...")
        print("="*60)
        export_folder(RAW_FAKE_DIR, OUTPUT_DIR / "fake", label=1, progress=progress)
        
        # Export real images (label=0)
        print("\n" + "="*60)
        print("📸 Processing REAL images...")
        print("="*60)
        export_folder(RAW_REAL_DIR, OUTPUT_DIR / "real", label=0, progress=progress)
        
        print("\n" + "="*60)
        print("✅ Export selesai! File .pt siap untuk training.")
        print(f"📁 Lokasi output:")
        print(f"   - Fake: {OUTPUT_DIR / 'fake'}")
        print(f"   - Real: {OUTPUT_DIR / 'real'}")
        print(f"\n📊 Final Statistics:")
        print(f"   ✅ Total processed: {len(progress['processed'])} files")
        print(f"   ❌ Total failed: {len(progress['failed'])} files")
        print("="*60)
        
        # Optionally clean up progress file after successful completion
        # if len(progress['failed']) == 0:
        #     print("\n🗑️  Cleaning up progress file...")
        #     PROGRESS_FILE.unlink(missing_ok=True)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user!")
        save_progress(progress)
        print(f"✅ Progress saved to: {PROGRESS_FILE}")
        print("💡 Run the script again to resume from where you left off.")
        sys.exit(0)