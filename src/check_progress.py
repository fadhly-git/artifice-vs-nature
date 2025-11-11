#!/usr/bin/env python3
"""
Script untuk melihat progress preprocessing
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Get project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

PROGRESS_FILE = PROJECT_ROOT / "data" / "processed" / "imaginet" / "progress.json"

def load_progress():
    """Load progress dari file JSON"""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading progress file: {e}")
            return None
    return None

def show_progress():
    """Display progress statistics"""
    progress = load_progress()
    
    if progress is None:
        print("📊 No progress file found.")
        print(f"   Expected location: {PROGRESS_FILE}")
        print("\n💡 Run `python src/precompute_to_pt.py` to start preprocessing.")
        return
    
    processed = progress.get("processed", [])
    failed = progress.get("failed", [])
    last_update = progress.get("last_update", "Unknown")
    
    print("="*60)
    print("📊 PREPROCESSING PROGRESS")
    print("="*60)
    print(f"\n✅ Processed: {len(processed)} files")
    print(f"❌ Failed: {len(failed)} files")
    print(f"📁 Total: {len(processed) + len(failed)} files")
    
    if last_update != "Unknown":
        try:
            dt = datetime.fromisoformat(last_update)
            print(f"🕐 Last update: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        except:
            print(f"🕐 Last update: {last_update}")
    
    print("\n" + "="*60)
    
    if failed:
        print(f"\n❌ Failed files ({len(failed)}):")
        for i, path in enumerate(failed[:10], 1):
            print(f"   {i}. {Path(path).name}")
        if len(failed) > 10:
            print(f"   ... and {len(failed) - 10} more")
    
    print("\n💡 Commands:")
    print("   - Resume: python src/precompute_to_pt.py")
    print("   - Reset:  python src/check_progress.py --reset")
    print("   - Retry failed: python src/check_progress.py --retry-failed")

def reset_progress():
    """Reset progress file"""
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print("✅ Progress file deleted.")
        print(f"   Location: {PROGRESS_FILE}")
    else:
        print("⚠️  No progress file to delete.")

def retry_failed():
    """Remove failed files from progress so they can be retried"""
    progress = load_progress()
    
    if progress is None:
        print("❌ No progress file found.")
        return
    
    failed_count = len(progress.get("failed", []))
    
    if failed_count == 0:
        print("✅ No failed files to retry.")
        return
    
    print(f"🔄 Removing {failed_count} failed files from progress...")
    progress["failed"] = []
    
    try:
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress, f, indent=2)
        print(f"✅ {failed_count} failed files will be retried on next run.")
    except Exception as e:
        print(f"❌ Error saving progress: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--reset":
            reset_progress()
        elif sys.argv[1] == "--retry-failed":
            retry_failed()
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python src/check_progress.py           # Show progress")
            print("  python src/check_progress.py --reset   # Reset progress")
            print("  python src/check_progress.py --retry-failed  # Retry failed files")
        else:
            print(f"❌ Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        show_progress()
