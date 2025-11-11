#!/usr/bin/env python3
"""
Script untuk check VRAM usage
"""

import torch
import subprocess

print("="*60)
print("🖥️  GPU MEMORY CHECK")
print("="*60)

if torch.cuda.is_available():
    print(f"\n✅ CUDA Available: True")
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Get current memory usage
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    
    print(f"\n📊 Current Usage:")
    print(f"   Allocated: {allocated:.2f} GB")
    print(f"   Reserved: {reserved:.2f} GB")
    print(f"   Free: {(torch.cuda.get_device_properties(0).total_memory / 1e9) - reserved:.2f} GB")
    
    # Try rocm-smi
    print("\n🔧 ROCm SMI Output:")
    try:
        result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
        print(result.stdout)
    except FileNotFoundError:
        print("   ⚠️  rocm-smi not found")
        
else:
    print("\n❌ CUDA not available")

print("="*60)

# Memory optimization tips
print("\n💡 Memory Optimization Tips:")
print("1. Reduce batch size: --batch-size 2 or 1")
print("2. Increase gradient accumulation: --accum-steps 16 or 32")
print("3. Disable AMP if causing issues: remove --amp flag")
print("4. Reduce num_workers: --num-workers 0")
print("5. Use smaller model: Change efficientnet_b4 to efficientnet_b0")
print("="*60)
