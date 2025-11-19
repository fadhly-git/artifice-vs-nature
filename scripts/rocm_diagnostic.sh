#!/bin/bash
# AMD ROCm GPU Crash Workaround Script

echo "=================================================="
echo "🔧 AMD ROCm GPU Crash Diagnostic & Fix"
echo "=================================================="

echo ""
echo "1️⃣ Checking GPU status..."
if command -v rocm-smi &> /dev/null; then
    rocm-smi
else
    echo "⚠️  rocm-smi not found - ROCm may not be properly installed"
fi

echo ""
echo "2️⃣ Checking GPU temperature..."
if [ -f /sys/class/drm/card0/device/hwmon/hwmon*/temp1_input ]; then
    temp=$(cat /sys/class/drm/card0/device/hwmon/hwmon*/temp1_input 2>/dev/null | head -1)
    if [ -n "$temp" ]; then
        temp_c=$((temp / 1000))
        echo "GPU Temperature: ${temp_c}°C"
        if [ $temp_c -gt 80 ]; then
            echo "⚠️  GPU is HOT! This may cause crashes."
        fi
    fi
fi

echo ""
echo "3️⃣ Recommended fixes for ROCm crashes:"
echo ""
echo "Option A: Reduce model complexity"
echo "  - Use smaller batch size (1-2)"
echo "  - Disable mixed precision (no AMP)"
echo "  - Add gradient clipping"
echo ""
echo "Option B: Fix ROCm environment variables for RX 580 (GFX803)"
echo "  export HSA_OVERRIDE_GFX_VERSION=8.0.3"
echo "  export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128"
echo "  export HIP_VISIBLE_DEVICES=0"
echo "  export GPU_MAX_HEAP_SIZE=100"
echo "  export GPU_MAX_ALLOC_PERCENT=100"
echo ""
echo "Option C: Reset GPU"
echo "  sudo rmmod amdgpu"
echo "  sudo modprobe amdgpu"
echo ""
echo "Option D: Use CPU-only mode (safest)"
echo "  export CUDA_VISIBLE_DEVICES=''"
echo ""
echo "=================================================="
echo "💡 Immediate Action: Try CPU-only training"
echo "=================================================="
