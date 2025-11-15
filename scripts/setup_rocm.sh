#!/bin/bash
# Setup script untuk ROCm 5.4.1 + PyTorch 1.11 di RX 580

echo "=================================================="
echo "🔧 ROCM 5.4.1 SETUP FOR RX 580"
echo "=================================================="
echo ""

# Detect shell
if [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
    SHELL_NAME="zsh"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_RC="$HOME/.bashrc"
    SHELL_NAME="bash"
else
    SHELL_RC="$HOME/.profile"
    SHELL_NAME="default"
fi

echo "🐚 Detected shell: $SHELL_NAME"
echo "📄 Config file: $SHELL_RC"
echo ""

# 1. Set environment variables permanently
echo "📝 Setting ROCm environment variables..."

# Add to shell rc if not already present
if ! grep -q "HSA_OVERRIDE_GFX_VERSION" "$SHELL_RC"; then
    cat >> "$SHELL_RC" << 'EOF'

# ROCm 5.4.1 settings for RX 580 (gfx803)
export HSA_OVERRIDE_GFX_VERSION=8.0.3
export PYTORCH_ROCM_ARCH=gfx803
export HIP_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# Optimize for training
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
EOF
    echo "✅ Environment variables added to $SHELL_RC"
else
    echo "✅ Environment variables already in $SHELL_RC"
fi

# Load environment variables for current session
export HSA_OVERRIDE_GFX_VERSION=8.0.3
export PYTORCH_ROCM_ARCH=gfx803
export HIP_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

echo "✅ Environment variables loaded for current session"

# 2. Verify ROCm installation
echo ""
echo "🔍 Verifying ROCm installation..."

if command -v rocminfo &> /dev/null; then
    echo "✅ rocminfo found"
    rocminfo | grep -E "Name:|Marketing Name:"
else
    echo "❌ rocminfo not found! Please install ROCm 5.4.1"
    exit 1
fi

# 3. Check PyTorch
echo ""
echo "🐍 Checking PyTorch installation..."

python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA/ROCm available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    
    # Check if ROCm
    try:
        import torch.version
        if hasattr(torch.version, 'hip'):
            print(f'ROCm version: {torch.version.hip}')
    except:
        pass
else:
    print('⚠️  WARNING: CUDA/ROCm not available!')
    print('PyTorch may not be built with ROCm support')
" || {
    echo "❌ PyTorch not installed or error occurred"
    echo ""
    echo "To install PyTorch 1.11 for ROCm 5.4.1:"
    echo "  pip3 install torch==1.11.0+rocm5.4.1 torchvision==0.12.0+rocm5.4.1 --extra-index-url https://download.pytorch.org/whl/rocm5.4.1"
    exit 1
}

# 4. Test GPU computation
echo ""
echo "🧪 Testing GPU computation..."

python3 << 'PYCODE'
import torch
import time

if not torch.cuda.is_available():
    print("❌ CUDA/ROCm not available, cannot test GPU")
    exit(1)

print("Testing matrix multiplication on GPU...")

# Warm-up
a = torch.randn(1000, 1000, device='cuda')
b = torch.randn(1000, 1000, device='cuda')
c = torch.matmul(a, b)
torch.cuda.synchronize()

# Benchmark
times = []
for i in range(5):
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    times.append(time.time() - start)

avg_time = sum(times) / len(times)
print(f"✅ GPU working! Avg time: {avg_time*1000:.2f}ms")

# Test memory allocation
try:
    large = torch.randn(5000, 5000, device='cuda')
    del large
    torch.cuda.empty_cache()
    print("✅ Memory allocation working")
except Exception as e:
    print(f"⚠️  Memory allocation issue: {e}")
PYCODE

# 5. Check dataset
echo ""
echo "📁 Checking dataset..."

if [ -d "data/raw/imaginet/subset/real" ] || [ -d "data/raw/imaginet/subset/fake" ]; then
    echo "✅ Dataset found"
    
    real_count=$(find data/raw/imaginet/subset/real -type f 2>/dev/null | wc -l)
    fake_count=$(find data/raw/imaginet/subset/fake -type f 2>/dev/null | wc -l)
    
    echo "   Real images: $real_count"
    echo "   Fake images: $fake_count"
    echo "   Total: $((real_count + fake_count))"
else
    echo "⚠️  Dataset not found in data/raw/imaginet/subset/"
    echo "Please extract dataset first"
fi

# 6. Print summary
echo ""
echo "=================================================="
echo "✅ SETUP COMPLETE"
echo "=================================================="
echo ""
echo "📋 Training recommendations for RX 580:"
echo "   • Batch size: 2-4 (RX 580 has 8GB VRAM)"
echo "   • Workers: 0 (REQUIRED for ROCm 5.4.1)"
echo "   • Mixed precision: Disabled (ROCm 5.4.1 limitation)"
echo "   • Gradient accumulation: 8-16 steps"
echo ""
echo "🚀 Example training command:"
echo "   python scripts/train_rocm.py --batch-size 4 --epochs 50"
echo ""
echo "💡 For overnight training with auto-shutdown:"
echo "   python scripts/train_rocm.py --batch-size 4 --epochs 50 --shutdown"
echo ""
echo "🔄 To apply environment variables now ($SHELL_NAME):"
if [ "$SHELL_NAME" = "zsh" ]; then
    echo "   source ~/.zshrc"
elif [ "$SHELL_NAME" = "bash" ]; then
    echo "   source ~/.bashrc"
else
    echo "   source $SHELL_RC"
fi
echo ""
echo "=================================================="
