# Training Guide - Hybrid Detector

## 🚀 Quick Start

### 1. Check GPU Memory
```bash
python scripts/check_vram.py
```

### 2. Start Training (Conservative Settings)
```bash
# For AMD RX 580 (8GB VRAM) - SAFE settings
python scripts/train.py \
  --batch-size 2 \
  --accum-steps 16 \
  --lr 1e-4 \
  --epochs 50 \
  --amp \
  --num-workers 0
```

### 3. Resume Training
```bash
python scripts/train.py --resume --amp
```

---

## ⚙️ Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | 2 | Batch size per step (lower = less VRAM) |
| `--accum-steps` | 16 | Gradient accumulation steps |
| `--lr` | 1e-4 | Learning rate |
| `--epochs` | 50 | Number of epochs |
| `--resume` | False | Resume from latest checkpoint |
| `--amp` | False | Use mixed precision training |
| `--num-workers` | 0 | DataLoader workers (0 = main thread only) |

**Effective Batch Size** = `batch-size × accum-steps`

---

## 🔧 Memory Optimization

### Problem: "Memory access fault by GPU"

**Causes:**
- VRAM overflow (> 8GB)
- Memory fragmentation
- ROCm driver issues

**Solutions:**

#### Option 1: Reduce Batch Size ✅
```bash
python scripts/train.py --batch-size 1 --accum-steps 32 --amp
```

#### Option 2: Disable AMP
```bash
python scripts/train.py --batch-size 2 --accum-steps 16
```

#### Option 3: Smaller Model
Edit `src/models/hybrid.py`:
```python
# Change from:
self.cnn = timm.create_model('efficientnet_b4', ...)

# To:
self.cnn = timm.create_model('efficientnet_b0', ...)  # Much smaller!
```

#### Option 4: Clear Cache
The script now auto-clears cache every 100 batches and after each epoch.

---

## 📊 Training Progress

### With tqdm (New!)
```
Epoch 1/50: 100%|████████| 8438/8438 [2:15:30<00:00, loss=0.6234, acc=65.23%]

📊 Epoch 1/50 Summary:
   Loss: 0.6234
   Accuracy: 65.23%
   🌟 New best accuracy: 65.23% - Saved!
```

### Old Format (Before)
```
Epoch [1/50] | Batch [0/8438] | Loss: 0.7189 | Acc: 50.00%
Epoch [1/50] | Batch [10/8438] | Loss: 0.6890 | Acc: 72.73%
...
```

---

## 💾 Checkpoints

Training saves two checkpoint files:

1. **`hybrid_imaginet_latest.pth`** - Latest epoch (for resuming)
2. **`hybrid_imaginet_best.pth`** - Best accuracy (for inference)

**Location:** `models/checkpoints/`

**Contents:**
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'loss': float,
    'accuracy': float,
    'best_acc': float
}
```

---

## 🎯 Recommended Settings

### AMD RX 580 (8GB VRAM)

#### Conservative (Safest) ✅
```bash
python scripts/train.py \
  --batch-size 1 \
  --accum-steps 32 \
  --lr 1e-4 \
  --epochs 50 \
  --amp \
  --num-workers 0
```
- **VRAM:** ~4-5 GB
- **Speed:** Slower but stable
- **Effective batch:** 32

#### Balanced (Recommended)
```bash
python scripts/train.py \
  --batch-size 2 \
  --accum-steps 16 \
  --lr 1e-4 \
  --epochs 50 \
  --amp \
  --num-workers 0
```
- **VRAM:** ~6-7 GB
- **Speed:** Medium
- **Effective batch:** 32

#### Aggressive (May OOM)
```bash
python scripts/train.py \
  --batch-size 4 \
  --accum-steps 8 \
  --lr 1e-4 \
  --epochs 50 \
  --amp \
  --num-workers 2
```
- **VRAM:** ~7-8 GB (risky!)
- **Speed:** Fastest
- **Effective batch:** 32

---

## 🐛 Troubleshooting

### 1. "Memory access fault by GPU node-1"
**Fix:** Reduce batch size or disable AMP
```bash
python scripts/train.py --batch-size 1 --accum-steps 32
```

### 2. Training too slow
**Fix:** Increase batch size (if VRAM allows)
```bash
python scripts/check_vram.py  # Check available memory first
python scripts/train.py --batch-size 4 --accum-steps 8 --amp
```

### 3. "CUDA out of memory"
**Fix:** Clear cache manually
```bash
# In Python:
import torch
torch.cuda.empty_cache()
```

### 4. Loss not decreasing
**Fixes:**
- Check learning rate: `--lr 1e-5` (lower)
- Check data balance (fake vs real)
- Verify preprocessing pipeline

### 5. Accuracy stuck at 50%
**Causes:**
- Model predicting same class always
- Learning rate too high/low
- Data imbalance

**Fix:** Check dataset:
```bash
python -c "
from pathlib import Path
fake = len(list(Path('data/processed/imaginet/subset_pt/fake').glob('*.pt')))
real = len(list(Path('data/processed/imaginet/subset_pt/real').glob('*.pt')))
print(f'Fake: {fake}, Real: {real}, Ratio: {fake/real:.2f}')
"
```

---

## 📈 Monitoring

### Real-time GPU Monitoring
```bash
# Terminal 1: Start training
python scripts/train.py --batch-size 2 --accum-steps 16 --amp

# Terminal 2: Monitor GPU
watch -n 1 rocm-smi
```

### TensorBoard (Future Enhancement)
```bash
# TODO: Add tensorboard logging
tensorboard --logdir results/logs
```

---

## 🎓 Training Tips

1. **Start with conservative settings** - Avoid OOM errors
2. **Monitor first epoch** - Check if VRAM is stable
3. **Use --resume** - Don't lose progress on crashes
4. **Save best model** - Script auto-saves best accuracy
5. **Gradient accumulation** - Maintains effective batch size with low VRAM

---

## 📝 Example Training Session

```bash
# Check VRAM first
python scripts/check_vram.py

# Start training
python scripts/train.py \
  --batch-size 2 \
  --accum-steps 16 \
  --lr 1e-4 \
  --epochs 50 \
  --amp \
  --num-workers 0

# If interrupted, resume:
python scripts/train.py --resume --amp

# Check final model
ls -lh models/checkpoints/
```

**Expected Output:**
```
Epoch 1/50: 100%|████████| 8438/8438 [loss=0.5234, acc=67.89%]
📊 Epoch 1/50 Summary:
   Loss: 0.5234
   Accuracy: 67.89%
   🌟 New best accuracy: 67.89% - Saved!

... (continues for 50 epochs)

✅ TRAINING COMPLETE
Best accuracy: 89.45%
Checkpoints saved in: models/checkpoints
```

---

**Last Updated:** November 11, 2025  
**Tested on:** AMD RX 580 (8GB VRAM), ROCm 3.5, PyTorch 1.7.0
