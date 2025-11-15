# 🔄 Training with Auto-Logging & Auto-Shutdown

## Fitur Baru

Sekarang `train.py` memiliki fitur:
1. **Auto-logging** - Semua output di-save ke `results/logs/`
2. **Auto-shutdown** - Sistem otomatis shutdown setelah training selesai

## Penggunaan

### Basic Training (tanpa shutdown)
```bash
python scripts/train.py --batch-size 16 --epochs 50 --amp
```

### Training dengan Auto-Shutdown
```bash
python scripts/train.py --batch-size 16 --epochs 50 --amp --shutdown
```

## Log Files

Semua log otomatis disimpan ke:
```
results/logs/train_log_YYYYMMDD_HHMMSS.txt
```

Contoh:
```
results/logs/train_log_20251113_152030.txt
results/logs/train_log_20251113_155045.txt
```

## Melihat Log Files

### List semua log files
```bash
ls -lh results/logs/
```

### Lihat log terbaru
```bash
tail -f results/logs/train_log_*.txt | tail -50
```

### Cari specific log
```bash
grep "Best validation" results/logs/*.txt
```

## Auto-Shutdown

### Enable Auto-Shutdown
```bash
python scripts/train.py --shutdown --epochs 50 --batch-size 16 --amp
```

### How it works:
1. Training selesai
2. System akan menampilkan countdown 10 detik
3. Sistem akan shutdown otomatis
4. Tekan **Ctrl+C** untuk cancel shutdown

### Shutdown Countdown
```
🔴 SYSTEM SHUTDOWN IN 10 SECONDS...
============================================================
Press Ctrl+C to cancel shutdown.
Shutting down in 10s...
Shutting down in 9s...
...
Initiating system shutdown...
```

## Example Usage

### Scenario: Long Training Overnight
```bash
# Start training, system akan auto-shutdown setelah selesai
nohup python scripts/train.py \
  --batch-size 16 \
  --epochs 100 \
  --amp \
  --shutdown > training.log 2>&1 &
```

### Scenario: Quick Test (no shutdown)
```bash
python scripts/train.py --batch-size 8 --epochs 2 --amp
```

## Complete Training Output

Training sekarang akan:
1. ✅ Print ke terminal
2. ✅ Save ke `results/logs/train_log_YYYYMMDD_HHMMSS.txt`
3. ✅ Save checkpoints ke `models/checkpoints/`
4. ✅ Show training progress dengan tqdm
5. ✅ Auto-shutdown jika `--shutdown` digunakan

## Log File Contents

Setiap log file berisi:

```
============================================================
📁 LOADING DATASET
============================================================
[TRAIN] Loaded 27000 images
[VAL] Loaded 6750 images

============================================================
🤖 INITIALIZING MODEL
============================================================
Model: 1,234,567 trainable / 4,321,098 total params

============================================================
🚀 STARTING TRAINING
============================================================

Config:
   Device: cuda
   Batch size: 16
   Learning rate: 1e-04
   Epochs: 50
   Mixed precision: True
   Seed: 42
   Shutdown after training: True
   Log file: /path/to/results/logs/train_log_20251113_152030.txt

📊 Epoch 1/50 Summary:
   Train Loss: 0.6234 | Train Acc: 62.45%
   Val Loss:   0.5123 | Val Acc:   65.78%
   LR: 6.28e-05

...

============================================================
✅ TRAINING COMPLETE
============================================================

Best validation accuracy: 89.23%
Checkpoints saved in: /path/to/models/checkpoints
   - Latest: hybrid_imaginet_latest.pth
   - Best: hybrid_imaginet_best.pth

Log file saved: /path/to/results/logs/train_log_20251113_152030.txt
Training finished at: 2025-11-13 23:25:45

🔴 SYSTEM SHUTDOWN IN 10 SECONDS...
```

## Tips

### Monitor Training in Real-time
```bash
# Terminal 1: Start training
python scripts/train.py --shutdown --epochs 50 --batch-size 16 --amp

# Terminal 2: Monitor log
watch -n 5 'tail -20 results/logs/train_log_*.txt'
```

### Save All Logs to Archive
```bash
# After training done
tar -czf training_logs_$(date +%Y%m%d).tar.gz results/logs/
```

### Find Best Training Run
```bash
grep "validation accuracy:" results/logs/*.txt | sort -k3 -rn | head -1
```

## Arguments Reference

```bash
python scripts/train.py --help

optional arguments:
  --batch-size BATCH_SIZE   Batch size per step (default: 4)
  --lr LR                   Learning rate (default: 1e-4)
  --epochs EPOCHS           Number of epochs (default: 50)
  --resume                  Resume from checkpoint
  --amp                     Use mixed precision
  --num-workers NUM_WORKERS DataLoader workers (default: 0)
  --freeze-cnn              Freeze CNN backbone
  --seed SEED               Random seed (default: 42)
  --shutdown                Shutdown system after training completes
```

---

**✅ Training dengan auto-logging & auto-shutdown siap digunakan!**
