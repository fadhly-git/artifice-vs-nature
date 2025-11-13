# Training dengan Auto-Shutdown

## 🌙 Overnight Training

Script training sekarang mendukung **auto-shutdown** setelah training selesai. Berguna untuk training overnight tanpa membuang listrik.

---

## 🚀 Cara Menggunakan

### **Method 1: Direct Command**

```bash
# Training dengan auto-shutdown
python scripts/train.py \
  --batch-size 2 \
  --accum-steps 16 \
  --epochs 50 \
  --amp \
  --shutdown
```

**Apa yang terjadi:**
1. ✅ Training berjalan normal untuk 50 epochs
2. ✅ Checkpoint disimpan otomatis
3. ⏰ **Countdown 60 detik** setelah training selesai
4. 🔌 **System shutdown** otomatis

---

### **Method 2: Overnight Script (Recommended)**

```bash
# Jalankan script overnight
./scripts/train_overnight.sh
```

Script ini akan:
- ✅ Auto-resume jika ada checkpoint
- ✅ Countdown 10 detik sebelum mulai (bisa cancel)
- ✅ Training 50 epochs
- ✅ Auto-shutdown setelah selesai

---

## ⚙️ Konfigurasi Overnight Training

Edit `scripts/train_overnight.sh`:

```bash
# Full Training (100% data, 50 epochs)
BATCH_SIZE=2
ACCUM_STEPS=16
EPOCHS=50
SUBSET=""  # Full dataset

# OR Test Training (10% data, 10 epochs)
BATCH_SIZE=4
ACCUM_STEPS=8
EPOCHS=10
SUBSET="0.1"  # 10% subset
```

---

## 🛡️ Safety Features

### **60 Second Countdown**

Setelah training selesai:
```
✅ TRAINING COMPLETE
🏆 Best Accuracy: 89.45%
💾 Checkpoints saved

🔌 System will shutdown in 60 seconds...
   Press Ctrl+C to cancel
   
   Shutting down in 60 seconds...
   Shutting down in 50 seconds...
   Shutting down in 40 seconds...
   ...
```

**Press Ctrl+C untuk cancel shutdown!**

### **Cancel Shutdown**

```
   Shutting down in 30 seconds...
^C
⚠️  Shutdown cancelled by user
   System will remain running
```

---

## 📋 Complete Examples

### **Example 1: Quick Test (NO Shutdown)**

```bash
# Test 5 epochs, NO shutdown
python scripts/train.py \
  --batch-size 4 \
  --subset 0.1 \
  --epochs 5
```

### **Example 2: Overnight Training (WITH Shutdown)**

```bash
# Full training, auto-shutdown
python scripts/train.py \
  --batch-size 2 \
  --accum-steps 16 \
  --epochs 50 \
  --amp \
  --shutdown
```

**Waktu estimasi:** ~13-15 jam untuk 50 epochs

### **Example 3: Resume + Shutdown**

```bash
# Resume dari checkpoint, lalu shutdown
python scripts/train.py \
  --resume \
  --epochs 50 \
  --shutdown
```

### **Example 4: Subset Training + Shutdown**

```bash
# 20% data untuk testing, shutdown setelah selesai
python scripts/train.py \
  --batch-size 4 \
  --subset 0.2 \
  --epochs 20 \
  --shutdown
```

---

## 🔐 Sudo Permission untuk Shutdown

Script menggunakan `sudo shutdown -h now`. Agar tidak perlu password:

```bash
# Edit sudoers
sudo visudo

# Tambahkan di akhir file:
YOUR_USERNAME ALL=(ALL) NOPASSWD: /sbin/shutdown
```

**Ganti `YOUR_USERNAME` dengan username Linux Anda.**

Atau jalankan script dengan sudo:
```bash
sudo python scripts/train.py --epochs 50 --shutdown
```

---

## ⏰ Timeline Estimasi

| Dataset | Epochs | Batch Size | Estimated Time | Auto-Shutdown |
|---------|--------|------------|----------------|---------------|
| 10% | 5 | 4 | ~15 menit | ❌ Not needed |
| 10% | 10 | 4 | ~30 menit | ❌ Not needed |
| 50% | 25 | 2 | ~6 jam | ✅ Recommended |
| 100% | 50 | 2 | **~13 jam** | ✅ **Overnight** |
| 100% | 100 | 2 | **~26 jam** | ✅ **Weekend** |

---

## 📊 Monitoring Overnight Training

### **Before You Sleep:**

1. ✅ Start training dengan `--shutdown`
2. ✅ Check progress bar berjalan
3. ✅ Note ETA di tqdm
4. ✅ Tutup semua aplikasi lain
5. 💤 Sleep!

### **Next Morning:**

1. ✅ Computer sudah shutdown otomatis
2. ✅ Buka computer, check results
3. ✅ Load checkpoint untuk evaluation

```bash
# Check hasil training
ls -lh models/checkpoints/
# hybrid_imaginet_best.pth
# hybrid_imaginet_latest.pth

# Load best model untuk testing
python scripts/evaluate.py --checkpoint models/checkpoints/hybrid_imaginet_best.pth
```

---

## 🐛 Troubleshooting

### **"Shutdown requires sudo password"**

**Solusi 1:** Add NOPASSWD untuk shutdown (recommended)
```bash
sudo visudo
# Add: YOUR_USERNAME ALL=(ALL) NOPASSWD: /sbin/shutdown
```

**Solusi 2:** Run dengan sudo
```bash
sudo python scripts/train.py --shutdown
```

**Solusi 3:** Manual shutdown
```bash
# Don't use --shutdown flag
python scripts/train.py --epochs 50

# Manually shutdown after
sudo shutdown -h now
```

### **"Training stopped, but system didn't shutdown"**

Check:
1. ✅ Apakah `--shutdown` flag digunakan?
2. ✅ Apakah ada error saat training?
3. ✅ Check terminal output

### **"Want to cancel shutdown after training done"**

```bash
# Press Ctrl+C during 60 second countdown
# Or in another terminal:
sudo shutdown -c
```

---

## 💡 Best Practices

### **✅ DO:**
- Use `--shutdown` for overnight/long training (>6 hours)
- Test without `--shutdown` first untuk pastikan tidak ada error
- Close semua aplikasi lain sebelum start
- Save progress pekerjaan lain sebelum start

### **❌ DON'T:**
- Don't use `--shutdown` untuk quick testing (<1 hour)
- Don't start training jika ada unsaved work
- Don't forget to check disk space untuk checkpoints

---

## 📝 Example Workflow

### **Friday Evening:**

```bash
# 1. Test training dulu (NO shutdown)
python scripts/train.py --subset 0.01 --epochs 1

# 2. Jika OK, start overnight training
python scripts/train.py \
  --batch-size 2 \
  --accum-steps 16 \
  --epochs 50 \
  --amp \
  --shutdown

# 3. Go to sleep 💤
```

### **Saturday Morning:**

```bash
# 1. Turn on computer
# 2. Check results
ls models/checkpoints/

# 3. Load dan evaluate
python scripts/evaluate.py
```

---

## 🎯 Quick Reference

| Command | Use Case |
|---------|----------|
| `--shutdown` | Auto-shutdown setelah training |
| `--resume` | Resume dari checkpoint |
| `--subset 0.1` | Use 10% data |
| `--epochs 50` | Train 50 epochs |
| `./scripts/train_overnight.sh` | Run overnight script |

---

**Last Updated:** November 11, 2025  
**Tested on:** AMD RX 580, Ubuntu 20.04, ROCm 3.5
