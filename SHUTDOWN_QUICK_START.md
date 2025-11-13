# ⚡ Quick Start: Auto-Shutdown Training

## 🎯 Goal
Training overnight dan auto-shutdown setelah selesai **tanpa prompt password**.

---

## 📋 Setup (Sekali Saja - 1 Menit)

```bash
# 1. Setup passwordless shutdown
chmod +x scripts/setup_passwordless_shutdown.sh
./scripts/setup_passwordless_shutdown.sh

# 2. Masukkan password sekali (untuk setup)
# 3. Done! Tidak akan diminta lagi
```

---

## 🚀 Usage

### Opsi 1: Script Otomatis (Recommended)
```bash
chmod +x scripts/train_overnight.sh
./scripts/train_overnight.sh
```

### Opsi 2: Manual Command
```bash
python scripts/train.py \
    --epochs 50 \
    --batch-size 4 \
    --amp \
    --shutdown
```

### Opsi 3: Background + Log
```bash
nohup python scripts/train.py --epochs 50 --amp --shutdown > training.log 2>&1 &
```

---

## ⏱️ Shutdown Countdown

Setelah training selesai:
```
✅ TRAINING COMPLETE
🔌 SHUTTING DOWN SYSTEM IN 60 SECONDS...
   Shutdown in 60 seconds...
   Shutdown in 50 seconds...
   ...
```

**Cancel shutdown:** Press `Ctrl+C` selama countdown

---

## 🧪 Test (Tanpa Training)

```bash
# Test 1: Check passwordless shutdown
sudo shutdown -c  # Harus TIDAK minta password

# Test 2: Quick training test
python scripts/train.py --epochs 1 --shutdown
```

---

## ❓ FAQ

**Q: Kenapa masih minta password?**  
A: Run setup script dulu: `./scripts/setup_passwordless_shutdown.sh`

**Q: Apakah aman?**  
A: Ya! Hanya shutdown/reboot yang tanpa password. Command lain tetap perlu password.

**Q: Bisa cancel shutdown?**  
A: Ya! Press Ctrl+C selama 60 detik countdown.

**Q: Shutdown tidak jalan?**  
A: Pastikan sudah run setup script dan test dengan `sudo shutdown -c`

---

## 📁 Related Files

- `scripts/setup_passwordless_shutdown.sh` - Setup script
- `scripts/train_overnight.sh` - Overnight training script  
- `scripts/train.py` - Main training script
- `docs/AUTO_SHUTDOWN_GUIDE.md` - Detailed guide

---

**Selamat training overnight! 🌙**
