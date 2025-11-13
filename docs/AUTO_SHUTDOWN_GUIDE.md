# 🔌 Auto-Shutdown Guide

## Problem
Saat menggunakan `--shutdown` flag, script membutuhkan password sudo yang membuat terminal menunggu input dan tidak bisa auto-shutdown.

## ✅ Solution: Passwordless Shutdown

### Quick Setup (1 menit)

1. **Run setup script:**
```bash
chmod +x scripts/setup_passwordless_shutdown.sh
./scripts/setup_passwordless_shutdown.sh
```

2. **Enter your password** sekali saat diminta

3. **Done!** Sekarang `sudo shutdown` tidak perlu password lagi

---

## Manual Setup (Alternative)

Jika script otomatis tidak bekerja, lakukan manual:

### Step 1: Edit sudoers
```bash
sudo visudo
```

### Step 2: Tambahkan baris ini di akhir file
```
fadhly ALL=(ALL) NOPASSWD: /sbin/shutdown, /sbin/poweroff, /sbin/reboot, /usr/bin/systemctl poweroff, /usr/bin/systemctl reboot
```

### Step 3: Save dan exit
- Press `Ctrl+X`
- Press `Y` untuk confirm
- Press `Enter`

### Step 4: Test
```bash
sudo shutdown -c  # Test tanpa shutdown
```

---

## 🚀 Usage

### Training dengan auto-shutdown:
```bash
python scripts/train.py --epochs 50 --batch-size 4 --amp --shutdown
```

### Overnight training:
```bash
nohup python scripts/train.py --epochs 100 --amp --shutdown > training.log 2>&1 &
```

### Cancel shutdown (60 detik countdown):
```
Press Ctrl+C selama countdown
```

---

## 🔒 Security Note

Setup ini **AMAN** karena:
- ✅ Hanya mengizinkan shutdown/reboot commands
- ✅ Tidak memberikan full root access
- ✅ User tetap harus sudo untuk command lain
- ✅ File permissions di-protect (0440)

---

## 🧪 Testing

### Test 1: Passwordless shutdown
```bash
sudo shutdown -c  # Should NOT ask password
```

### Test 2: Other commands still need password
```bash
sudo apt update  # Should STILL ask password
```

### Test 3: Training script shutdown
```bash
python scripts/train.py --epochs 1 --shutdown
```

---

## ❌ Troubleshooting

### "sudo: no tty present and no askpass program specified"
**Solution:** Run setup script lagi dengan proper permissions

### "sudoers file: syntax error"
**Solution:** 
```bash
sudo rm /etc/sudoers.d/allow-shutdown-*
sudo visudo  # Fix manually
```

### Shutdown tidak jalan setelah training
**Solution:** Cek log file:
```bash
tail -f training.log
```

---

## 🔧 Uninstall

Jika ingin kembali ke setting normal:
```bash
sudo rm /etc/sudoers.d/allow-shutdown-$USER
```

---

## 📊 Recommended Training Command

```bash
# Overnight training dengan auto-shutdown
python scripts/train.py \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4 \
    --amp \
    --shutdown \
    --num-workers 0 \
    --seed 42 \
    > training_$(date +%Y%m%d_%H%M%S).log 2>&1
```

Training selesai → Countdown 60 detik → Auto shutdown! 🎉
