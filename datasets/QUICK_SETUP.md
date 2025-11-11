# Quick Setup Guide - ImagiNet Dataset

## 🎯 Langkah Cepat (5 Menit)

### 1. Buat Akun HuggingFace
👉 https://huggingface.co/join

### 2. Request Access
👉 https://huggingface.co/datasets/delyanboychev/imaginet
- Klik **"Agree and Access Repository"**
- Tunggu beberapa detik (approval otomatis)

### 3. Buat Access Token
👉 https://huggingface.co/settings/tokens
- Klik **"New token"**
- Name: `imaginet-download` (atau nama apapun)
- Type: **Read**
- Klik **"Generate"**
- **Copy token** (simpan di tempat aman!)

### 4. Login di Terminal

```bash
huggingface-cli login
```

Paste token yang sudah dicopy, tekan Enter.

### 5. Download Dataset

```bash
cd datasets
python download_imaginet.py
```

**Pilihan Download**:
- Option 1: HuggingFace Official (global)
- Option 2: HF Mirror (lebih cepat untuk Asia) 🚀

💡 Pilih option 2 jika koneksi ke HF lambat!

## ✅ Troubleshooting Cepat

### Error: "401 Unauthorized" atau "GatedRepoError"

**Penyebab**: Belum request access atau belum login

**Solusi**:
1. Pastikan sudah klik "Agree and Access" di halaman dataset
2. Login ulang: `huggingface-cli login`
3. Coba jalankan script lagi

### Error: "huggingface-cli not found"

**Solusi**:
```bash
pip install huggingface-hub
```

### Download Lambat

**Solusi 1: Gunakan HF Mirror**
```bash
# Script akan tanya - pilih option 2
python download_imaginet.py
```

**Solusi 2: Set HF Mirror secara manual**
```bash
export HF_ENDPOINT=https://hf-mirror.com
python download_imaginet.py
```

**Tips Lain**:
- Download saat malam (bandwidth lebih stabil)
- Gunakan `tmux` atau `screen` agar tidak terputus
- Script support resume - jalankan ulang jika terputus

## 📞 Bantuan

Jika masih ada masalah:
1. Cek [README.md](README.md) lengkap
2. Lihat [ImagiNet GitHub Issues](https://github.com/delyan-boychev/imaginet/issues)
3. Tanya di [HuggingFace Discussions](https://huggingface.co/datasets/delyanboychev/imaginet/discussions)

---

**Estimasi Waktu Total**:
- Setup: 5 menit
- Download: 1-6 jam (tergantung internet)
- Extract: 10-30 menit

**Storage Required**: ~60GB
