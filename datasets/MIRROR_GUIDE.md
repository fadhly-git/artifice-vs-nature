# HuggingFace Mirror Options

Jika download dari HuggingFace official lambat, gunakan salah satu mirror endpoint berikut.

## 🌏 Mirror Endpoints

### 1. **HF-Mirror (Recommended for Asia)**
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
- **Region**: Global (optimized for Asia/China)
- **Status**: Active & maintained
- **Website**: https://hf-mirror.com

### 2. **Official HuggingFace**
```bash
unset HF_ENDPOINT
# atau
export HF_ENDPOINT=https://huggingface.co
```
- **Region**: Global
- **Default**: Ya

## 🚀 Cara Menggunakan

### Option 1: Via Script (Mudah)

Jalankan script, lalu pilih option 2:

```bash
python download_imaginet.py

# Saat ditanya:
# Pilih (1/2) [default: 1]: 2  ← pilih ini
```

### Option 2: Set Environment Variable

**Temporary (sekali pakai)**:
```bash
export HF_ENDPOINT=https://hf-mirror.com
python download_imaginet.py
```

**Permanent (semua session)**:
```bash
# Tambahkan ke ~/.bashrc atau ~/.zshrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc

# Sekarang semua download HF akan pakai mirror
python download_imaginet.py
```

### Option 3: Manual CLI

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  delyanboychev/imaginet \
  --repo-type dataset \
  --local-dir data/raw/imaginet
```

## 📊 Perbandingan Speed

Estimasi download time untuk dataset 50GB:

| Endpoint | Region | Speed | Time |
|----------|--------|-------|------|
| HF Official | Indonesia | ~5-10 MB/s | 1.5-3 jam |
| HF Mirror | Indonesia | ~20-50 MB/s | 20-45 menit |
| HF Official | China | ~1-3 MB/s | 5-14 jam |
| HF Mirror | China | ~30-100 MB/s | 10-30 menit |

*Speed bervariasi tergantung ISP dan waktu*

## ⚠️ Catatan Penting

1. **Gated Datasets**: Tetap perlu login & request access meski pakai mirror
2. **Authentication**: Token HuggingFace tetap diperlukan
3. **Mirror Safety**: HF-Mirror adalah proxy resmi, aman digunakan
4. **Resume Support**: Semua endpoint support resume download

## 🔍 Verifikasi Mirror

Cek endpoint yang sedang digunakan:

```bash
echo $HF_ENDPOINT
```

Jika kosong = pakai official HuggingFace.

## 🔧 Troubleshooting

### Mirror Tidak Jalan

```bash
# Clear environment
unset HF_ENDPOINT

# Set ulang
export HF_ENDPOINT=https://hf-mirror.com

# Cek
echo $HF_ENDPOINT
```

### Kembali ke Official

```bash
unset HF_ENDPOINT
# atau
export HF_ENDPOINT=https://huggingface.co
```

### Test Connection

```bash
# Test official
curl -I https://huggingface.co

# Test mirror
curl -I https://hf-mirror.com
```

## 📚 Referensi

- [HF Mirror Official](https://hf-mirror.com)
- [HuggingFace Environment Variables](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)
- [Download from CLI](https://huggingface.co/docs/huggingface_hub/guides/download)
