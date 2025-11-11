# ImagiNet Dataset Download Guide

Script untuk mendownload **ImagiNet dataset** yang digunakan dalam proyek deteksi gambar AI-generated.

## 📦 Tentang ImagiNet

**ImagiNet** adalah dataset multi-konten untuk deteksi gambar sintetis (AI-generated images) menggunakan Contrastive Learning.

- **Source**: [delyanboychev/imaginet](https://huggingface.co/datasets/delyanboychev/imaginet)
- **Paper**: [ImagiNet: A Multi-Content Dataset for Generalizable Synthetic Image Detection](https://arxiv.org/abs/2403.01077)
- **GitHub**: [github.com/delyan-boychev/imagenet](https://github.com/delyan-boychev/imaginet)
- **Ukuran**: ~50GB+ (compressed)
- **Format**: Images dalam format `.7z` multi-part archive
- **Konten**: Real images (ImageNet, COCO, LSUN, dll) + Synthetic images (DALL-E, Stable Diffusion, Midjourney, dll)

## 🚀 Cara Menggunakan

### ⚠️ PENTING: ImagiNet adalah Gated Dataset

Dataset ini memerlukan **authentication dan approval**. Ikuti langkah-langkah berikut:

#### 1. Buat Akun HuggingFace

Jika belum punya akun, daftar di: https://huggingface.co/join

#### 2. Request Access ke Dataset

- Kunjungi: https://huggingface.co/datasets/delyanboychev/imaginet
- Klik tombol **"Agree and Access Repository"** atau **"Request Access"**
- Approval biasanya instant (otomatis)

#### 3. Login ke HuggingFace CLI

```bash
# Buat token di: https://huggingface.co/settings/tokens
# Pastikan token punya scope "read"

huggingface-cli login
# Paste token saat diminta
```

### Prerequisites

Setelah login, pastikan:
- Python 3.7+ terinstall
- Koneksi internet stabil
- Storage minimal 60GB (50GB dataset + 10GB untuk extract)

### Menjalankan Script

```bash
cd datasets
python download_imaginet.py
```

Script akan:
1. ✅ Cek dan install `huggingface-cli` jika diperlukan
2. � Verifikasi login status
3. 💡 Tawarkan pilihan: Official HF atau HF Mirror (lebih cepat untuk Asia)
4. �📥 Download dataset dari HuggingFace (~50GB)
5. 💾 Simpan ke `data/raw/imaginet/`
6. 📋 Memberikan instruksi untuk extract file `.7z`

#### 🌏 HuggingFace Mirror

Jika download dari official HF lambat, pilih **HF Mirror** saat script tanya:
- **Mirror**: https://hf-mirror.com
- **Keuntungan**: Download lebih cepat untuk region Asia/China
- **Catatan**: Mirror adalah proxy resmi, aman digunakan

## 📁 Struktur Output

Setelah download, struktur folder:

```
data/raw/imaginet/
├── imaginet.7z.001
├── imaginet.7z.002
├── imaginet.7z.003
├── ...
└── README.md
```

## 📦 Extract Dataset

Setelah download selesai, extract file `.7z`:

### Install 7z (jika belum ada)

```bash
# Ubuntu/Debian
sudo apt install p7zip-full

# macOS
brew install p7zip

# Windows: Download dari https://www.7-zip.org/
```

### Extract Dataset

```bash
cd data/raw/imaginet
7z x imaginet.7z.001
```

Proses extract akan:
- Otomatis menggabungkan semua part (`.001`, `.002`, dst)
- Extract ke folder yang sama
- Memakan waktu 10-30 menit tergantung hardware

Setelah extract, struktur akan jadi:

```
data/raw/imaginet/
├── train/                # Training images
│   ├── real/
│   └── synthetic/
├── test/                 # Test images
│   ├── real/
│   └── synthetic/
├── annotations/          # Annotation files
└── README.md
```

## ⚠️ Catatan Penting

- **Ukuran**: Dataset sangat besar (~50GB compressed, ~60GB extracted)
- **Waktu Download**: Bergantung koneksi internet (bisa 1-6 jam)
- **Resume**: Jika download terputus, jalankan ulang script - akan otomatis resume
- **Storage**: Pastikan punya space 60GB+ di disk
- **Git**: File dataset otomatis di-ignore (sudah ada di `.gitignore`)

## 🔧 Troubleshooting

### Error: `huggingface-cli not found`

Script seharusnya auto-install. Jika gagal, manual install:

```bash
pip install huggingface-hub
```

### Download Lambat/Terputus

- Script support resume - jalankan ulang saja
- Gunakan koneksi internet yang stabil
- Pertimbangkan download di waktu off-peak

### Storage Penuh

Cek space tersedia:

```bash
df -h
```

Bersihkan space atau gunakan disk lain:

```bash
# Pindahkan OUT_DIR di script ke disk lain
# Edit line 6 di download_imaginet.py:
OUT_DIR = "/path/to/disk/lain/data/raw"
```

### Extract Gagal

Pastikan semua `.7z` parts terdownload lengkap:

```bash
cd data/raw/imaginet
ls -lh imaginet.7z.*
```

Harus ada semua parts (.001, .002, dst). Jika ada yang missing, download ulang.

## 📊 Dataset Statistics

**ImagiNet Dataset** berisi:
- **Training**: ~200K images (100K real + 100K synthetic)
- **Testing**: Multiple test sets untuk berbagai generative models
- **Real Images**: ImageNet, COCO, LSUN, FFHQ, WikiArt, Danbooru, Photozilla
- **Synthetic**: DALL-E 3, Stable Diffusion, SDXL, Midjourney, StyleGAN, dll

## 📚 Referensi

- [ImagiNet HuggingFace](https://huggingface.co/datasets/delyanboychev/imaginet)
- [ImagiNet GitHub](https://github.com/delyan-boychev/imaginet)
- [ImagiNet Paper (arXiv)](https://arxiv.org/abs/2403.01077)
- [Model Checkpoints](https://drive.google.com/drive/folders/1En2BI9H9LxqA5XIpNaMXhqhF8--XAKns)

## 💡 Tips

- Download di malam hari saat bandwidth lebih stabil
- Gunakan `tmux` atau `screen` agar download tidak terputus saat terminal tertutup
- Monitor progress dengan `du -sh data/raw/imaginet/` di terminal lain
- Backup file `.7z` sebelum extract (opsional, untuk berjaga-jaga)
