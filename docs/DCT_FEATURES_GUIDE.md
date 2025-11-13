# DCT Features Guide

## Apa itu DCT Features?

**DCT (Discrete Cosine Transform)** adalah transformasi yang mengubah sinyal gambar dari domain spasial ke domain frekuensi. DCT features sangat berguna untuk mendeteksi AI-generated images karena:

1. **Menangkap pola kompresi JPEG** - Gambar AI sering memiliki pola kompresi yang berbeda
2. **Deteksi artefak frekuensi tinggi** - AI generators meninggalkan "sidik jari" di frekuensi tinggi
3. **Invariant terhadap transformasi spasial** - Lebih robust terhadap crop, resize, dll

## 📊 Cara Mendapatkan DCT Features

### Opsi 1: Precompute DCT Features (RECOMMENDED)

Metode ini **lebih cepat** karena DCT hanya dihitung sekali dan disimpan sebagai file `.npy`.

#### Langkah 1: Jalankan Script Precompute

```bash
# Basic usage
python scripts/precompute_dct.py

# Custom parameters
python scripts/precompute_dct.py \
    --image-dir data/processed/imaginet/subset \
    --output-dir data/processed/imaginet/dct_features \
    --top-k 1024 \
    --block-size 8

# Overwrite existing files
python scripts/precompute_dct.py --overwrite

# Verify all images have DCT files
python scripts/precompute_dct.py --verify
```

#### Langkah 2: Struktur Folder yang Dihasilkan

```
data/processed/imaginet/
├── subset/
│   ├── real/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   └── fake/
│       ├── image001.jpg
│       ├── image002.jpg
│       └── ...
└── dct_features/
    ├── image001.npy  # DCT for real/image001.jpg
    ├── image002.npy  # DCT for real/image002.jpg
    └── ...
```

#### Langkah 3: Training dengan DCT Features

```bash
python scripts/train.py --batch-size 4 --epochs 50 --amp
```

Script training akan otomatis load DCT features dari `data/processed/imaginet/dct_features/`.

---

### Opsi 2: Compute DCT On-the-Fly

Jika Anda **tidak ingin** precompute, DCT bisa dihitung on-the-fly di dataset. Namun ini **lebih lambat**.

#### Modifikasi `src/data/dataset.py`

```python
from src.preprocessing.dct import extract_dct_features

class HybridDataset(Dataset):
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = T.ToTensor()(img)
        
        # Compute DCT on-the-fly
        img_gray = Image.open(img_path).convert('L')
        img_np = np.array(img_gray)
        dct_features = extract_dct_features(img_np, top_k=1024, block_size=8)
        dct_feat = torch.from_numpy(dct_features).float()
        
        return img_tensor, dct_feat, torch.tensor(label, dtype=torch.long)
```

⚠️ **Catatan**: Opsi ini akan memperlambat training karena DCT dihitung setiap epoch.

---

### Opsi 3: Training Tanpa DCT Features

Jika Anda ingin training **hanya dengan CNN** tanpa DCT features:

#### Langkah 1: Modifikasi Dataset

Dataset sudah support mode tanpa DCT. Pastikan `dct_dir=None`:

```python
train_loader, val_loader = create_dataloaders(
    root_dir=DATA_ROOT,
    dct_dir=None,  # Tidak gunakan DCT
    batch_size=4,
    num_workers=0,
    train_ratio=0.8,
    seed=42
)
```

Dataset akan return `torch.zeros(1024)` sebagai dummy DCT features.

#### Langkah 2: Modifikasi Model (Opsional)

Anda bisa memodifikasi `src/models/hybrid.py` agar tidak menggunakan MLP untuk DCT:

```python
class HybridDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_cnn=False, use_dct=False):
        super().__init__()
        self.use_dct = use_dct
        
        # CNN backbone
        self.cnn = models.resnet18(pretrained=pretrained)
        # ... (kode lain)
        
        # Classifier
        if use_dct:
            self.classifier = nn.Linear(512 + 256, num_classes)
        else:
            self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, img, dct_feat=None):
        cnn_feat = self.cnn(img)
        
        if self.use_dct and dct_feat is not None:
            mlp_feat = self.mlp(dct_feat)
            combined = torch.cat([cnn_feat, mlp_feat], dim=1)
        else:
            combined = cnn_feat
        
        return self.classifier(combined)
```

---

## 🔧 Parameter DCT

### `top_k` (default: 1024)
- Jumlah koefisien DCT yang diambil
- Semakin besar, semakin detail tapi lebih lambat
- Rekomendasi: 512-2048

### `block_size` (default: 8)
- Ukuran blok untuk DCT (seperti JPEG)
- Standar: 8x8
- Jangan diubah kecuali Anda tahu apa yang dilakukan

---

## 📈 Perbandingan Performa

| Method | Speed | Accuracy | Disk Space |
|--------|-------|----------|------------|
| **Precompute DCT** | ⚡⚡⚡ Fast | ✅ High | 📦 ~500MB/10k images |
| **On-the-fly DCT** | 🐢 Slow | ✅ High | 💾 None |
| **No DCT (CNN only)** | ⚡⚡ Medium | ⚠️ Medium | 💾 None |

---

## ✅ Checklist Setup DCT

- [ ] Dataset tersedia di `data/processed/imaginet/subset/`
- [ ] Jalankan `python scripts/precompute_dct.py`
- [ ] Verifikasi folder `data/processed/imaginet/dct_features/` terisi
- [ ] Jalankan `python scripts/precompute_dct.py --verify`
- [ ] Jalankan training: `python scripts/train.py --amp`

---

## 🐛 Troubleshooting

### Error: "No images found"
**Solusi**: Pastikan folder `data/processed/imaginet/subset/` berisi subfolder `real/` dan `fake/`.

### Error: "DCT file not found"
**Solusi**: Jalankan ulang `python scripts/precompute_dct.py --overwrite`.

### Training sangat lambat
**Solusi**: Gunakan precompute DCT, jangan on-the-fly.

### Out of memory saat precompute
**Solusi**: Process gambar secara batch atau reduce `top_k`.

---

## 📚 Referensi

- [DCT dalam JPEG Compression](https://en.wikipedia.org/wiki/JPEG#Discrete_cosine_transform)
- [Forensics using DCT](https://arxiv.org/abs/1811.00661)
- [AI-Generated Image Detection](https://arxiv.org/abs/1912.11035)
