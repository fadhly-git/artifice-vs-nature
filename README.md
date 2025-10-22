# Artifice vs Nature

Proyek klasifikasi gambar untuk membedakan antara gambar buatan (artifice) dan gambar alami (nature) menggunakan data dari Hugging Face.

## Struktur Folder

```
artifice-vs-nature.git/
├── data/
│   ├── raw/           # Data mentah dari Hugging Face
│   └── processed/     # Data yang sudah diproses
│
├── src/               # Source code Python
│
├── models/            # Model weights yang sudah dilatih
│   └── checkpoints/   # Model checkpoints
│
├── notebooks/         # Jupyter notebooks untuk eksplorasi
│
├── results/           # Hasil training dan evaluasi
│   ├── figures/       # Visualisasi (plots, confusion matrix)
│   ├── logs/          # Training logs
│   └── metrics/       # Metrics (JSON files)
│
└── configs/           # File konfigurasi
```

## Dataset

Dataset diambil dari Hugging Face:
```python
from datasets import load_dataset
dataset = load_dataset("nama-dataset")
```

## Usage

1. Download dataset ke folder `data/raw/`
2. Tulis code training di folder `src/`
3. Simpan model di folder `models/`
4. Hasil akan tersimpan di folder `results/`
