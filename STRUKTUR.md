# Struktur Folder - Artifice vs Nature

## Overview
```
artifice-vs-nature.git/
├── 📁 data/
│   ├── 📁 raw/
│   └── 📁 processed/
├── 📁 src/
├── 📁 models/
│   └── 📁 checkpoints/
├── 📁 notebooks/
├── 📁 results/
│   ├── 📁 figures/
│   ├── 📁 logs/
│   └── 📁 metrics/
└── 📁 configs/
```

## Penjelasan

### 📁 data/
Tempat menyimpan dataset
- **raw/** → Data mentah dari Hugging Face (tidak diubah)
- **processed/** → Data yang sudah diproses (split train/val/test, augmented, dll)

### 📁 src/
Semua source code Python Anda
- Script untuk load data
- Script untuk training model
- Script untuk evaluasi
- Utility functions

### 📁 models/
Model yang sudah dilatih
- **checkpoints/** → Checkpoint model saat training
- File .pth atau .pt untuk model weights

### 📁 notebooks/
Jupyter notebooks untuk:
- Eksplorasi data
- Visualisasi
- Prototyping
- Eksperimen

### 📁 results/
Output dari training dan evaluasi
- **figures/** → Grafik training loss/accuracy, confusion matrix, dll
- **logs/** → Log file dari training
- **metrics/** → File JSON berisi metrics (accuracy, precision, recall, dll)

### 📁 configs/
File konfigurasi (YAML/JSON)
- Hyperparameters
- Model settings
- Path settings

## Workflow

1. **Download data** → Simpan di `data/raw/`
2. **Eksplorasi** → Buat notebook di `notebooks/`
3. **Preprocess** → Hasil ke `data/processed/`
4. **Training** → Code di `src/`, model di `models/`
5. **Evaluasi** → Hasil di `results/`
