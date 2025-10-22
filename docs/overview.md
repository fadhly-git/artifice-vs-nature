# Artifice vs Nature — Struktur & Referensi

## Struktur Folder (ringkasan)
```
artifice-vs-nature/
├── .gitignore
├── README.md
├── LICENSE
├── pyproject.toml / requirements.txt
├── data/
│   ├── raw/              # real/ dan synthetic/ (tidak di-commit)
│   └── processed/        # splits, resized, augmentations
├── datasets/             # dataset wrappers / HF dataset scripts
├── configs/              # yaml / hydra configs per experiment
├── src/
│   └── artifice_vs_nature/
│      ├── __init__.py
│      ├── data.py
│      ├── transforms.py
│      ├── models/
│      ├── eval/
│      ├── train.py
│      └── inference.py
├── experiments/          # per-run artifacts (exp-YYYYMMDD_tag/)
├── models/               # pointers ke model registry (HF/S3) atau small demo checkpoints
├── notebooks/
├── results/
│   ├── figures/
│   ├── logs/
│   └── metrics/
├── web_demo/             # streamlit / fastapi demo
├── tests/
└── docs/
```

---

## Tabel Referensi: Paper & Ringkasan Metodologi

| No | Paper (year) / Venue | Model / Arsitektur | Metodologi / Langkah-langkah | Dataset (train / eval) | Hasil / Performa |
|----|----------------------|--------------------|------------------------------|------------------------|------------------|
| 1 | Lu et al. — "Seeing is not always believing" (NeurIPS 2023) | Benchmark framework (no single detector) | 1) Koleksi 2M fake images dari 21 generator 2) Membuat 11 subset evaluasi MPBench 3) Bandingkan human vs model perception | Fake2M (2M train) + MPBench (11 val sets) | Baseline benchmark untuk deteksi gambar AI |
| 2 | Cao et al. — "HyperDet" (2024) | CLIP backbone + Hypernetwork + Mixture of LoRA experts | 1) Group SRM filters jadi 5 grup 2) Hypernetwork generate LoRA weights 3) Merge multiple LoRA networks (ensemble) 4) Loss balancing pixel & semantic artifacts | Fake2M + UnivFD | SOTA: +5.03% acc, +10.02 mAP vs baseline di Fake2M |
| 3 | Diao et al. — "Adversarial Attack" (2024) | Detector trained on Fake2M + attack pipeline | 1) Train detector on Fake2M 2) Implement Fast Projected Binary Attack (FPBA) 3) Eval robustness under adversarial perturbations | Fake2M (training detector) | Akurasi turun >30% setelah serangan FPBA |
| 4 | Xu et al. — "FAMSeC" (2024) | CLIP ViT-L/14 + LoRA-based Forgery Awareness Module (FAM) | 1) Dual CLIP: fixed + trainable 2) LoRA-based FAM untuk fine-tuning 3) Semantic feature-guided Contrastive learning (SeC) 4) Train hanya menggunakan 0.56% samples dari Fake2M | Subset Fake2M (≈4k) + ForenSynths | 95.22% accuracy, +14.55% vs SOTA dengan 0.56% data |
| 5 | Tan et al. — "NPR" (2023) | CNN-based, fokus pada up-sampling operations | 1) Analisis up-sampling in generative CNNs 2) Rethink architectural components 3) Cross-generator evaluation on unseen diffusion domain | Fake2M (dipakai sebagai unseen diffusion test domain) | Baseline comparison untuk HyperDet |
| 6 | Cheng et al. — "CO-SPY" (CVPR 2025) | Hybrid semantic + pixel feature extractor (dual-stream) | 1) Gabungkan semantic features & pixel-level features 2) Multi-scale feature fusion 3) Eval pada Co-Spy-Bench (22 generators) | Fake2M + 21 other generators in Co-Spy-Bench | Fokus: robustness across multiple generators |
| 7 | Wu et al. — "Diffusion Timestep Ensembling" (2025) | Diffusion-based features + timestep ensembling | 1) Extract features from multiple diffusion timesteps 2) Ensemble predictions across timesteps 3) Timestep analysis untuk explainability | Subset MPBench-SDXL + Fake2M-trained baseline | 95.9% accuracy pada MPBench-SDXL |
| 8 | Ştefan et al. — "Deepfake Sentry" (2024) | Ensemble of detectors (Xception, others) + fingerprint AE | 1) Train multiple detector models 2) Ensemble intelligence untuk robust 3) Data augmentation (100k samples dari Fake2M) 4) Multi-model consensus | Fake2M (100k augment) + multiple other datasets | Enhanced generalization via ensemble |

---

## Ekstraksi Fitur (ringkasan per-paper)

1. Ekstraksi Fitur Berbasis Deep Learning
   - Lu et al. (2023): CLIP embeddings, ViT features, CNN features
   - Cao et al. (2024): CLIP-ViT features + SRM filter features (30 filter dibagi 5 grup)
   - Xu et al. (2024): CLIP ViT-L/14 features dengan LoRA adaptation
   - Cheng et al. (2025): Dual-stream → CLIP semantic features + VAE pixel artifact features
   - Wu et al. (2025): CLIP encoder features dari multiple diffusion timesteps

2. Ekstraksi Fitur Khusus / Hand-crafted
   - Tan et al. (2023): NPR (Neighboring Pixel Relationship) — korelasi piksel tetangga untuk deteksi artefak up-sampling
   - Ştefan et al. (2024): Xception CNN features + autoencoder-generated "fingerprint" features

3. Analisis Adversarial (butuh fitur juga)
   - Diao et al. (2024): Analisis fitur dari ResNet-50, EfficientNet, Swin-T, CLIP yang sudah dilatih; fokus pada robustness terhadap attack (FPBA)

---

## Template Metadata untuk tiap gambar (direkomendasikan)
Simpan metadata per gambar dalam CSV/JSON (lihat fields contoh):
- id, filename, split, source (real|synthetic), generator_name, generator_version, prompt (if any), seed, sampling_params, original_url, license, collected_date, experiment_id

---

## Notes & Praktik Baik
- Jangan commit data mentah atau model besar ke git; gunakan DVC / Git LFS / remote storage.
- Simpan prompt & generator config untuk setiap synthetic image untuk auditability.
- Simpan seed, commit-hash, dan environment (requirements.txt) di setiap eksperimen.
- Siapkan skrip otomatis: scripts/generate_gallery.py untuk membuat side-by-side galleries; scripts/eval_all.py untuk run metrik (FID/KID/CLIP-score).
- Untuk human evaluation, simpan anotator IDs anonim, instruksi, dan hasil annotasi terpisah di `data/metadata/human_eval/`.

---

## Cara pakai file ini
- Tempelkan file ini ke root repo (`artifice_vs_nature.md`) sebagai "living document" referensi eksperimen.
- Gunakan tabel di atas sebagai checklist ketika menambah paper baru: isi kolom model, metodologi, dataset, dan hasil.
- Jika mau, konversi tabel ini ke CSV/JSON untuk diimpor ke eksperimen tracker (W&B / MLflow / SQLite).
