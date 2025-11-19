# 🚀 Lightweight Model Architecture

## Perubahan dari Model Lama

### ❌ **Model Lama (EfficientNet-B0)**
```python
# Arsitektur berat:
- CNN: EfficientNet-B0 (5.3M params)
- MLP: 1024 → 512 → 256 (LayerNorm + Dropout 0.3)
- Classifier: 1536 → 512 → 256 → 2 (LayerNorm + Dropout 0.5)
- Total: ~6.5M parameters
```

**Masalah:**
- ❌ Parameter banyak = risiko gradient explosion tinggi
- ❌ Memory usage besar = batch size kecil
- ❌ Training lambat
- ❌ Prone to NaN saat unfreeze

---

### ✅ **Model Baru (MobileNetV3-Small) - DEFAULT**
```python
# Arsitektur ringan:
- CNN: MobileNetV3-Small (2.5M params, 576 dim)
- MLP: 1024 → 256 → 128 (BatchNorm + Dropout 0.2)
- Classifier: 704 → 256 → 128 → 2 (BatchNorm + Dropout 0.3)
- Total: ~3.2M parameters
```

**Keuntungan:**
- ✅ **50% lebih sedikit parameter** = lebih stabil
- ✅ **2-3x lebih cepat** training
- ✅ **Batch size bisa 2x lebih besar** (16-32 vs 8)
- ✅ **Risiko NaN lebih rendah**
- ✅ **Tetap akurat** untuk binary classification

---

## 📊 Perbandingan Arsitektur

| Model | Params | CNN Dim | Speed | Memory | Stability |
|-------|--------|---------|-------|--------|-----------|
| **mobilenetv3_small_100** | 3.2M | 576 | ⚡⚡⚡ | 💚💚💚 | ⭐⭐⭐ |
| mobilenetv3_large_100 | 4.5M | 960 | ⚡⚡ | 💚💚 | ⭐⭐ |
| efficientnet_lite0 | 5.8M | 1280 | ⚡ | 💚 | ⭐ |
| efficientnet_b0 | 6.5M | 1280 | 🐢 | ⚠️ | ⚠️ |

---

## 🚀 Cara Menggunakan

### 1. **Default (Recommended - MobileNetV3-Small)**
```bash
python scripts/train.py \
  --batch-size 16 \
  --lr 5e-5 \
  --epochs 50 \
  --freeze-cnn \
  --grad-clip 1.0 \
  --amp
```

**Keuntungan:**
- Batch size bisa 16 (2x lebih besar dari EfficientNet)
- Training ~2x lebih cepat
- Risiko NaN minimal

---

### 2. **Balanced (MobileNetV3-Large)**
```bash
python scripts/train.py \
  --model mobilenetv3_large_100 \
  --batch-size 12 \
  --lr 5e-5 \
  --epochs 50 \
  --freeze-cnn \
  --grad-clip 1.0 \
  --amp
```

**Keuntungan:**
- Lebih kuat dari Small tapi tetap ringan
- Good balance antara speed dan accuracy

---

### 3. **Heavy (EfficientNet-B0 - Original)**
```bash
python scripts/train.py \
  --model efficientnet_b0 \
  --batch-size 8 \
  --lr 5e-5 \
  --epochs 50 \
  --freeze-cnn \
  --grad-clip 1.0 \
  --amp
```

**⚠️ Warning:**
- Lebih lambat
- Memory usage tinggi
- Risiko NaN lebih besar
- Hanya gunakan jika accuracy Small/Large tidak cukup

---

## 🔄 Perubahan Arsitektur Detail

### MLP Branch (DCT Features)
```python
# OLD:
nn.Linear(1024, 512), nn.LayerNorm(512), nn.ReLU()
nn.Dropout(0.3)
nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU()
# Output: 256 dim

# NEW:
nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU()
nn.Dropout(0.2)  # Reduced dropout
nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU()
# Output: 128 dim
```

**Kenapa BatchNorm instead of LayerNorm?**
- ✅ BatchNorm lebih stabil untuk small batches
- ✅ Lebih cepat (built-in CUDA optimization)
- ✅ Reduce internal covariate shift

---

### Classifier Branch
```python
# OLD (untuk EfficientNet-B0):
fusion_dim = 1280 + 256 = 1536
nn.Linear(1536, 512), nn.LayerNorm(512), nn.ReLU()
nn.Dropout(0.5)
nn.Linear(512, 256), nn.ReLU()
nn.Dropout(0.3)
nn.Linear(256, 2)

# NEW (untuk MobileNetV3-Small):
fusion_dim = 576 + 128 = 704
nn.Linear(704, 256), nn.BatchNorm1d(256), nn.ReLU()
nn.Dropout(0.3)  # Reduced dropout
nn.Linear(256, 128), nn.ReLU()
nn.Dropout(0.2)  # Reduced dropout
nn.Linear(128, 2)
```

**Perubahan:**
- ✅ Dimensi lebih kecil = lebih stabil
- ✅ Dropout dikurangi (0.5→0.3, 0.3→0.2) = less aggressive regularization
- ✅ BatchNorm untuk stability

---

## 📈 Expected Results

### MobileNetV3-Small (Predicted)
```
Epoch 10: Val Acc ~85-87%
Epoch 20: Val Acc ~88-90%
Epoch 30: Val Acc ~90-92%
Epoch 50: Val Acc ~91-93%
```

**Training Speed:**
- ~25-30 seconds per epoch (vs ~45 seconds untuk EfficientNet-B0)
- Total training time: ~20-25 minutes untuk 50 epochs

---

### EfficientNet-B0 (Reference)
```
Epoch 10: Val Acc ~86%
Epoch 20: Val Acc ~89%
Epoch 27: Val Acc ~90.30% (best)
Epoch 29: CRASH (NaN loss)
```

**Training Speed:**
- ~42-45 seconds per epoch
- Total training time: ~35-40 minutes untuk 50 epochs

---

## 🎯 Rekomendasi

### Untuk Development/Testing:
```bash
python scripts/train.py \
  --model mobilenetv3_small_100 \
  --batch-size 32 \
  --lr 1e-4 \
  --epochs 20 \
  --grad-clip 1.0 \
  --amp
```
- Fast iteration
- Low memory
- Good for debugging

---

### Untuk Production/Final Model:
```bash
python scripts/train.py \
  --model mobilenetv3_large_100 \
  --batch-size 16 \
  --lr 5e-5 \
  --epochs 80 \
  --freeze-cnn \
  --unfreeze-epoch 20 \
  --unfreeze-all-epoch 999 \
  --grad-clip 1.0 \
  --amp
```
- Best balance
- Stable training
- Good accuracy

---

## 🔧 Migration Guide

### Jika Anda sudah punya checkpoint EfficientNet-B0:
```bash
# 1. Checkpoint lama tidak compatible dengan model baru
# 2. Mulai training dari awal dengan model baru
python scripts/train.py --model mobilenetv3_small_100 --batch-size 16

# 3. Atau tetap gunakan EfficientNet tapi dengan fix NaN:
python scripts/train.py \
  --model efficientnet_b0 \
  --batch-size 8 \
  --grad-clip 0.5 \
  --unfreeze-all-epoch 999 \
  --resume
```

---

## 📊 Benchmark Comparison

Run this to compare all models:
```bash
python scripts/compare_models.py
```

Output akan menunjukkan:
- Parameter count
- Memory usage
- Forward pass speed
- Recommendations

---

## 💡 Tips & Tricks

### 1. **Batch Size Guidelines**
```python
mobilenetv3_small_100:  16-32 (recommended: 16)
mobilenetv3_large_100:  12-20 (recommended: 16)
efficientnet_lite0:     8-12  (recommended: 10)
efficientnet_b0:        4-8   (recommended: 8)
```

### 2. **Learning Rate Guidelines**
```python
Small/Large models: 1e-4 to 5e-5
Heavy models: 5e-5 to 1e-5
```

### 3. **Gradient Clipping**
```python
Always use: --grad-clip 1.0
For extra safety: --grad-clip 0.5
```

### 4. **Unfreezing Strategy**
```python
# Conservative (safest):
--unfreeze-epoch 20 --unfreeze-all-epoch 999

# Moderate:
--unfreeze-epoch 15 --unfreeze-all-epoch 40

# Aggressive (risky):
--unfreeze-epoch 10 --unfreeze-all-epoch 25
```

---

## ✅ Kesimpulan

**Model baru (MobileNetV3-Small) adalah default karena:**
1. ✅ 50% lebih sedikit parameter
2. ✅ 2x lebih cepat training
3. ✅ Lebih stabil (risiko NaN rendah)
4. ✅ Batch size bisa lebih besar
5. ✅ Tetap accurate untuk binary classification

**Kapan menggunakan EfficientNet-B0?**
- Jika accuracy dengan MobileNet tidak cukup (< 90%)
- Jika punya GPU besar (>8GB VRAM)
- Jika tidak masalah dengan training lambat

**Untuk kebanyakan kasus, MobileNetV3-Small/Large sudah lebih dari cukup!** 🚀
