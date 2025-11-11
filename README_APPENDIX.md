
### ImagiNet Dataset

Dataset menggunakan **ImagiNet** - koleksi gambar AI-generated dan real.

**Location:** `data/processed/imaginet/subset/`

**Structure:**
```
imaginet/subset/
├── fake/
│   └── ffhq_stylegan/    # StyleGAN-generated faces
└── real/
    └── ffhq/             # Real human faces (FFHQ dataset)
```

**Statistics:**
- Real images: FFHQ subset
- Fake images: StyleGAN-generated (FFHQ-based)
- Image size: 1024×1024 PNG
- Total samples: Varies by subset

---

## 🧪 Testing

### Test Individual Modules

```bash
python src/preprocessing/jpeg.py
python src/preprocessing/resize.py
python src/preprocessing/augment.py
python src/preprocessing/normalize.py
python src/preprocessing/mask.py
python src/preprocessing/dct.py
python src/preprocessing/pipeline.py
```

### Test in Notebook

Run all cells in `notebooks/preprocessing_demo.ipynb`

---

## 🐛 Troubleshooting

### Import Error: Cannot import preprocessing modules

**Solution:**
```python
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
```

### GPU Not Detected

**Check ROCm:**
```bash
rocm-smi
```

**Verify PyTorch CUDA:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### NumPy Version Conflict

Warning tentang numpy version dengan pandas adalah normal. PyTorch 1.7.0 memerlukan numpy<1.20, sementara pandas terbaru memerlukan numpy>=1.20.3.

**Option 1:** Keep numpy 1.19.x (PyTorch compatibility)
**Option 2:** Upgrade numpy (pandas compatibility, may affect PyTorch)

---

## 📈 Next Steps

1. ✅ **Setup Complete** - Preprocessing pipeline verified
2. 🔄 **Batch Processing** - Process entire ImagiNet dataset
3. 🤖 **Model Development** - Build CNN classifier
4. 🎯 **Training** - Train on preprocessed data
5. 📊 **Evaluation** - Test on validation set
6. 🚀 **Deployment** - Export model for inference

---

## 📝 Citation

If you use this project, please cite:

```bibtex
@misc{artifice-vs-nature-2025,
  author = {Fadhly},
  title = {Artifice vs Nature: AI-Generated Image Detection},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/fadhly-git/artifice-vs-nature}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **PyTorch Team** - Deep learning framework
- **AMD ROCm** - GPU compute platform
- **ImagiNet** - Dataset for AI image detection
- **FFHQ** - High-quality face dataset

---

## 📬 Contact

- **Author:** Fadhly
- **GitHub:** [@fadhly-git](https://github.com/fadhly-git)
- **Project:** [artifice-vs-nature](https://github.com/fadhly-git/artifice-vs-nature)

---

**Last Updated:** November 11, 2025  
**Version:** 0.1.0  
**Status:** ✅ Setup Complete & Verified
