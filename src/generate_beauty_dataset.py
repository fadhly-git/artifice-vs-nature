# src/generate_beauty_dataset.py
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import random

DATA_ROOT = Path("data/raw/imaginet/subset")
OUT_ROOT = DATA_ROOT.parent / "subset_beauty"
OUT_ROOT.mkdir(exist_ok=True)

for split in ["real", "fake"]:
    (OUT_ROOT / split).mkdir(exist_ok=True)

def beauty_filter(img):
    # 5 filter agresif yang mirip FaceApp/BeautyPlus/TikTok
    img = cv2.bilateralFilter(img, d=9, sigmaColor=85, sigmaSpace=85)
    img = cv2.GaussianBlur(img, (0,0), 2)
    alpha = 1.4
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=30)
    # skin smoothing + whitening
    if random.random() > 0.5:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img

for folder in ["real", "fake"]:
    src = DATA_ROOT / folder
    dst = OUT_ROOT / folder
    files = list(src.glob("*.jpg")) + list(src.glob("*.png"))
    print(f"Processing {folder}: {len(files)} images → beauty version")
    
    for path in tqdm(files):
        img = cv2.imread(str(path))
        if img is None: continue
        img = cv2.resize(img, (512, 512))
        
        for i in range(5):  # 5 variasi filter per gambar
            aug = beauty_filter(img.copy())
            # tambah sedikit noise + compression biar mirip Instagram
            noise = np.random.normal(0, random.randint(8,25), aug.shape).astype(np.uint8)
            aug = cv2.add(aug, noise)
            _, buf = cv2.imencode(".jpg", aug, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(55,85)])
            out_path = dst / f"{path.stem}_{i:02d}.jpg"
            cv2.imwrite(str(out_path), aug)

print("Dataset beauty selesai! → data/raw/imaginet/subset_beauty/real & fake")