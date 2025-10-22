from datasets import load_dataset
from tqdm import tqdm
import os

# Target direktori output
OUT_DIR = "data/raw"
os.makedirs(f"{OUT_DIR}/real", exist_ok=True)
os.makedirs(f"{OUT_DIR}/synthetic", exist_ok=True)

# Ambil subset 10% dari train split dan 10% dari validation
train_subset = load_dataset(
    "InfImagine/FakeImageDataset",
    split="train[:10%]"
)
val_subset = load_dataset(
    "InfImagine/FakeImageDataset",
    split="validation[:5%]"
)

# Gabungkan keduanya
dataset = train_subset.concatenate(val_subset)

print(f"Total subset images: {len(dataset)}")

# Simpan gambar berdasarkan label
for item in tqdm(dataset, desc="Saving images"):
    label = item["label"]  # 'real' atau 'fake'
    img = item["image"]
    fname = f"{item['image_id']}.png"

    if label == "real":
        img.save(f"{OUT_DIR}/real/{fname}")
    else:
        img.save(f"{OUT_DIR}/synthetic/{fname}")

print("✅ Subset dataset berhasil disimpan di data/raw/")
