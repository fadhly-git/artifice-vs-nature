# src/train_rx580_albumentations.py
# PyTorch 1.13 + ROCm 5.4.3 + RX 580 + Albumentations 1.3.1 → PERFECT

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import datasets
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from pathlib import Path
from PIL import Image

# ============== CONFIG ==============
DATA_ROOT = Path("data/raw/imaginet/subset")
CHECKPOINT_DIR = Path("models/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CFG = {
    'model': 'efficientnet_b0',
    'img_size': 380,
    'batch_size': 16,
    'epochs': 50,
    'lr': 3e-4,
    'num_workers': 4,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0)}")

# ============== ALBUMENTATIONS (LEBIH BAGUS!) ==============
train_transform = A.Compose([
    A.RandomResizedCrop(CFG['img_size'], CFG['img_size'], scale=(0.6, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.HueSaturationValue(p=0.4),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.CoarseDropout(max_holes=12, max_height=32, max_width=32, p=0.4),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(CFG['img_size'], CFG['img_size']),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ============== CUSTOM DATASET UNTUK ALBUMENTATIONS ==============
class AlbumentationsDataset(Dataset):
    def __init__(self, root, transform=None):
        self.data = datasets.ImageFolder(root)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data.samples[idx]
        image = Image.open(path).convert('RGB')
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label

# Load dataset
full_dataset = AlbumentationsDataset(DATA_ROOT)

# Split FIRST
val_size = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Hitung weight karena 2:1 - USE TRAIN SET ONLY
train_labels = [full_dataset.data.samples[idx][1] for idx in train_set.indices]
num_real = train_labels.count(1)
num_fake = train_labels.count(0)
print(f"Train - Real: {num_real} | Fake: {num_fake}")

weights = [1.0/num_fake if full_dataset.data.samples[idx][1] == 0 else 1.0/num_real 
           for idx in train_set.indices]
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

# Set transforms AFTER split
train_set.dataset.transform = train_transform
val_set.dataset.transform = val_transform

train_loader = DataLoader(train_set, batch_size=CFG['batch_size'], sampler=sampler,
                          num_workers=CFG['num_workers'], pin_memory=True)
val_loader = DataLoader(val_set, batch_size=CFG['batch_size'], shuffle=False,
                        num_workers=CFG['num_workers'], pin_memory=True)

# ============== MODEL & TRAINING (sama seperti sebelumnya) ==============
model = timm.create_model(CFG['model'], pretrained=True, num_classes=2).to(device)

class_weights = torch.tensor([1.0, num_fake/num_real], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=1e-5)

best_acc = 0.0
total_steps = CFG['epochs'] * len(train_loader)
step = 0

for epoch in range(CFG['epochs']):
    model.train()
    # bookkeeping for training metrics
    train_correct = 0
    train_total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG['epochs']}")
    for x, y in pbar:
        step += 1
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update training accuracy counters
        batch_preds = pred.argmax(1)
        train_correct += (batch_preds == y).sum().item()
        train_total += y.size(0)

        lr = 3e-4 * 0.5 * (1 + np.cos(step / total_steps * np.pi))
        for g in optimizer.param_groups:
            g['lr'] = lr

        # show running training accuracy in the progress bar
        train_acc_running = train_correct / train_total if train_total > 0 else 0.0
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.2e}', 'train_acc': f'{train_acc_running:.4f}'})

    # Validation
    model.eval()
    preds, probs, trues = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            out = model(x)
            prob = torch.softmax(out, 1)[:, 0].cpu().numpy()  # prob fake
            pred = out.argmax(1).cpu().numpy()
            probs.extend(prob)
            preds.extend(pred)
            trues.extend(y.numpy())

    acc = accuracy_score(trues, preds)
    auc = roc_auc_score(trues, probs)
    prec = precision_score(trues, preds, zero_division=0)
    rec = recall_score(trues, preds, zero_division=0)
    f1 = f1_score(trues, preds, zero_division=0)

    print(f"\nEpoch {epoch+1} → Acc: {acc:.5f} | AUC: {auc:.5f} | Prec: {prec:.5f} | Rec: {rec:.5f} | F1: {f1:.5f}\n")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), CHECKPOINT_DIR / "best_effnetb3_albumentations.pth")
        print(f"NEW BEST → {best_acc:.5f}\n")

print("SELESAI! Best accuracy:", best_acc)