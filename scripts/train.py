import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import timm
import sys
import argparse
from tqdm import tqdm
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# ================== ARGUMENTS ==================
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--accum-steps', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--amp', action='store_true', default=True)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--subset', type=float, default=None)
parser.add_argument('--freeze-cnn', action='store_true')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
ACCUM_STEPS = args.accum_steps
EPOCHS = args.epochs
USE_AMP = args.amp
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import HybridDataset
from src.models.hybrid import HybridDetector

# ================== PATHS ==================
FAKE_DIR = PROJECT_ROOT / "data" / "processed" / "imaginet" / "subset_pt" / "fake"
REAL_DIR = PROJECT_ROOT / "data" / "processed" / "imaginet" / "subset_pt" / "real"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

LATEST_CHECKPOINT = CHECKPOINT_DIR / "hybrid_imaginet_latest.pth"
BEST_CHECKPOINT = CHECKPOINT_DIR / "hybrid_imaginet_best.pth"

# ================== DATASET ==================
fake_dataset = HybridDataset(FAKE_DIR)
real_dataset = HybridDataset(REAL_DIR)

if args.subset is not None and 0.0 < args.subset < 1.0:
    import random
    from torch.utils.data import Subset
    fake_indices = list(range(len(fake_dataset)))
    real_indices = list(range(len(real_dataset)))
    random.shuffle(fake_indices)
    random.shuffle(real_indices)
    fake_size = int(len(fake_dataset) * args.subset)
    real_size = int(len(real_dataset) * args.subset)
    fake_dataset = Subset(fake_dataset, fake_indices[:fake_size])
    real_dataset = Subset(real_dataset, real_indices[:real_size])

full_dataset = ConcatDataset([real_dataset, fake_dataset])

print(f"\nDataset Info:")
print(f"   Fake: {len(fake_dataset)} | Real: {len(real_dataset)} | Total: {len(full_dataset)}")
if args.subset:
    print(f"   Using {args.subset*100:.0f}% subset")

dataloader = DataLoader(
    full_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True if torch.cuda.is_available() else False,
    drop_last=True,
    persistent_workers=False,
    prefetch_factor=2
)

# ================== MODEL ==================
model = HybridDetector(num_classes=2, pretrained=True, freeze_cnn=args.freeze_cnn).to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {trainable:,} trainable params")

# ================== OPTIMIZER + SCHEDULER ==================
optimizer = optim.Adam([
    {'params': model.mlp.parameters(), 'lr': args.lr},           # DCT head
    {'params': model.classifier.parameters(), 'lr': args.lr},    # Classifier
    {'params': model.cnn.parameters(), 'lr': args.lr / 10}       # CNN: 10x lebih kecil
], lr=args.lr)

scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)  # 25 epoch

scaler = GradScaler() if USE_AMP else None

# ================== RESUME ==================
start_epoch = 0
best_acc = 0.0

if args.resume and LATEST_CHECKPOINT.exists():
    print(f"Resuming from {LATEST_CHECKPOINT}")
    checkpoint = torch.load(LATEST_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint.get('best_acc', 0.0)
    # Reload scheduler state
    if 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    print(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.2f}%")

torch.backends.cudnn.benchmark = True

# ================== TRAINING LOOP ==================
for epoch in range(start_epoch, EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=100)

    for batch_idx, (img_masked, dct_feat, labels) in enumerate(pbar):
        img_masked = img_masked.to(DEVICE, non_blocking=True)
        dct_feat = dct_feat.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with autocast(enabled=USE_AMP):
            outputs = model(img_masked, dct_feat)
            loss = nn.CrossEntropyLoss()(outputs, labels) / ACCUM_STEPS

        if USE_AMP:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % ACCUM_STEPS == 0 or (batch_idx + 1) == len(dataloader):
            if USE_AMP:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * ACCUM_STEPS
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    pbar.close()

    # Step scheduler
    scheduler.step()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'best_acc': best_acc
    }
    torch.save(checkpoint, LATEST_CHECKPOINT)

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(checkpoint, BEST_CHECKPOINT)
        print(f"NEW BEST: {best_acc:.2f}%")

    print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")