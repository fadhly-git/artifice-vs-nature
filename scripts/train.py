import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import sys
import argparse
from tqdm import tqdm
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# ================== ARGUMENTS ==================
parser = argparse.ArgumentParser(description='Train Hybrid Detector with on-the-fly preprocessing')
parser.add_argument('--batch-size', type=int, default=4, help='Batch size per step')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers (use 0 for ROCm)')
parser.add_argument('--freeze-cnn', action='store_true', help='Freeze CNN backbone')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
USE_AMP = args.amp
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds for reproducibility
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import create_dataloaders
from src.models.hybrid import HybridDetector

# ================== PATHS ==================
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "imaginet" / "subset"
DCT_DIR = PROJECT_ROOT / "data" / "processed" / "imaginet" / "dct_features"  # Optional
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

LATEST_CHECKPOINT = CHECKPOINT_DIR / "hybrid_imaginet_latest.pth"
BEST_CHECKPOINT = CHECKPOINT_DIR / "hybrid_imaginet_best.pth"

# ================== DATASET ==================
print("\n" + "="*60)
print("📁 LOADING DATASET")
print("="*60)

# Create dataloaders with automatic 80/20 split
train_loader, val_loader = create_dataloaders(
    root_dir=DATA_ROOT,
    dct_dir=DCT_DIR if DCT_DIR.exists() else None,
    batch_size=BATCH_SIZE,
    num_workers=args.num_workers,
    train_ratio=0.8,
    seed=args.seed
)

# ================== MODEL ==================
print("\n" + "="*60)
print("🤖 INITIALIZING MODEL")
print("="*60)

model = HybridDetector(num_classes=2, pretrained=True, freeze_cnn=args.freeze_cnn).to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Model: {trainable:,} trainable / {total:,} total params")

# ================== OPTIMIZER + SCHEDULER ==================
optimizer = optim.Adam([
    {'params': model.mlp.parameters(), 'lr': args.lr},
    {'params': model.classifier.parameters(), 'lr': args.lr},
    {'params': model.cnn.parameters(), 'lr': args.lr / 10}
], lr=args.lr)

scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = GradScaler() if USE_AMP else None
criterion = nn.CrossEntropyLoss()

# ================== RESUME ==================
start_epoch = 0
best_val_acc = 0.0

if args.resume and LATEST_CHECKPOINT.exists():
    print(f"\n📂 Resuming from {LATEST_CHECKPOINT}")
    checkpoint = torch.load(LATEST_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint.get('best_val_acc', 0.0)
    if 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    print(f"✅ Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")

torch.backends.cudnn.benchmark = True

# ================== TRAINING LOOP ==================
print("\n" + "="*60)
print("🚀 STARTING TRAINING")
print("="*60)
print(f"\nConfig:")
print(f"   Device: {DEVICE}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Learning rate: {args.lr}")
print(f"   Epochs: {EPOCHS}")
print(f"   Mixed precision: {USE_AMP}")
print(f"   Seed: {args.seed}")
print()

for epoch in range(start_epoch, EPOCHS):
    # ========== TRAINING PHASE ==========
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f"[TRAIN] Epoch {epoch+1}/{EPOCHS}", ncols=100)
    
    for img_masked, dct_feat, labels in pbar:
        img_masked = img_masked.to(DEVICE, non_blocking=True)
        dct_feat = dct_feat.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast(enabled=USE_AMP):
            outputs = model(img_masked, dct_feat)
            loss = criterion(outputs, labels)
        
        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        train_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*train_correct/train_total:.2f}%'
        })
    
    train_loss = train_loss / train_total
    train_acc = 100. * train_correct / train_total
    
    # ========== VALIDATION PHASE ==========
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        pbar_val = tqdm(val_loader, desc=f"[VAL]   Epoch {epoch+1}/{EPOCHS}", ncols=100)
        
        for img_masked, dct_feat, labels in pbar_val:
            img_masked = img_masked.to(DEVICE, non_blocking=True)
            dct_feat = dct_feat.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            with autocast(enabled=USE_AMP):
                outputs = model(img_masked, dct_feat)
                loss = criterion(outputs, labels)
            
            val_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            
            pbar_val.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*val_correct/val_total:.2f}%'
            })
    
    val_loss = val_loss / val_total
    val_acc = 100. * val_correct / val_total
    
    # Update learning rate
    scheduler.step()
    
    # ========== LOGGING ==========
    print(f"\n📊 Epoch {epoch+1}/{EPOCHS} Summary:")
    print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    print(f"   LR: {scheduler.get_last_lr()[0]:.2e}")
    
    # ========== SAVE CHECKPOINTS ==========
    # Save latest checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'best_val_acc': best_val_acc
    }, LATEST_CHECKPOINT)
    
    # Save best checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss
        }, BEST_CHECKPOINT)
        print(f"   🌟 New best validation accuracy: {val_acc:.2f}% - Checkpoint saved!")
    
    print()
    
    # Clear cache every epoch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ================== TRAINING COMPLETE ==================
print("\n" + "="*60)
print("✅ TRAINING COMPLETE")
print("="*60)
print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
print(f"Checkpoints saved in: {CHECKPOINT_DIR}")
print(f"   - Latest: {LATEST_CHECKPOINT.name}")
print(f"   - Best: {BEST_CHECKPOINT.name}")