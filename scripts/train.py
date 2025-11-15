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
from datetime import datetime
import subprocess

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
parser.add_argument('--shutdown', action='store_true', help='Shutdown system after training completes')
parser.add_argument('--unfreeze-epoch', type=int, default=10, help='Epoch to unfreeze CNN (default: 10)')
parser.add_argument('--cnn-lr-ratio', type=float, default=0.01, help='CNN LR ratio after unfreeze (default: 0.01)')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
USE_AMP = args.amp
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNFREEZE_EPOCH = args.unfreeze_epoch
CNN_LR_RATIO = args.cnn_lr_ratio

# Create logs directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOG_DIR = PROJECT_ROOT / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Create log file with timestamp
LOG_FILE = LOG_DIR / f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def log_print(message):
    """Print to console and save to log file"""
    print(message)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")

# Set seeds for reproducibility
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

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
log_print("\n" + "="*60)
log_print("📁 LOADING DATASET")
log_print("="*60)

# Create dataloaders with automatic 80/20 split
train_loader, val_loader = create_dataloaders(
    root_dir=DATA_ROOT,
    dct_dir=DCT_DIR if DCT_DIR.exists() else None,
    batch_size=BATCH_SIZE,
    num_workers=args.num_workers,
    train_ratio=0.8,
    seed=args.seed,
    current_epoch=0,  # Will be updated in training loop
    max_epochs=EPOCHS
)

# ================== MODEL ==================
log_print("\n" + "="*60)
log_print("🤖 INITIALIZING MODEL")
log_print("="*60)

# Always start with frozen CNN for stable training
model = HybridDetector(num_classes=2, pretrained=True, freeze_cnn=True).to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
log_print(f"Model: {trainable:,} trainable / {total:,} total params")

if not args.freeze_cnn:
    log_print(f"🔒 CNN will be unfrozen at epoch {UNFREEZE_EPOCH + 1}")
    log_print(f"   CNN LR ratio after unfreeze: {CNN_LR_RATIO}x")
else:
    log_print(f"🔒 CNN will remain frozen (--freeze-cnn flag set)")

# ================== OPTIMIZER + SCHEDULER ==================
# Start with only MLP and classifier trainable
optimizer = optim.Adam([
    {'params': model.mlp.parameters(), 'lr': args.lr},
    {'params': model.classifier.parameters(), 'lr': args.lr}
], lr=args.lr)

scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = GradScaler() if USE_AMP else None
criterion = nn.CrossEntropyLoss()

# Track if CNN has been unfrozen
cnn_unfrozen = False

# ================== RESUME ==================
start_epoch = 0
best_val_acc = 0.0

if args.resume and LATEST_CHECKPOINT.exists():
    log_print(f"\n📂 Resuming from {LATEST_CHECKPOINT}")
    checkpoint = torch.load(LATEST_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint.get('best_val_acc', 0.0)
    cnn_unfrozen = checkpoint.get('cnn_unfrozen', False)
    if 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    log_print(f"✅ Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")
    log_print(f"   CNN unfrozen status: {cnn_unfrozen}")

torch.backends.cudnn.benchmark = True

# ================== TRAINING LOOP ==================
log_print("\n" + "="*60)
log_print("🚀 STARTING TRAINING")
log_print("="*60)
log_print(f"\nConfig:")
log_print(f"   Device: {DEVICE}")
log_print(f"   Batch size: {BATCH_SIZE}")
log_print(f"   Learning rate: {args.lr}")
log_print(f"   Epochs: {EPOCHS}")
log_print(f"   Mixed precision: {USE_AMP}")
log_print(f"   Seed: {args.seed}")
log_print(f"   Unfreeze epoch: {UNFREEZE_EPOCH + 1}")
log_print(f"   CNN LR ratio: {CNN_LR_RATIO}x")
log_print(f"   Shutdown after training: {args.shutdown}")
log_print(f"   Log file: {LOG_FILE}")
log_print("")

for epoch in range(start_epoch, EPOCHS):
    # ========== GRADUAL UNFREEZING ==========
    if epoch == UNFREEZE_EPOCH and not cnn_unfrozen and not args.freeze_cnn:
        log_print("\n" + "="*60)
        log_print(f"🔓 UNFREEZING CNN AT EPOCH {epoch+1}")
        log_print("="*60)
        
        # Unfreeze CNN
        for param in model.cnn.parameters():
            param.requires_grad = True
        
        # Recreate optimizer with CNN parameters
        cnn_lr = args.lr * CNN_LR_RATIO
        optimizer = optim.Adam([
            {'params': model.mlp.parameters(), 'lr': args.lr},
            {'params': model.classifier.parameters(), 'lr': args.lr},
            {'params': model.cnn.parameters(), 'lr': cnn_lr}
        ], lr=args.lr)
        
        # Reset scheduler for remaining epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - epoch)
        
        cnn_unfrozen = True
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log_print(f"   Trainable params: {trainable:,}")
        log_print(f"   MLP/Classifier LR: {args.lr:.2e}")
        log_print(f"   CNN LR: {cnn_lr:.2e}")
        log_print("="*60 + "\n")
    
    # Update dataloaders with current epoch for progressive JPEG
    train_loader.dataset.current_epoch = epoch
    
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
            scaler.unscale_(optimizer)
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    log_print(f"\n📊 Epoch {epoch+1}/{EPOCHS} Summary:")
    log_print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    log_print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    log_print(f"   LR: {scheduler.get_last_lr()[0]:.2e}")
    
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
        'best_val_acc': best_val_acc,
        'cnn_unfrozen': cnn_unfrozen
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
        log_print(f"   🌟 New best validation accuracy: {val_acc:.2f}% - Checkpoint saved!")
    
    log_print("")
    
    # Clear cache every epoch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ================== TRAINING COMPLETE ==================
log_print("\n" + "="*60)
log_print("✅ TRAINING COMPLETE")
log_print("="*60)
log_print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
log_print(f"Checkpoints saved in: {CHECKPOINT_DIR}")
log_print(f"   - Latest: {LATEST_CHECKPOINT.name}")
log_print(f"   - Best: {BEST_CHECKPOINT.name}")
log_print(f"\nLog file saved: {LOG_FILE}")
log_print(f"Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ================== AUTOMATIC SHUTDOWN ==================
if args.shutdown:
    log_print("\n" + "="*60)
    log_print("🔴 SYSTEM SHUTDOWN IN 10 SECONDS...")
    log_print("="*60)
    log_print("Press Ctrl+C to cancel shutdown.")
    
    import time
    try:
        for i in range(10, 0, -1):
            print(f"Shutting down in {i}s...", end='\r')
            time.sleep(1)
        
        log_print("\nInitiating system shutdown...")
        
        # Linux/Mac shutdown
        if sys.platform in ['linux', 'darwin']:
            os.system('sudo shutdown -h now')
        # Windows shutdown
        elif sys.platform == 'win32':
            os.system('shutdown /s /t 0')
    except KeyboardInterrupt:
        log_print("\n⚠️  Shutdown cancelled by user")
        sys.exit(0)
else:
    log_print("\nTo enable automatic shutdown, run with: --shutdown")