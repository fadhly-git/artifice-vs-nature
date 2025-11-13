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

ssl._create_default_https_context = ssl._create_unverified_context

# Create log file with timestamp
LOG_FILE = Path("results") / f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def log_print(message):
    """Print to terminal and save to log file (without tqdm interference)"""
    print(message)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")

# ================== ARGUMENTS ==================
parser = argparse.ArgumentParser(description='Train Hybrid Detector with on-the-fly preprocessing')
parser.add_argument('--batch-size', type=int, default=4, help='Batch size per step')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--epoch', type=int, dest='epochs', help='Alias for --epochs')
parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
parser.add_argument('--amp', type=lambda x: str(x).lower() in ('true','1','yes','y'), default=True, help='Use mixed precision (True/False)')
parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers (use 0 for ROCm)')
parser.add_argument('--freeze-cnn', action='store_true', help='Freeze CNN backbone')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--shutdown', action='store_true', help='Shutdown system after training completes')
parser.add_argument('--accum-steps', type=int, default=1, help='Gradient accumulation steps')
args = parser.parse_args()

ACCUM_STEPS = args.accum_steps
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
    seed=args.seed
)

# ================== MODEL ==================
log_print("\n" + "="*60)
log_print("🤖 INITIALIZING MODEL")
log_print("="*60)

model = HybridDetector(num_classes=2, pretrained=True, freeze_cnn=args.freeze_cnn).to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
log_print(f"Model: {trainable:,} trainable / {total:,} total params")

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
    log_print(f"\n📂 Resuming from {LATEST_CHECKPOINT}")
    checkpoint = torch.load(LATEST_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint.get('best_val_acc', 0.0)
    if 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    log_print(f"✅ Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")

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
log_print(f"   Freeze CNN: {args.freeze_cnn}")
log_print("")

for epoch in range(start_epoch, EPOCHS):
    # ========== TRAINING PHASE ==========
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    optimizer.zero_grad()  # Initialize gradients at start of epoch
    
    pbar = tqdm(train_loader, desc=f"[TRAIN] Epoch {epoch+1}/{EPOCHS}", ncols=100)
    
    for batch_idx, (img_masked, dct_feat, labels) in enumerate(pbar):
        img_masked = img_masked.to(DEVICE, non_blocking=True)
        dct_feat = dct_feat.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        
        with autocast(enabled=USE_AMP):
            outputs = model(img_masked, dct_feat)
            loss = criterion(outputs, labels)
            
            # Scale loss by accumulation steps for proper gradient averaging
            if ACCUM_STEPS > 1:
                loss = loss / ACCUM_STEPS
        
        if USE_AMP:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Only update weights every ACCUM_STEPS batches
        if (batch_idx + 1) % ACCUM_STEPS == 0:
            if USE_AMP:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Accumulate metrics (un-scale loss for logging)
        train_loss += loss.item() * labels.size(0) * ACCUM_STEPS
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
        # Update progress bar with effective batch size
        effective_batch = BATCH_SIZE * ACCUM_STEPS
        pbar.set_postfix({
            'loss': f'{loss.item()*ACCUM_STEPS:.4f}',
            'acc': f'{100.*train_correct/train_total:.2f}%',
            'eff_bs': effective_batch
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
log_print(f"\n📝 Training log saved to: {LOG_FILE}")

# ================== AUTO SHUTDOWN ==================
if args.shutdown:
    log_print("\n" + "="*60)
    log_print("🔌 SHUTTING DOWN SYSTEM IN 60 SECONDS...")
    log_print("="*60)
    log_print("Press Ctrl+C to cancel shutdown")
    
    import subprocess
    import time
    
    # Countdown
    for i in range(60, 0, -10):
        log_print(f"   Shutdown in {i} seconds...")
        time.sleep(10)
    
    log_print("\n🔴 Shutting down now...")
    
    # Try different shutdown methods (fallback if one fails)
    try:
        # Method 1: systemctl (no password needed if configured)
        subprocess.run(['sudo', 'systemctl', 'poweroff'], check=True)
    except:
        try:
            # Method 2: shutdown command
            subprocess.run(['sudo', 'shutdown', '-h', 'now'], check=True)
        except:
            # Method 3: poweroff
            subprocess.run(['sudo', 'poweroff'], check=True)