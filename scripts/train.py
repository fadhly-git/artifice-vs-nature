import datetime
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
parser.add_argument('--model', type=str, default='mobilenet_v3_small', 
                    choices=['mobilenet_v3_small', 'mobilenet_v3_large', 'mobilenet_v2',
                             'resnet18', 'resnet34', 'resnet50',
                             'efficientnet_b0', 'efficientnet_b1'],
                    help='CNN backbone architecture from torchvision')
parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
parser.add_argument('--no-amp', action='store_false', dest='amp', help='Disable mixed precision')
parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers (use 0 for ROCm)')
parser.add_argument('--freeze-cnn', action='store_true', help='Freeze CNN backbone')
parser.add_argument('--unfreeze-epoch', type=int, default=10, help='Epoch to unfreeze CNN backbone (last block only)')
parser.add_argument('--unfreeze-all-epoch', type=int, default=15, help='Epoch to unfreeze all CNN layers')
parser.add_argument('--cnn-lr-ratio', type=float, default=0.01, help='CNN LR ratio when unfrozen')
parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping max norm')
parser.add_argument('--label-smoothing', type=float, default=0.05, help='Label smoothing for CE loss')
parser.add_argument('--split-file', type=str, default=None, help='Path to save/load dataset split indices (.pt)')
parser.add_argument('--shutdown', action='store_true', help='Shutdown system after training completes')
parser.add_argument('--shutdown-delay', type=int, default=60, help='Shutdown delay in seconds')
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
LOG_DIR = PROJECT_ROOT / "results" / "logs"; LOG_DIR.mkdir(exist_ok=True, parents=True)
LOG_FILE = LOG_DIR / f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
LATEST_CHECKPOINT = CHECKPOINT_DIR / "hybrid_imaginet_latest.pth"
BEST_CHECKPOINT = CHECKPOINT_DIR / "hybrid_imaginet_best.pth"

def log(m): print(m); open(LOG_FILE, 'a').write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {m}\n")

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
    seed=args.seed,
    split_file=args.split_file,
    save_split=True if args.split_file else False
)

# ================== MODEL ==================
print("\n" + "="*60)
print("🤖 INITIALIZING MODEL")
print("="*60)

model = HybridDetector(
    num_classes=2, 
    pretrained=True, 
    freeze_cnn=args.freeze_cnn,
    model_name=args.model
).to(DEVICE)

# Show model info
model_info = model.get_model_info()
print(f"Backbone: {model_info['backbone']}")
print(f"CNN output dim: {model_info['cnn_output_dim']}")
print(f"Total params: {model_info['total_params']:,}")
print(f"Trainable params: {model_info['trainable_params']:,}")
if model_info['frozen_params'] > 0:
    print(f"Frozen params: {model_info['frozen_params']:,}")

# ================== CLASS WEIGHTS FOR IMBALANCED DATA ==================
# Calculate class weights from dataset (Real:Fake ratio ~2:1)
# Weight = 1 / frequency, normalized
class_counts = torch.tensor([22500.0, 11250.0])  # [Real, Fake] counts
class_weights = (1.0 / class_counts)
class_weights = class_weights / class_weights.sum() * 2.0  # Normalize to sum=2
class_weights = class_weights.to(DEVICE)

print(f"\n⚖️  Class weights: Real={class_weights[0]:.3f}, Fake={class_weights[1]:.3f}")

# ================== OPTIMIZER + SCHEDULER ==================
cnn_lr = args.lr * args.cnn_lr_ratio
optimizer = optim.Adam([
    {'params': model.mlp.parameters(), 'lr': args.lr},
    {'params': model.classifier.parameters(), 'lr': args.lr},
    {'params': model.cnn.parameters(), 'lr': cnn_lr}
], lr=args.lr, weight_decay=1e-5, betas=(0.9, 0.999), eps=1e-8)

scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = GradScaler(init_scale=2.0**10, growth_factor=1.5, backoff_factor=0.5, growth_interval=2000) if USE_AMP else None
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

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
log("\n" + "="*60)
log("🚀 STARTING TRAINING")
log("="*60)
log(f"\nConfig:")
log(f"   Device: {DEVICE}")
log(f"   Model: {args.model}")
log(f"   Batch size: {BATCH_SIZE}")
log(f"   Learning rate: {args.lr}")
log(f"   Epochs: {EPOCHS}")
log(f"   Mixed precision: {USE_AMP}")
log(f"   Seed: {args.seed}")
log(f"   Freeze CNN: {args.freeze_cnn}")
if args.freeze_cnn:
    log(f"   Unfreeze CNN at epoch: {args.unfreeze_epoch}")
    log(f"   Unfreeze ALL CNN at epoch: {args.unfreeze_all_epoch}")
log(f"   Gradient clipping: {args.grad_clip}")
log(f"   Checkpoints dir: {CHECKPOINT_DIR}")
log(f"shutdown: {args.shutdown}")
log("")

for epoch in range(start_epoch, EPOCHS):
    
    # Gradual CNN unfreezing strategy
    if args.freeze_cnn and epoch == args.unfreeze_epoch:
        log(f"\n🔓 Unfreezing CNN backbone (last block only) at epoch {epoch+1}")
        
        # Unfreeze only last block of EfficientNet
        for name, param in model.cnn.named_parameters():
            if 'blocks.6' in name or 'conv_head' in name or 'bn2' in name:
                param.requires_grad = True
        
        # Update only the CNN param group LR, don't recreate optimizer
        cnn_lr = args.lr * args.cnn_lr_ratio
        optimizer.param_groups[2]['lr'] = cnn_lr
        
        log(f"   Last CNN block unfrozen")
        log(f"   CNN LR: {cnn_lr:.2e}")
    
    # Unfreeze all CNN layers at later epoch with CAREFUL learning rate
    if args.freeze_cnn and epoch == args.unfreeze_all_epoch:
        log(f"\n🔓 Unfreezing ALL CNN layers at epoch {epoch+1}")
        
        for param in model.cnn.parameters():
            param.requires_grad = True
        
        # Use MUCH smaller LR for full CNN unfreezing to prevent gradient explosion
        # Early layers need very conservative learning rate
        cnn_lr = args.lr * args.cnn_lr_ratio * 0.1  # 10x smaller than partial unfreeze
        optimizer.param_groups[2]['lr'] = cnn_lr
        
        log(f"   All CNN layers unfrozen")
        log(f"   CNN LR (conservative): {cnn_lr:.2e}")
    
    # ========== TRAINING PHASE ==========
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    train_correct_per_class = [0, 0]  # [Real, Fake]
    train_total_per_class = [0, 0]
    
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

                # Clip gradients with ROCm-safe implementation
                scaler.unscale_(optimizer)
                

                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()

            
            optimizer.step()
        
        train_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
        # Per-class accuracy
        for c in range(2):
            mask = labels == c
            train_total_per_class[c] += mask.sum().item()
            train_correct_per_class[c] += (predicted[mask] == c).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*train_correct/train_total:.2f}%'
        })
    
    train_loss = train_loss / train_total
    train_acc = 100. * train_correct / train_total
    
    # Check for NaN loss - stop training if detected
    if torch.isnan(torch.tensor(train_loss)) or torch.isinf(torch.tensor(train_loss)):
        log(f"\n❌ NaN/Inf detected in training loss! Stopping training.")
        log(f"   This usually means gradient explosion occurred.")
        log(f"   Try: Lower LR, increase gradient clipping, or slower unfreezing schedule")
        break
    
    # ========== VALIDATION PHASE ==========
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_correct_per_class = [0, 0]  # [Real, Fake]
    val_total_per_class = [0, 0]
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    try:
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"[VAL]   Epoch {epoch+1}/{EPOCHS}", ncols=100)
            
            for batch_idx, (img_masked, dct_feat, labels) in enumerate(pbar_val):
                # 🔧 FIX: Print progress setiap 10 batch
                if batch_idx % 10 == 0:
                    print(f"\nValidation batch {batch_idx}/{len(val_loader)}", flush=True)
                
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
                
                # Per-class accuracy
                for c in range(2):
                    mask = labels == c
                    val_total_per_class[c] += mask.sum().item()
                    val_correct_per_class[c] += (predicted[mask] == c).sum().item()
                
                pbar_val.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
                
                # 🔧 FIX: Force flush GPU setiap 50 batch
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.synchronize()

    except Exception as e:
        log(f"❌ Validation error: {e}")
        val_loss = float('inf')
        val_acc = 0.0
    
    val_loss = val_loss / val_total
    val_acc = 100. * val_correct / val_total
    
    # Check for NaN in validation
    if torch.isnan(torch.tensor(val_loss)) or torch.isinf(torch.tensor(val_loss)):
        log(f"\n⚠️  NaN/Inf detected in validation loss!")
        log(f"   Model may have diverged. Consider loading best checkpoint.")
        # Don't break, but skip checkpoint saving
        val_loss = float('inf')
        val_acc = 0.0
    
    # Update learning rate
    scheduler.step()
    
    # ========== LOGGING ==========
    train_acc_real = 100. * train_correct_per_class[0] / train_total_per_class[0] if train_total_per_class[0] > 0 else 0
    train_acc_fake = 100. * train_correct_per_class[1] / train_total_per_class[1] if train_total_per_class[1] > 0 else 0
    val_acc_real = 100. * val_correct_per_class[0] / val_total_per_class[0] if val_total_per_class[0] > 0 else 0
    val_acc_fake = 100. * val_correct_per_class[1] / val_total_per_class[1] if val_total_per_class[1] > 0 else 0
    
    log(f"\n📊 Epoch {epoch+1}/{EPOCHS} Summary:")
    log(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    log(f"      Real: {train_acc_real:.2f}% | Fake: {train_acc_fake:.2f}%")
    log(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    log(f"      Real: {val_acc_real:.2f}% | Fake: {val_acc_fake:.2f}%")
    log(f"   LR: {scheduler.get_last_lr()[0]:.2e}")
    
    # Warning if model is collapsing
    if val_acc_fake < 20.0:
        log(f"   ⚠️  WARNING: Fake class accuracy very low ({val_acc_fake:.2f}%) - possible collapse!")
    
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
        log(f"   🌟 New best validation accuracy: {val_acc:.2f}% - Checkpoint saved!")
    
    print()
    
    # Clear cache every epoch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ================== TRAINING COMPLETE ==================
log("\n" + "="*60)
log("✅ TRAINING COMPLETE")
log("="*60)
log(f"\nBest validation accuracy: {best_val_acc:.2f}%")
log(f"Checkpoints saved in: {CHECKPOINT_DIR}")
log(f"   - Latest: {LATEST_CHECKPOINT.name}")
log(f"   - Best: {BEST_CHECKPOINT.name}")

import time
import platform
# ========== AUTO SHUTDOWN ==========
if args.shutdown:
    print("\n🔴 SYSTEM SHUTDOWN SCHEDULED")
    print(f"   System will shutdown in {args.shutdown_delay} seconds. Press Ctrl+C to cancel.")
    try:
        for i in range(args.shutdown_delay, 0, -1):
            print(f"⏱️  Shutting down in {i} seconds...", end='\r', flush=True)
            time.sleep(1)
        print("\nInitiating shutdown...")
        sys_os = platform.system()
        if sys_os == "Linux" or sys_os == "Darwin":
            os.system("sudo shutdown -h now")
        elif sys_os == "Windows":
            os.system("shutdown /s /t 0")
        else:
            print("❌ Unsupported OS for auto-shutdown.")
    except KeyboardInterrupt:
        print("\n❌ Shutdown cancelled by user.")
else:
    print("\n✨ Training complete. System will NOT shutdown automatically.")