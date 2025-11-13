#!/bin/bash
# Auto-shutdown Training Script
# Run overnight training and auto-shutdown when complete

# Configuration
BATCH_SIZE=2
ACCUM_STEPS=16
EPOCHS=50
SUBSET=""  # Leave empty for full dataset, or set to "0.1" for 10%

# Build command
CMD="python scripts/train.py \
  --batch-size $BATCH_SIZE \
  --accum-steps $ACCUM_STEPS \
  --lr 1e-4 \
  --epochs $EPOCHS \
  --amp \
  --num-workers 0 \
  --shutdown"

# Add subset if specified
if [ -n "$SUBSET" ]; then
    CMD="$CMD --subset $SUBSET"
fi

# Add resume if checkpoint exists
if [ -f "models/checkpoints/hybrid_imaginet_latest.pth" ]; then
    CMD="$CMD --resume"
    echo "✅ Found existing checkpoint, will resume training"
fi

# Print command
echo "="
echo "🚀 Starting Overnight Training"
echo "="
echo "Command: $CMD"
echo ""
echo "⏰ Training will run for $EPOCHS epochs"
echo "🔌 System will auto-shutdown after completion"
echo "⚠️  Press Ctrl+C within 10 seconds to cancel"
echo ""

# Countdown
for i in {10..1}; do
    echo -ne "Starting in $i seconds...\r"
    sleep 1
done
echo ""

# Run training
echo "🏃 Training started..."
eval $CMD

# This line will only be reached if --shutdown flag causes shutdown
# or if shutdown is cancelled
echo "✅ Script completed"
