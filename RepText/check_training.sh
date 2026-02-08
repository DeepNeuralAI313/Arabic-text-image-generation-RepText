#!/bin/bash
# Quick script to check if Arabic RepText training is complete

clear
echo "========================================"
echo "  RepText Arabic Training Status"
echo "========================================"
echo ""

OUTPUT_DIR="./output/arabic_reptext"
FINAL_MODEL="$OUTPUT_DIR/final_model"

# Check if training started
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ Training hasn't started yet!"
    echo ""
    echo "To start training:"
    echo "  cd RepText"
    echo "  accelerate launch train_arabic.py --config train_config_48gb.yaml --use_wandb"
    exit 1
fi

echo "✓ Training output directory found"
echo ""

# Check for checkpoints
CHECKPOINTS=$(find "$OUTPUT_DIR" -name "checkpoint-*" -type d 2>/dev/null | wc -l)

if [ "$CHECKPOINTS" -gt 0 ]; then
    echo "Found $CHECKPOINTS checkpoint(s):"
    find "$OUTPUT_DIR" -name "checkpoint-*" -type d | sort | tail -5 | while read cp; do
        STEP=$(basename "$cp" | cut -d'-' -f2)
        echo "  - Step $STEP"
    done
    echo ""
fi

# Check if training completed
if [ -d "$FINAL_MODEL" ]; then
    echo "🎉 TRAINING COMPLETED!"
    echo ""
    echo "Final model location: $FINAL_MODEL"
    echo ""
    echo "To test your model:"
    echo "  python check_training.py"
    exit 0
else
    echo "⏳ Training still in progress..."
    echo ""
    
    # Show latest checkpoint
    LATEST=$(find "$OUTPUT_DIR" -name "checkpoint-*" -type d | sort | tail -1)
    if [ -n "$LATEST" ]; then
        echo "Latest checkpoint: $(basename $LATEST)"
    fi
    
    echo ""
    echo "Check training progress:"
    echo "  - W&B dashboard: https://wandb.ai/"
    echo "  - GPU usage: nvidia-smi"
    echo "  - Logs: tail -f $OUTPUT_DIR/logs/*.log"
    echo ""
    echo "Estimated time remaining: 1-2 days (depending on GPU)"
    exit 0
fi
