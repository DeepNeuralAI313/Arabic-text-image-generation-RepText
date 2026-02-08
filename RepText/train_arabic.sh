#!/bin/bash
# Launch script for RepText Arabic training
# This script handles the complete training pipeline

set -e

echo "========================================="
echo "RepText Arabic Training Pipeline"
echo "========================================="

# Configuration
FONT_DIR="${FONT_DIR:-./arabic_fonts}"
TEXT_FILE="${TEXT_FILE:-./arabic_texts.txt}"
DATA_DIR="${DATA_DIR:-./arabic_training_data}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
CONFIG_FILE="${CONFIG_FILE:-train_config.yaml}"
USE_WANDB="${USE_WANDB:-false}"

# Step 1: Check for fonts
echo ""
echo "Step 1: Checking for Arabic fonts..."
if [ ! -d "$FONT_DIR" ] || [ -z "$(ls -A $FONT_DIR 2>/dev/null)" ]; then
    echo "ERROR: Font directory '$FONT_DIR' not found or empty!"
    echo "Please create a directory with Arabic fonts (.ttf or .otf files)"
    echo ""
    echo "You can download Arabic fonts from:"
    echo "  - Google Fonts: https://fonts.google.com/?subset=arabic"
    echo "  - 1001 Fonts: https://www.1001fonts.com/arabic-fonts.html"
    echo ""
    echo "Suggested fonts:"
    echo "  - Amiri, Cairo, Noto Sans Arabic, Tajawal, Almarai, etc."
    echo ""
    echo "Example folder structure:"
    echo "  $FONT_DIR/"
    echo "    ├── Amiri-Regular.ttf"
    echo "    ├── Cairo-Regular.ttf"
    echo "    └── NotoSansArabic-Regular.ttf"
    exit 1
fi

FONT_COUNT=$(find "$FONT_DIR" -name "*.ttf" -o -name "*.otf" | wc -l)
echo "Found $FONT_COUNT fonts in $FONT_DIR"

# Step 2: Check for text samples (optional)
echo ""
echo "Step 2: Checking for Arabic text samples..."
if [ -f "$TEXT_FILE" ]; then
    LINE_COUNT=$(wc -l < "$TEXT_FILE")
    echo "Found $LINE_COUNT text samples in $TEXT_FILE"
else
    echo "No text file provided. Will generate random Arabic text."
fi

# Step 3: Prepare dataset
echo ""
echo "Step 3: Preparing training dataset..."
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating dataset with $NUM_SAMPLES samples..."
    
    if [ -f "$TEXT_FILE" ]; then
        python prepare_arabic_dataset.py \
            --output_dir "$DATA_DIR" \
            --font_dir "$FONT_DIR" \
            --text_file "$TEXT_FILE" \
            --num_samples "$NUM_SAMPLES" \
            --width 1024 \
            --height 1024 \
            --min_font_size 60 \
            --max_font_size 120
    else
        python prepare_arabic_dataset.py \
            --output_dir "$DATA_DIR" \
            --font_dir "$FONT_DIR" \
            --num_samples "$NUM_SAMPLES" \
            --width 1024 \
            --height 1024 \
            --min_font_size 60 \
            --max_font_size 120
    fi
    
    echo "Dataset preparation complete!"
else
    echo "Dataset already exists at $DATA_DIR"
    echo "Skipping dataset preparation. Delete the directory to regenerate."
fi

# Step 4: Launch training
echo ""
echo "Step 4: Launching training..."
echo "Config file: $CONFIG_FILE"
echo ""

if [ "$USE_WANDB" = "true" ]; then
    echo "Using Weights & Biases for logging"
    accelerate launch train_arabic.py \
        --config "$CONFIG_FILE" \
        --use_wandb
else
    echo "Training without W&B logging"
    accelerate launch train_arabic.py \
        --config "$CONFIG_FILE"
fi

echo ""
echo "========================================="
echo "Training complete!"
echo "========================================="
