# Windows PowerShell script for RepText Arabic training
# This script handles the complete training pipeline

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "RepText Arabic Training Pipeline" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Configuration
$FONT_DIR = if ($env:FONT_DIR) { $env:FONT_DIR } else { ".\arabic_fonts" }
$TEXT_FILE = if ($env:TEXT_FILE) { $env:TEXT_FILE } else { ".\arabic_texts.txt" }
$DATA_DIR = if ($env:DATA_DIR) { $env:DATA_DIR } else { ".\arabic_training_data" }
$NUM_SAMPLES = if ($env:NUM_SAMPLES) { $env:NUM_SAMPLES } else { "10000" }
$CONFIG_FILE = if ($env:CONFIG_FILE) { $env:CONFIG_FILE } else { "train_config.yaml" }
$USE_WANDB = if ($env:USE_WANDB) { $env:USE_WANDB } else { "false" }

# Step 1: Check for fonts
Write-Host ""
Write-Host "Step 1: Checking for Arabic fonts..." -ForegroundColor Yellow

if (-not (Test-Path $FONT_DIR) -or ((Get-ChildItem $FONT_DIR -ErrorAction SilentlyContinue).Count -eq 0)) {
    Write-Host "ERROR: Font directory '$FONT_DIR' not found or empty!" -ForegroundColor Red
    Write-Host "Please create a directory with Arabic fonts (.ttf or .otf files)" -ForegroundColor Red
    Write-Host ""
    Write-Host "You can download Arabic fonts from:" -ForegroundColor Yellow
    Write-Host "  - Google Fonts: https://fonts.google.com/?subset=arabic"
    Write-Host "  - 1001 Fonts: https://www.1001fonts.com/arabic-fonts.html"
    Write-Host ""
    Write-Host "Suggested fonts:" -ForegroundColor Yellow
    Write-Host "  - Amiri, Cairo, Noto Sans Arabic, Tajawal, Almarai, etc."
    Write-Host ""
    Write-Host "Example folder structure:" -ForegroundColor Yellow
    Write-Host "  $FONT_DIR\"
    Write-Host "    ├── Amiri-Regular.ttf"
    Write-Host "    ├── Cairo-Regular.ttf"
    Write-Host "    └── NotoSansArabic-Regular.ttf"
    exit 1
}

$FONT_COUNT = (Get-ChildItem -Path $FONT_DIR -Include @("*.ttf", "*.otf") -Recurse).Count
Write-Host "Found $FONT_COUNT fonts in $FONT_DIR" -ForegroundColor Green

# Step 2: Check for text samples
Write-Host ""
Write-Host "Step 2: Checking for Arabic text samples..." -ForegroundColor Yellow

if (Test-Path $TEXT_FILE) {
    $LINE_COUNT = (Get-Content $TEXT_FILE).Count
    Write-Host "Found $LINE_COUNT text samples in $TEXT_FILE" -ForegroundColor Green
} else {
    Write-Host "No text file provided. Will generate random Arabic text." -ForegroundColor Yellow
}

# Step 3: Prepare dataset
Write-Host ""
Write-Host "Step 3: Preparing training dataset..." -ForegroundColor Yellow

if (-not (Test-Path $DATA_DIR)) {
    Write-Host "Creating dataset with $NUM_SAMPLES samples..." -ForegroundColor Yellow
    
    $prepareArgs = @(
        "prepare_arabic_dataset.py",
        "--output_dir", $DATA_DIR,
        "--font_dir", $FONT_DIR,
        "--num_samples", $NUM_SAMPLES,
        "--width", "1024",
        "--height", "1024",
        "--min_font_size", "60",
        "--max_font_size", "120"
    )
    
    if (Test-Path $TEXT_FILE) {
        $prepareArgs += "--text_file"
        $prepareArgs += $TEXT_FILE
    }
    
    python @prepareArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Dataset preparation complete!" -ForegroundColor Green
    } else {
        Write-Host "Error during dataset preparation!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Dataset already exists at $DATA_DIR" -ForegroundColor Green
    Write-Host "Skipping dataset preparation. Delete the directory to regenerate." -ForegroundColor Yellow
}

# Step 4: Launch training
Write-Host ""
Write-Host "Step 4: Launching training..." -ForegroundColor Yellow
Write-Host "Config file: $CONFIG_FILE" -ForegroundColor Cyan
Write-Host ""

$trainArgs = @(
    "train_arabic.py",
    "--config", $CONFIG_FILE
)

if ($USE_WANDB -eq "true") {
    Write-Host "Using Weights & Biases for logging" -ForegroundColor Cyan
    $trainArgs += "--use_wandb"
} else {
    Write-Host "Training without W&B logging" -ForegroundColor Yellow
}

accelerate launch @trainArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=========================================" -ForegroundColor Cyan
    Write-Host "Training complete!" -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "Training failed!" -ForegroundColor Red
    exit 1
}
