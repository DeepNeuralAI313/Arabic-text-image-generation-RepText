# PowerShell script to check Arabic RepText training status

Clear-Host
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  RepText Arabic Training Status" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$OUTPUT_DIR = "./output/arabic_reptext"
$FINAL_MODEL = "$OUTPUT_DIR/final_model"

# Check if training started
if (-not (Test-Path $OUTPUT_DIR)) {
    Write-Host "❌ Training hasn't started yet!" -ForegroundColor Red
    Write-Host ""
    Write-Host "To start training:" -ForegroundColor Yellow
    Write-Host "  cd RepText"
    Write-Host "  accelerate launch train_arabic.py --config train_config_48gb.yaml --use_wandb"
    exit 1
}

Write-Host "✓ Training output directory found" -ForegroundColor Green
Write-Host ""

# Check for checkpoints
$CHECKPOINTS = Get-ChildItem -Path $OUTPUT_DIR -Directory -Filter "checkpoint-*" -ErrorAction SilentlyContinue

if ($CHECKPOINTS) {
    $COUNT = ($CHECKPOINTS | Measure-Object).Count
    Write-Host "Found $COUNT checkpoint(s):" -ForegroundColor Green
    $CHECKPOINTS | Sort-Object Name | Select-Object -Last 5 | ForEach-Object {
        $STEP = $_.Name -replace "checkpoint-", ""
        Write-Host "  - Step $STEP"
    }
    Write-Host ""
}

# Check if training completed
if (Test-Path $FINAL_MODEL) {
    Write-Host "🎉 TRAINING COMPLETED!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Final model location: $FINAL_MODEL" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To test your model:" -ForegroundColor Yellow
    Write-Host "  python check_training.py"
    exit 0
} else {
    Write-Host "⏳ Training still in progress..." -ForegroundColor Yellow
    Write-Host ""
    
    # Show latest checkpoint
    if ($CHECKPOINTS) {
        $LATEST = $CHECKPOINTS | Sort-Object Name | Select-Object -Last 1
        Write-Host "Latest checkpoint: $($LATEST.Name)" -ForegroundColor Cyan
    }
    
    Write-Host ""
    Write-Host "Check training progress:" -ForegroundColor Yellow
    Write-Host "  - W&B dashboard: https://wandb.ai/"
    Write-Host "  - GPU usage: nvidia-smi"
    Write-Host "  - Logs: Get-Content $OUTPUT_DIR/logs/*.log -Tail 50 -Wait"
    Write-Host ""
    Write-Host "Estimated time remaining: 1-2 days (depending on GPU)" -ForegroundColor Cyan
    exit 0
}
