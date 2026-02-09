# Cleanup script for old training configs
# Run this to remove outdated YAML files

Write-Host "Cleaning up old training configuration files..." -ForegroundColor Cyan

$filesToRemove = @(
    "train_config_48gb.yaml",
    "train_config_48gb_optimized.yaml",
    "train_config_sdxl.yaml"
)

foreach ($file in $filesToRemove) {
    $filePath = Join-Path $PSScriptRoot $file
    if (Test-Path $filePath) {
        Remove-Item $filePath -Force
        Write-Host "✓ Removed: $file" -ForegroundColor Green
    } else {
        Write-Host "  Not found: $file" -ForegroundColor Yellow
    }
}

Write-Host "`nCleanup complete!" -ForegroundColor Cyan
Write-Host "`nAvailable configs:" -ForegroundColor White
Write-Host "  • train_config.yaml       - Default config (48GB+ VRAM)" -ForegroundColor White
Write-Host "  • train_config_98gb.yaml  - Optimized for 98GB VRAM (max performance)" -ForegroundColor White
