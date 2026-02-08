# RepText Arabic Training Files

This directory contains all necessary files for pretraining RepText for Arabic text generation.

## 📁 File Structure

### Core Training Files
- **`train_arabic.py`** - Main training script with text perceptual loss
- **`arabic_dataset.py`** - Dataset loader for Arabic training samples
- **`prepare_arabic_dataset.py`** - Script to generate training data
- **`train_config.yaml`** - Training configuration file

### Helper Scripts
- **`train_arabic.sh`** - Bash script for Linux/Mac training pipeline
- **`train_arabic.ps1`** - PowerShell script for Windows training pipeline
- **`download_arabic_fonts.py`** - Downloads free Arabic fonts from Google Fonts

### Data Files
- **`arabic_texts.txt`** - Sample Arabic text for training (60+ phrases)
- **`TRAINING_GUIDE.md`** - Complete training documentation
- **`arabic_training_quickstart.ipynb`** - Interactive Jupyter notebook guide

### Model Files (from original repo)
- **`controlnet_flux.py`** - FLUX ControlNet model implementation
- **`pipeline_flux_controlnet.py`** - Inference pipeline for ControlNet
- **`infer.py`** - Inference script example

## 🚀 Quick Start (3 Steps)

### 1. Download Fonts
```bash
python download_arabic_fonts.py
```

### 2. Prepare Dataset
```bash
python prepare_arabic_dataset.py \
    --output_dir ./arabic_training_data \
    --font_dir ./arabic_fonts \
    --text_file ./arabic_texts.txt \
    --num_samples 10000
```

### 3. Train
```bash
# Linux/Mac
./train_arabic.sh

# Windows
.\train_arabic.ps1

# Or manually
accelerate launch train_arabic.py --config train_config.yaml
```

## 📊 What Gets Created

### After Dataset Preparation
```
arabic_training_data/
├── sample_000000/
│   ├── glyph.png          # Rendered Arabic text
│   ├── position.png       # Position heatmap
│   ├── mask.png          # Text region mask
│   ├── canny.png         # Edge detection
│   └── metadata.json     # Text & font info
├── sample_000001/
│   └── ...
└── dataset_info.json
```

### After Training
```
output/arabic_reptext/
├── checkpoint-1000/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── checkpoint-2000/
├── final_model/
└── logs/
```

## ⚙️ Configuration

Edit `train_config.yaml` to customize:

```yaml
# Dataset
data:
  batch_size: 4              # Reduce if OOM
  num_samples: 10000         # More = better model

# Training
training:
  num_epochs: 100            # Typical: 50-100
  learning_rate: 1.0e-5      # ControlNet LR
  text_perceptual_loss_weight: 0.1  # Text accuracy weight
```

## 🔧 Common Adjustments

### Limited GPU Memory?
```yaml
data:
  batch_size: 2
  image_size: [512, 512]    # Instead of [1024, 1024]
  
training:
  gradient_accumulation_steps: 2
  mixed_precision: "fp16"
```

### Want Faster Training?
```yaml
data:
  batch_size: 8             # If you have 40GB+ VRAM
  num_workers: 8
  
training:
  gradient_accumulation_steps: 1
```

### Better Text Quality?
```yaml
training:
  text_perceptual_loss_weight: 0.2  # Increase from 0.1
  
ocr:
  model_name: "arabic-specific-ocr-model"  # Use Arabic OCR
```

## 📝 Customization Examples

### Use Your Own Arabic Text
Create `my_arabic_texts.txt`:
```
نص عربي مخصص
عبارة أخرى
...
```

Then:
```bash
python prepare_arabic_dataset.py \
    --text_file ./my_arabic_texts.txt \
    --output_dir ./my_data \
    --num_samples 20000
```

### Use Different Fonts
```bash
# Add your fonts to arabic_fonts/
cp /path/to/MyArabicFont.ttf ./arabic_fonts/

# Regenerate dataset
python prepare_arabic_dataset.py \
    --font_dir ./arabic_fonts \
    --output_dir ./arabic_training_data
```

### Resume Training
```bash
accelerate launch train_arabic.py \
    --config train_config.yaml \
    --resume_from_checkpoint ./output/arabic_reptext/checkpoint-5000
```

## 🧪 Testing

### Test Dataset Preparation
```bash
python prepare_arabic_dataset.py \
    --output_dir ./test_data \
    --font_dir ./arabic_fonts \
    --num_samples 10
```

### Test Dataset Loading
```bash
python arabic_dataset.py ./arabic_training_data
```

### Test Trained Model
```python
from controlnet_flux import FluxControlNetModel
model = FluxControlNetModel.from_pretrained("./output/arabic_reptext/final_model")
```

## 📚 Documentation

- **`TRAINING_GUIDE.md`** - Complete guide with troubleshooting
- **`arabic_training_quickstart.ipynb`** - Interactive tutorial
- Original RepText: [README.md](README.md)

## 🆘 Troubleshooting

### "No fonts found"
```bash
python download_arabic_fonts.py
```

### "CUDA out of memory"
Reduce `batch_size` in `train_config.yaml`

### "Dataset empty"
Check `arabic_training_data/` was created successfully:
```bash
ls arabic_training_data/
```

### "Poor Arabic text quality"
1. Use more fonts (10+ recommended)
2. Increase `text_perceptual_loss_weight`
3. Train longer (100+ epochs)
4. Use larger dataset (50K+ samples)

## 💡 Tips

1. **Start small**: Test with 100 samples first
2. **Use tmux/screen**: Training takes hours/days
3. **Monitor GPU**: `watch -n 1 nvidia-smi`
4. **Save checkpoints**: Set `save_steps: 500` for frequent saves
5. **Use W&B**: Add `--use_wandb` for better monitoring

## 🔗 Resources

- [RepText Paper](https://arxiv.org/abs/2504.19724)
- [FLUX.1 Model](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [Google Arabic Fonts](https://fonts.google.com/?subset=arabic)
- [HuggingFace Diffusers](https://huggingface.co/docs/diffusers)

## 📧 Support

For issues:
1. Check [TRAINING_GUIDE.md](TRAINING_GUIDE.md) troubleshooting section
2. Verify GPU memory: `nvidia-smi`
3. Check logs: `./output/arabic_reptext/logs/`

---

Happy Training! 🎉
