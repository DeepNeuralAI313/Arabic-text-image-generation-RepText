# RepText Arabic Training Guide

This guide will help you pretrain RepText for Arabic text generation.

## 📋 Prerequisites

1. **GPU**: NVIDIA GPU with at least 24GB VRAM (e.g., RTX 3090, A100, etc.)
2. **Python**: Python 3.8 or higher
3. **CUDA**: CUDA 11.7 or higher

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Arabic Fonts

Create a directory called `arabic_fonts` and add multiple Arabic fonts:

```bash
mkdir arabic_fonts
cd arabic_fonts
```

**Recommended Arabic Fonts:**
- [Amiri](https://fonts.google.com/specimen/Amiri) - Traditional Arabic
- [Cairo](https://fonts.google.com/specimen/Cairo) - Modern Arabic
- [Noto Sans Arabic](https://fonts.google.com/noto/specimen/Noto+Sans+Arabic) - Clean sans-serif
- [Tajawal](https://fonts.google.com/specimen/Tajawal) - Readable and modern
- [Almarai](https://fonts.google.com/specimen/Almarai) - Contemporary design
- [Changa](https://fonts.google.com/specimen/Changa) - Bold and strong
- [Lalezar](https://fonts.google.com/specimen/Lalezar) - Decorative
- [Markazi Text](https://fonts.google.com/specimen/Markazi+Text) - Text-friendly

Download these from [Google Fonts](https://fonts.google.com/?subset=arabic) or other font repositories.

Example folder structure:
```
arabic_fonts/
├── Amiri-Regular.ttf
├── Cairo-Regular.ttf
├── Cairo-Bold.ttf
├── NotoSansArabic-Regular.ttf
├── NotoSansArabic-Bold.ttf
├── Tajawal-Regular.ttf
└── Almarai-Regular.ttf
```

### 3. Prepare Arabic Text Samples (Optional)

Create a file `arabic_texts.txt` with Arabic text samples (one per line):

```
السلام عليكم
مرحبا بكم
شكرا جزيلا
الله أكبر
كتاب مفيد
طالب مجتهد
مدينة جميلة
صباح الخير
مساء الخير
```

If you don't provide this file, the script will generate random Arabic text.

### 4. Configure Training

Edit `train_config.yaml` to adjust training parameters:

```yaml
# Data
data:
  data_dir: "./arabic_training_data"
  batch_size: 4  # Adjust based on your GPU memory
  num_samples: 10000  # Number of training samples

# Training
training:
  num_epochs: 100
  learning_rate: 1.0e-5
  save_steps: 1000
  eval_steps: 500
```

### 5. Set Up Accelerate

Configure accelerate for distributed training:

```bash
accelerate config
```

For single GPU, use these settings:
- Compute environment: This machine
- Number of processes: 1
- Use FP16 precision: No
- Use BF16 precision: Yes (if supported)

### 6. Run Training

**On Linux/Mac:**
```bash
chmod +x train_arabic.sh
./train_arabic.sh
```

**On Windows PowerShell:**
```powershell
.\train_arabic.ps1
```

**Or manually:**
```bash
# Step 1: Prepare dataset
python prepare_arabic_dataset.py \
    --output_dir ./arabic_training_data \
    --font_dir ./arabic_fonts \
    --text_file ./arabic_texts.txt \
    --num_samples 10000 \
    --width 1024 \
    --height 1024

# Step 2: Launch training
accelerate launch train_arabic.py \
    --config train_config.yaml
```

## 📊 Monitor Training

### Using Weights & Biases (Recommended)

1. Install wandb:
```bash
pip install wandb
wandb login
```

2. Run with W&B:
```bash
accelerate launch train_arabic.py --config train_config.yaml --use_wandb
```

### Without W&B

Training logs will be saved to `./output/arabic_reptext/logs/`

## 🎯 Training Parameters Explained

### Dataset Preparation

- `--num_samples`: Number of training samples to generate (default: 10000)
  - More samples = better model but longer prep time
  - Recommended: 10,000 - 50,000 for good results

- `--min_font_size` / `--max_font_size`: Font size range (default: 60-120)
  - Larger fonts are easier to learn
  - RepText paper recommends >= 60pt

### Training Configuration

- `batch_size`: Batch size per GPU (default: 4)
  - Larger = faster training but needs more VRAM
  - Reduce if you encounter OOM errors

- `learning_rate`: Learning rate (default: 1e-5)
  - ControlNet training typically uses lower LR

- `text_perceptual_loss_weight`: Weight for OCR-based text loss (default: 0.1)
  - Helps ensure text accuracy
  - Can be increased for stronger text guidance

- `num_epochs`: Number of training epochs
  - Typical: 50-100 epochs
  - Monitor validation loss to avoid overfitting

## 💾 Model Checkpoints

Checkpoints are saved to `./output/arabic_reptext/`:

```
output/arabic_reptext/
├── checkpoint-1000/
├── checkpoint-2000/
├── checkpoint-3000/
└── final_model/
```

Each checkpoint can be used for inference.

## 🧪 Testing Your Model

After training, test your model:

```python
import torch
from controlnet_flux import FluxControlNetModel
from pipeline_flux_controlnet import FluxControlNetPipeline
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Load your trained model
base_model = "black-forest-labs/FLUX.1-dev"
controlnet_model = "./output/arabic_reptext/final_model"

controlnet = FluxControlNetModel.from_pretrained(
    controlnet_model, 
    torch_dtype=torch.bfloat16
)
pipe = FluxControlNetPipeline.from_pretrained(
    base_model, 
    controlnet=controlnet, 
    torch_dtype=torch.bfloat16
).to("cuda")

# Test with Arabic text
# ... (use the inference code from infer.py)
```

## 🔧 Troubleshooting

### Out of Memory (OOM) Errors

1. Reduce `batch_size` in config
2. Enable gradient checkpointing
3. Use smaller image size (e.g., 512x512)
4. Use gradient accumulation:
   ```yaml
   gradient_accumulation_steps: 2  # Effective batch size = batch_size * 2
   ```

### Training Too Slow

1. Use multiple GPUs with `accelerate`
2. Increase `batch_size` if you have VRAM
3. Use mixed precision (bf16/fp16)
4. Reduce `num_samples` for faster experimentation

### Poor Text Quality

1. Increase `text_perceptual_loss_weight`
2. Use more diverse fonts
3. Increase training data size
4. Train for more epochs
5. Use an Arabic-specific OCR model in config:
   ```yaml
   ocr:
     model_name: "arabert/AraGPT2-base"  # Example Arabic model
   ```

### Font Rendering Issues

1. Ensure fonts support Arabic characters
2. Install language packs: `sudo apt-get install fonts-arabeyes`
3. Test fonts with PIL before training:
   ```python
   from PIL import Image, ImageDraw, ImageFont
   font = ImageFont.truetype("./arabic_fonts/Amiri-Regular.ttf", 80)
   img = Image.new('RGB', (500, 200), 'white')
   draw = ImageDraw.Draw(img)
   draw.text((10, 50), "مرحبا", font=font, fill='black')
   img.show()
   ```

## 📚 Advanced Tips

### Data Augmentation

Add more variety by:
- Using background images instead of solid colors
- Adding random rotations (small angles)
- Varying text colors and backgrounds
- Using different text layouts (multi-line, curved)

### Custom OCR Model

Replace the OCR model with an Arabic-specific one for better text perceptual loss:

```python
# In train_config.yaml
ocr:
  model_name: "microsoft/trocr-base-printed"  # Change to Arabic OCR
  use_ocr_loss: true
```

### Resume Training

Resume from a checkpoint:

```bash
accelerate launch train_arabic.py \
    --config train_config.yaml \
    --resume_from_checkpoint ./output/arabic_reptext/checkpoint-5000
```

## 📖 Additional Resources

- [RepText Paper](https://arxiv.org/abs/2504.19724)
- [FLUX.1 Documentation](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate)

## 🤝 Support

If you encounter issues:
1. Check GPU memory usage: `nvidia-smi`
2. Verify dataset preparation completed successfully
3. Check logs in `./output/arabic_reptext/logs/`
4. Ensure all fonts support Arabic Unicode range (U+0600 to U+06FF)

## 📄 License

Same as original RepText repository.

---

Happy training! 🚀
