"""
Simplified inference script for RepText Arabic text generation.

Uses the trained ControlNet to generate images with Arabic text.

Usage:
    python infer_simple.py --text "السلام عليكم" --output results/
    python infer_simple.py --text "مرحبا" --num_steps 50
"""

import os
import argparse
import yaml
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import torch
from controlnet_flux import FluxControlNetModel
from pipeline_flux_controlnet import FluxControlNetPipeline


def prepare_glyph_and_controls(text, font_path, font_size=80, width=512, height=512):
    """Prepare glyph, canny, and position maps for inference.
    
    Returns:
        glyph: RGB glyph image
        canny: RGB canny edges
        position: Grayscale position map (will be expanded to 3ch by pipeline)
    """
    
    # Render text (glyph)
    glyph_img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(glyph_img)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"Warning: Could not load font {font_path}: {e}")
        font = ImageFont.load_default()
    
    # Draw text
    draw.text((50, (height - font_size) // 2), text, font=font, fill='black')
    
    # Create canny edges
    glyph_array = np.array(glyph_img.convert('L'))
    canny_edges = cv2.Canny(glyph_array, 50, 100)
    canny_img = Image.fromarray(np.dstack([255 - canny_edges, 255 - canny_edges, 255 - canny_edges]))
    
    # Create position map as GRAYSCALE (single channel)
    # Pipeline will expand to 3 channels automatically
    position_array = np.zeros((height, width), dtype=np.uint8)
    position_array[100:height-100, 50:width-50] = 200
    position_img = Image.fromarray(position_array)  # Grayscale
    
    return glyph_img, canny_img, position_img


def generate_image(text, config_path="train_config.yaml", num_steps=30, 
                   controlnet_step=20, font_size=80, output_dir="results"):
    """Generate image with Arabic text using trained ControlNet."""
    
    # Load config
    if not Path(config_path).exists():
        print(f"❌ Error: Config not found at {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get model paths
    checkpoint_dir = config['output']['output_dir']
    model_path = os.path.join(checkpoint_dir, "final_model")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Trained model not found at {model_path}")
        print(f"Available items in {checkpoint_dir}:")
        if os.path.exists(checkpoint_dir):
            for item in os.listdir(checkpoint_dir):
                print(f"  - {item}")
        return None
    
    print(f"✓ Loading model from {model_path}")
    
    try:
        # Load models
        base_model = config['model']['base_model']
        
        # Load ControlNet WITHOUT config overrides
        # The checkpoint already has the correct architecture (in_channels, num_layers, etc)
        print("Loading ControlNet...")
        controlnet = FluxControlNetModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        )
        
        print("Loading FLUX pipeline...")
        pipe = FluxControlNetPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            torch_dtype=torch.bfloat16
        ).to("cuda")
        
        print("✅ Models loaded successfully!")
        
        # Prepare conditioning
        print(f"\nGenerating image for: {text}")
        
        # Get first available font
        font_dir = "./arabic_fonts"
        if not os.path.exists(font_dir):
            print(f"❌ Error: Font directory not found at {font_dir}")
            return None
        
        fonts = [f for f in os.listdir(font_dir) if f.endswith(('.ttf', '.otf'))]
        if not fonts:
            print(f"❌ Error: No fonts found in {font_dir}")
            return None
        
        font_path = os.path.join(font_dir, fonts[0])
        print(f"Using font: {fonts[0]}")
        
        # Prepare controls
        glyph, canny, position = prepare_glyph_and_controls(
            text, font_path, font_size=font_size, width=512, height=512
        )
        
        print(f"Inference steps: {num_steps}")
        print(f"ControlNet conditioning until step: {controlnet_step}")
        
        # Generate with glyph as primary control and position as spatial guide
        with torch.no_grad():
            generator = torch.Generator(device="cuda").manual_seed(42)
            
            image = pipe(
                "",  # Empty prompt
                height=512,
                width=512,
                num_inference_steps=num_steps,
                guidance_scale=0.0,
                controlnet_conditioning_scale=1.0,
                controlnet_conditioning_step=controlnet_step,
                control_image=[glyph],  # Use glyph as main spatial control
                control_position=[position],  # Position map (grayscale, expanded to 3ch by pipeline)
                control_glyph=glyph,  # Also used for latent initialization
                control_mask=None,
                generator=generator,
            ).images[0]
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        safe_text = "".join(c for c in text if ord(c) >= 32).strip()[:20]
        output_file = os.path.join(output_dir, f"generated_{safe_text}.png")
        image.save(output_file)
        
        print(f"✅ Image saved to {output_file}")
        return image
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Arabic text images with RepText")
    parser.add_argument("--text", type=str, default="السلام عليكم", help="Arabic text to render")
    parser.add_argument("--config", type=str, default="train_config.yaml", help="Path to training config")
    parser.add_argument("--num_steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--controlnet_step", type=int, default=20, help="When to stop applying ControlNet")
    parser.add_argument("--font_size", type=int, default=80, help="Font size for text rendering")
    parser.add_argument("--output", type=str, default="results_inference", help="Output directory")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RepText Arabic Text Generation - Inference")
    print("=" * 60)
    
    image = generate_image(
        text=args.text,
        config_path=args.config,
        num_steps=args.num_steps,
        controlnet_step=args.controlnet_step,
        font_size=args.font_size,
        output_dir=args.output
    )
    
    if image:
        print("\n✅ Inference completed successfully!")
    else:
        print("\n❌ Inference failed!")
