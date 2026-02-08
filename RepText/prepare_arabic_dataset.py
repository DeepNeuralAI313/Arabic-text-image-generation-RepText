"""
Arabic Dataset Preparation Script for RepText Training
This script prepares training data by generating glyph images and position maps for Arabic text.
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from tqdm import tqdm
import argparse


def get_arabic_fonts(font_dir: str) -> List[str]:
    """Get list of Arabic-compatible font files."""
    font_extensions = ['.ttf', '.otf']
    fonts = []
    
    if os.path.exists(font_dir):
        for file in os.listdir(font_dir):
            if any(file.lower().endswith(ext) for ext in font_extensions):
                fonts.append(os.path.join(font_dir, file))
    
    return fonts


def load_arabic_text_samples(text_file: str) -> List[str]:
    """Load Arabic text samples from file."""
    with open(text_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def generate_random_arabic_text(min_words: int = 1, max_words: int = 5) -> str:
    """Generate random Arabic text for training."""
    # Common Arabic words for synthetic data
    arabic_words = [
        "السلام", "عليكم", "مرحبا", "شكرا", "الله", "محمد", "كتاب", "قلم",
        "مدرسة", "جامعة", "طالب", "معلم", "بيت", "شارع", "مدينة", "بلد",
        "صديق", "عائلة", "أب", "أم", "أخ", "أخت", "ابن", "بنت",
        "يوم", "ليلة", "صباح", "مساء", "وقت", "ساعة", "دقيقة", "ثانية",
        "طعام", "ماء", "خبز", "لحم", "فواكه", "خضروات", "قهوة", "شاي"
    ]
    
    num_words = random.randint(min_words, max_words)
    return " ".join(random.choices(arabic_words, k=num_words))


def canny_edge(img: np.ndarray) -> np.ndarray:
    """Apply Canny edge detection."""
    low_threshold = 50
    high_threshold = 100
    img = cv2.Canny(img, low_threshold, high_threshold)
    img = img[:, :, None]
    img = 255 - np.concatenate([img, img, img], axis=2)
    return img


def create_glyph_image(
    text: str,
    font: ImageFont.FreeTypeFont,
    width: int,
    height: int,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create glyph image, position map, and mask for given text.
    
    Returns:
        glyph: Rendered text on black background
        position: Position heatmap
        mask: Binary mask for text region
    """
    # Create glyph image
    glyph_img = Image.new('RGB', (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(glyph_img)
    
    # Draw text - Arabic is RTL, PIL handles it
    draw.text(position, text, font=font, fill=color)
    
    # Get text bounding box
    bbox = draw.textbbox(position, text, font=font)
    
    # Create position map (heatmap around text)
    position_img = Image.new('L', (width, height), 0)
    position_draw = ImageDraw.Draw(position_img)
    
    # Create circular gradient around text center
    center_x = (bbox[0] + bbox[2]) // 2
    center_y = (bbox[1] + bbox[3]) // 2
    max_radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) // 2
    
    for radius in range(max_radius, 0, -5):
        intensity = int(255 * (1 - radius / max_radius))
        position_draw.ellipse(
            [center_x - radius, center_y - radius, 
             center_x + radius, center_y + radius],
            fill=intensity
        )
    
    # Create binary mask
    mask_img = Image.new('L', (width, height), 0)
    mask_draw = ImageDraw.Draw(mask_img)
    
    # Expand bbox slightly for mask
    padding = 10
    mask_bbox = [
        max(0, bbox[0] - padding),
        max(0, bbox[1] - padding),
        min(width, bbox[2] + padding),
        min(height, bbox[3] + padding)
    ]
    mask_draw.rectangle(mask_bbox, fill=255)
    
    # Convert to numpy arrays
    glyph = np.array(glyph_img)
    position = np.array(position_img)
    position = cv2.cvtColor(position, cv2.COLOR_GRAY2RGB)
    mask = np.array(mask_img)
    
    return glyph, position, mask


def prepare_training_sample(
    text: str,
    font_path: str,
    font_size: int,
    width: int,
    height: int,
    background_image_path: str = None
) -> Dict:
    """
    Prepare a single training sample with all necessary components.
    
    Returns:
        Dictionary containing glyph, position, mask, canny edge, and metadata
    """
    # Load font
    font = ImageFont.truetype(font_path, font_size)
    
    # Random position for text (avoid edges)
    margin = 100
    position = (
        random.randint(margin, width - margin - 200),
        random.randint(margin, height - margin - 100)
    )
    
    # Random color
    color = tuple(random.randint(200, 255) for _ in range(3))
    
    # Create glyph, position, and mask
    glyph, position_map, mask = create_glyph_image(
        text, font, width, height, position, color
    )
    
    # Create canny edge from glyph
    canny = canny_edge(glyph)
    
    return {
        'glyph': glyph,
        'position': position_map,
        'mask': mask,
        'canny': canny,
        'text': text,
        'font_path': font_path,
        'font_size': font_size,
        'text_position': position,
        'text_color': color
    }


def save_training_sample(sample: Dict, output_dir: str, sample_id: int):
    """Save training sample to disk."""
    sample_dir = os.path.join(output_dir, f"sample_{sample_id:06d}")
    os.makedirs(sample_dir, exist_ok=True)
    
    # Save images
    Image.fromarray(sample['glyph']).save(
        os.path.join(sample_dir, 'glyph.png')
    )
    Image.fromarray(sample['position']).save(
        os.path.join(sample_dir, 'position.png')
    )
    Image.fromarray(sample['mask']).save(
        os.path.join(sample_dir, 'mask.png')
    )
    Image.fromarray(sample['canny']).save(
        os.path.join(sample_dir, 'canny.png')
    )
    
    # Save metadata
    metadata = {
        'text': sample['text'],
        'font_path': sample['font_path'],
        'font_size': sample['font_size'],
        'text_position': sample['text_position'],
        'text_color': sample['text_color']
    }
    
    with open(os.path.join(sample_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Prepare Arabic dataset for RepText training')
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='Output directory for training data')
    parser.add_argument('--font_dir', type=str, required=True,
                       help='Directory containing Arabic fonts')
    parser.add_argument('--text_file', type=str, default=None,
                       help='File containing Arabic text samples (one per line)')
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of training samples to generate')
    parser.add_argument('--width', type=int, default=1024,
                       help='Image width')
    parser.add_argument('--height', type=int, default=1024,
                       help='Image height')
    parser.add_argument('--min_font_size', type=int, default=60,
                       help='Minimum font size')
    parser.add_argument('--max_font_size', type=int, default=120,
                       help='Maximum font size')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load fonts
    print("Loading Arabic fonts...")
    fonts = get_arabic_fonts(args.font_dir)
    
    if not fonts:
        raise ValueError(f"No fonts found in {args.font_dir}")
    
    print(f"Found {len(fonts)} fonts")
    
    # Load or generate text samples
    if args.text_file and os.path.exists(args.text_file):
        print(f"Loading text samples from {args.text_file}...")
        text_samples = load_arabic_text_samples(args.text_file)
        print(f"Loaded {len(text_samples)} text samples")
    else:
        print("Generating random Arabic text samples...")
        text_samples = None
    
    # Generate training samples
    print(f"Generating {args.num_samples} training samples...")
    
    for i in tqdm(range(args.num_samples)):
        # Select random font
        font_path = random.choice(fonts)
        font_size = random.randint(args.min_font_size, args.max_font_size)
        
        # Get text
        if text_samples:
            text = random.choice(text_samples)
        else:
            text = generate_random_arabic_text()
        
        # Create sample
        try:
            sample = prepare_training_sample(
                text=text,
                font_path=font_path,
                font_size=font_size,
                width=args.width,
                height=args.height
            )
            
            # Save sample
            save_training_sample(sample, args.output_dir, i)
            
        except Exception as e:
            print(f"\nError creating sample {i}: {e}")
            continue
    
    print(f"\nDataset preparation complete! Samples saved to {args.output_dir}")
    
    # Save dataset info
    dataset_info = {
        'num_samples': args.num_samples,
        'image_size': [args.width, args.height],
        'font_size_range': [args.min_font_size, args.max_font_size],
        'num_fonts': len(fonts),
        'fonts': fonts
    }
    
    with open(os.path.join(args.output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=2)


if __name__ == '__main__':
    main()
