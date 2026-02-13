"""
EvArEST Dataset Converter for RepText Training

Converts the EvArEST Arabic Scene Text detection dataset into the RepText
training format. Each annotated text region in a scene image becomes a
training sample with: target, glyph, position, mask, canny, and metadata.

EvArEST format:
  - Images: *.jpg / *.png
  - Annotations: *.txt (one word per line)
    Each line: x1,y1,x2,y2,x3,y3,x4,y4,language,text
    (four corners of a polygon, language tag, word text)

RepText output format:
  sample_NNNNNN/
    target.png      - Full scene image (ground truth for denoising loss)
    glyph.png       - Rendered Arabic text on black background
    position.png    - Position heatmap from polygon
    mask.png        - Binary mask from polygon
    canny.png       - Canny edge of glyph
    metadata.json   - Text, bbox, prompt, etc.
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from tqdm import tqdm

# Arabic text shaping
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_SHAPING_AVAILABLE = True
except ImportError:
    ARABIC_SHAPING_AVAILABLE = False
    print("Warning: arabic_reshaper/python-bidi not installed. Arabic text shaping will be skipped.")


def shape_arabic(text: str) -> str:
    """Shape Arabic text for correct RTL display."""
    if not ARABIC_SHAPING_AVAILABLE:
        return text
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:
        return text


def is_arabic(text: str) -> bool:
    """Check if text contains Arabic characters."""
    import re
    return bool(re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))


def get_arabic_fonts(font_dir: str) -> List[str]:
    """Get available Arabic font files."""
    fonts = []
    if os.path.exists(font_dir):
        for f in os.listdir(font_dir):
            if f.lower().endswith(('.ttf', '.otf')):
                fonts.append(os.path.join(font_dir, f))
    return fonts


def parse_evarest_annotation(ann_path: str) -> List[Dict]:
    """
    Parse EvArEST annotation file.

    Each line format: x1,y1,x2,y2,x3,y3,x4,y4,language[,text]
    Some lines may have the text, some may not.
    Language is typically 'Arabic' or 'English'.

    Returns:
        List of dicts with keys: polygon, language, text
    """
    annotations = []
    with open(ann_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')

            # Need at least 8 coords + language = 9 parts
            if len(parts) < 9:
                continue

            try:
                coords = [int(float(p.strip())) for p in parts[:8]]
                polygon = [
                    (coords[0], coords[1]),  # top-left
                    (coords[2], coords[3]),  # top-right
                    (coords[4], coords[5]),  # bottom-right
                    (coords[6], coords[7]),  # bottom-left
                ]
                language = parts[8].strip()
                text = ','.join(parts[9:]).strip() if len(parts) > 9 else ""

                annotations.append({
                    'polygon': polygon,
                    'language': language,
                    'text': text,
                })
            except (ValueError, IndexError):
                continue

    return annotations


def polygon_to_bbox(polygon: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """Convert polygon to axis-aligned bounding box (x1, y1, x2, y2)."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def create_polygon_mask(polygon: List[Tuple[int, int]], width: int, height: int,
                        padding: int = 5) -> np.ndarray:
    """Create binary mask from polygon with padding."""
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    if padding > 0:
        kernel = np.ones((padding * 2 + 1, padding * 2 + 1), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def create_position_heatmap(polygon: List[Tuple[int, int]], width: int, height: int) -> np.ndarray:
    """Create position heatmap centered on the polygon region."""
    mask = create_polygon_mask(polygon, width, height, padding=0)
    # Use distance transform for smooth heatmap
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist = (dist / dist.max() * 255).astype(np.uint8)
    else:
        dist = mask
    return dist


def render_glyph(text: str, font: ImageFont.FreeTypeFont, width: int, height: int,
                 bbox: Tuple[int, int, int, int],
                 color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Render Arabic text on black background, positioned to match the bbox.

    The glyph is centered within the bbox region of the full image.
    """
    glyph_img = Image.new('RGB', (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(glyph_img)

    # Shape Arabic text
    display_text = shape_arabic(text)

    # Get text size
    text_bbox = draw.textbbox((0, 0), display_text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    # Center text within the target bbox
    target_cx = (bbox[0] + bbox[2]) // 2
    target_cy = (bbox[1] + bbox[3]) // 2
    pos_x = target_cx - text_w // 2
    pos_y = target_cy - text_h // 2

    # Clamp to image bounds
    pos_x = max(0, min(pos_x, width - text_w))
    pos_y = max(0, min(pos_y, height - text_h))

    draw.text((pos_x, pos_y), display_text, font=font, fill=color)
    return np.array(glyph_img)


def canny_edge(img: np.ndarray, dilate: bool = True, kernel_size: int = 3) -> np.ndarray:
    """Apply Canny edge detection with optional dilation."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    edges = cv2.Canny(gray, 50, 100)
    if dilate:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    edges_rgb = 255 - np.stack([edges, edges, edges], axis=2)
    return edges_rgb


# Scene description prompts for variety in training
SCENE_PROMPTS = [
    "a photo of Arabic text on a street sign, urban scene, realistic",
    "Arabic calligraphy on a shop storefront, warm lighting, photorealistic",
    "text on a commercial billboard, city background, high quality photo",
    "Arabic script on a wall plaque, stone surface, natural lighting",
    "sign with Arabic writing, outdoor scene, professional photography",
    "Arabic lettering on a banner, colorful market scene, clear text",
    "printed Arabic text on a poster, indoor lighting, sharp detail",
    "Arabic store sign, sunset lighting, architectural context, realistic",
    "hand-painted Arabic text on a wall, textured surface, daylight",
    "Arabic road sign, clear sky background, sharp lettering, photography",
    "Arabic restaurant menu sign, neon lighting, urban night scene",
    "Arabic text on glass window, reflections, soft bokeh background",
    "Arabic graffiti on concrete wall, street art style, natural light",
    "formal Arabic signage on marble surface, elegant architecture",
    "Arabic text on vehicle, transportation context, daytime photo",
]


def auto_fit_font(text: str, bbox: Tuple[int, int, int, int],
                  font_path: str, min_size: int = 16, max_size: int = 120) -> ImageFont.FreeTypeFont:
    """Automatically find font size that fits text within the bbox."""
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]

    display_text = shape_arabic(text) if is_arabic(text) else text
    best_font = ImageFont.truetype(font_path, min_size)

    for size in range(max_size, min_size - 1, -2):
        try:
            font = ImageFont.truetype(font_path, size)
            tmp = Image.new('RGB', (1, 1))
            draw = ImageDraw.Draw(tmp)
            tb = draw.textbbox((0, 0), display_text, font=font)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]

            if tw <= bbox_w * 1.2 and th <= bbox_h * 1.5:
                return font
            best_font = font
        except Exception:
            continue

    return best_font


def convert_evarest_sample(
    image_path: str,
    annotation: Dict,
    fonts: List[str],
    output_dir: str,
    sample_id: int,
    target_size: Tuple[int, int] = (512, 512),
) -> bool:
    """
    Convert one EvArEST annotation into a RepText training sample.

    Returns True on success, False on failure.
    """
    try:
        # Load scene image
        scene = Image.open(image_path).convert('RGB')
        orig_w, orig_h = scene.size

        # Get polygon and text from annotation
        polygon = annotation['polygon']
        text = annotation.get('text', '')

        if not text or len(text.strip()) < 1:
            return False

        # Only process Arabic text
        if not is_arabic(text):
            return False

        # Compute bbox on original image
        bbox = polygon_to_bbox(polygon)

        # Scale factors to target size
        scale_x = target_size[0] / orig_w
        scale_y = target_size[1] / orig_h

        # Scale polygon and bbox
        scaled_polygon = [(int(x * scale_x), int(y * scale_y)) for x, y in polygon]
        scaled_bbox = (
            int(bbox[0] * scale_x), int(bbox[1] * scale_y),
            int(bbox[2] * scale_x), int(bbox[3] * scale_y)
        )

        # Resize scene image
        target_img = scene.resize(target_size, Image.LANCZOS)
        target_np = np.array(target_img)

        w, h = target_size

        # Select random font
        font_path = random.choice(fonts)
        font = auto_fit_font(text, scaled_bbox, font_path)

        # Create glyph
        glyph_np = render_glyph(text, font, w, h, scaled_bbox)

        # Create position heatmap
        position_np = create_position_heatmap(scaled_polygon, w, h)
        position_rgb = cv2.cvtColor(position_np, cv2.COLOR_GRAY2RGB)

        # Create mask from polygon
        mask_np = create_polygon_mask(scaled_polygon, w, h, padding=5)

        # Create canny edges
        canny_np = canny_edge(glyph_np, dilate=True, kernel_size=3)

        # Select a random prompt
        prompt = random.choice(SCENE_PROMPTS)

        # Save sample
        sample_dir = os.path.join(output_dir, f"sample_{sample_id:06d}")
        os.makedirs(sample_dir, exist_ok=True)

        Image.fromarray(target_np).save(os.path.join(sample_dir, 'target.png'))
        Image.fromarray(glyph_np).save(os.path.join(sample_dir, 'glyph.png'))
        Image.fromarray(position_rgb).save(os.path.join(sample_dir, 'position.png'))
        Image.fromarray(mask_np).save(os.path.join(sample_dir, 'mask.png'))
        Image.fromarray(canny_np).save(os.path.join(sample_dir, 'canny.png'))

        metadata = {
            'text': text,
            'font_path': font_path,
            'font_size': font.size,
            'prompt': prompt,
            'source_image': os.path.basename(image_path),
            'language': annotation.get('language', 'Arabic'),
            'original_polygon': annotation['polygon'],
            'scaled_polygon': scaled_polygon,
            'scaled_bbox': list(scaled_bbox),
        }

        with open(os.path.join(sample_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return True

    except Exception as e:
        print(f"  Error processing sample {sample_id}: {e}")
        return False


def find_image_annotation_pairs(input_dir: str) -> List[Tuple[str, str]]:
    """
    Find matching image-annotation pairs in the EvArEST directory.

    EvArEST typically stores:
      - Images in a folder (e.g., images/)
      - Annotations as .txt files with same base name

    We search recursively for (image, annotation) pairs.
    """
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    pairs = []

    input_path = Path(input_dir)

    # Collect all images
    images = {}
    for f in input_path.rglob('*'):
        if f.suffix.lower() in image_exts:
            images[f.stem] = str(f)

    # Find matching annotations
    for f in input_path.rglob('*.txt'):
        stem = f.stem
        # EvArEST sometimes prefixes annotations with "gt_"
        clean_stem = stem.replace('gt_', '').replace('GT_', '')
        if stem in images:
            pairs.append((images[stem], str(f)))
        elif clean_stem in images:
            pairs.append((images[clean_stem], str(f)))
        # Also try: annotation name matches image name
        elif f'img_{clean_stem}' in images:
            pairs.append((images[f'img_{clean_stem}'], str(f)))

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description='Convert EvArEST dataset to RepText training format'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing EvArEST data (images + annotations)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for RepText training samples')
    parser.add_argument('--font_dir', type=str, default='./arabic_fonts',
                        help='Directory containing Arabic font files')
    parser.add_argument('--target_size', type=int, nargs=2, default=[512, 512],
                        help='Target image size (width height)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max number of samples to convert (None = all)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be done without writing files')
    args = parser.parse_args()

    print("=" * 60)
    print("EvArEST → RepText Dataset Converter")
    print("=" * 60)

    # Find fonts
    fonts = get_arabic_fonts(args.font_dir)
    if not fonts:
        # Fallback: try system fonts
        fallback_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "./assets/Arial_Unicode.ttf",
        ]
        for fp in fallback_paths:
            if os.path.exists(fp):
                fonts.append(fp)
        if not fonts:
            print("❌ No fonts found! Please run: python download_arabic_fonts.py")
            return
    print(f"✓ Found {len(fonts)} font(s)")

    # Find image-annotation pairs
    pairs = find_image_annotation_pairs(args.input_dir)
    if not pairs:
        # If no pairs found, try listing all files for debugging
        print(f"\n❌ No image-annotation pairs found in {args.input_dir}")
        print("\nDirectory contents:")
        for item in sorted(Path(args.input_dir).rglob('*'))[:20]:
            print(f"  {item}")
        print("\nExpected: image files (*.jpg, *.png) with matching *.txt annotation files")
        print("Download EvArEST dataset from:")
        print("  Training: https://drive.google.com/file/d/1a1Jf12nyIDswunky5kLM4JishRj2_4Jy/view")
        print("  Test:     https://drive.google.com/file/d/15jWxmZb9zoKHys40Cuz-57kV2PTO-cvH/view")
        return
    print(f"✓ Found {len(pairs)} image-annotation pair(s)")

    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        for img, ann in pairs[:5]:
            annotations = parse_evarest_annotation(ann)
            arabic_count = sum(1 for a in annotations if is_arabic(a.get('text', '')))
            print(f"  {os.path.basename(img)} → {arabic_count} Arabic word(s) (of {len(annotations)} total)")
        if len(pairs) > 5:
            print(f"  ... and {len(pairs) - 5} more image(s)")
        return

    # Convert
    os.makedirs(args.output_dir, exist_ok=True)
    sample_id = 0
    success = 0
    skipped = 0

    target_size = tuple(args.target_size)

    pbar = tqdm(pairs, desc="Converting images", unit="img")
    for img_path, ann_path in pbar:
        annotations = parse_evarest_annotation(ann_path)

        for ann in annotations:
            if args.max_samples and sample_id >= args.max_samples:
                break

            ok = convert_evarest_sample(
                img_path, ann, fonts, args.output_dir, sample_id, target_size
            )
            if ok:
                success += 1
            else:
                skipped += 1
            sample_id += 1
            pbar.set_postfix(ok=success, skip=skipped)

        if args.max_samples and sample_id >= args.max_samples:
            break

    # Save dataset info
    info = {
        'total_samples': success,
        'skipped': skipped,
        'target_size': list(target_size),
        'source': 'EvArEST',
        'num_fonts': len(fonts),
    }
    with open(os.path.join(args.output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n✓ Conversion complete!")
    print(f"  Samples created: {success}")
    print(f"  Skipped: {skipped}")
    print(f"  Output: {args.output_dir}")


if __name__ == '__main__':
    main()
