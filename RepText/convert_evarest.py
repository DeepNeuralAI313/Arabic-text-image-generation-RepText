"""
EvArEST Dataset Converter for RepText Training

Converts the EvArEST Arabic Scene Text detection dataset into the RepText
training format. Each IMAGE becomes ONE training sample.

EvArEST data format (flat directory):
  img_N.jpg  - Scene image
  img_N.txt  - Annotation file, one word per line:
               x1,y1,x2,y2,x3,y3,x4,y4,Language,Text
               (four polygon corners, language tag, actual word)

RepText output format (one sample per image):
  sample_NNNNNN/
    target.png      - Full scene image resized to target_size
    glyph.png       - ALL text words rendered on black background
    position.png    - Position heatmap from ALL polygons
    mask.png        - Binary mask from ALL polygons
    canny.png       - Canny edge of the glyph image
    metadata.json   - All text annotations, prompt, etc.
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

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
    print("Warning: arabic_reshaper/python-bidi not installed. Arabic shaping skipped.")


def shape_arabic(text: str) -> str:
    """Shape Arabic text for correct RTL rendering."""
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


def parse_annotation_file(ann_path: str) -> List[Dict]:
    """
    Parse an EvArEST annotation file.

    Each line: x1,y1,x2,y2,x3,y3,x4,y4,Language,Text
    Returns list of dicts with keys: polygon, language, text
    """
    annotations = []
    with open(ann_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) < 10:
                continue

            try:
                coords = [int(float(p.strip())) for p in parts[:8]]
                polygon = [
                    (coords[0], coords[1]),
                    (coords[2], coords[3]),
                    (coords[4], coords[5]),
                    (coords[6], coords[7]),
                ]
                language = parts[8].strip()
                text = ','.join(parts[9:]).strip()

                if text and text != '###':
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


def auto_fit_font(text: str, bbox_w: int, bbox_h: int,
                  font_path: str, min_size: int = 12, max_size: int = 120) -> ImageFont.FreeTypeFont:
    """Find font size that fits text within the bbox dimensions."""
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

            if tw <= bbox_w * 1.3 and th <= bbox_h * 1.5:
                return font
            best_font = font
        except Exception:
            continue

    return best_font


# Scene description prompts for training variety
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


def convert_image_to_sample(
    image_path: str,
    ann_path: str,
    fonts: List[str],
    output_dir: str,
    sample_id: int,
    target_size: Tuple[int, int] = (512, 512),
) -> bool:
    """
    Convert one EvArEST image + its annotation file into ONE RepText sample.
    ALL text regions from the annotation are combined into a single sample.

    Returns True on success, False on failure.
    """
    try:
        # Parse ALL annotations for this image
        annotations = parse_annotation_file(ann_path)
        if not annotations:
            return False

        # Filter: keep only Arabic text
        arabic_anns = [a for a in annotations if is_arabic(a['text'])]
        if not arabic_anns:
            return False

        # Load scene image
        scene = Image.open(image_path).convert('RGB')
        orig_w, orig_h = scene.size

        # Scale factors to target size
        scale_x = target_size[0] / orig_w
        scale_y = target_size[1] / orig_h

        w, h = target_size

        # Resize scene image → target
        target_img = scene.resize(target_size, Image.LANCZOS)
        target_np = np.array(target_img)

        # Create blank images for combined outputs
        glyph_img = Image.new('RGB', (w, h), (0, 0, 0))
        glyph_draw = ImageDraw.Draw(glyph_img)

        mask_np = np.zeros((h, w), dtype=np.uint8)
        position_np = np.zeros((h, w), dtype=np.float64)

        # Collect all text for metadata
        all_texts = []
        all_annotations_meta = []

        # Select a random font for this image (all words same font)
        font_path = random.choice(fonts)

        for ann in arabic_anns:
            text = ann['text']
            polygon = ann['polygon']

            # Scale polygon to target size
            scaled_polygon = [(int(x * scale_x), int(y * scale_y)) for x, y in polygon]
            scaled_bbox = polygon_to_bbox(scaled_polygon)
            bbox_w = scaled_bbox[2] - scaled_bbox[0]
            bbox_h = scaled_bbox[3] - scaled_bbox[1]

            # Skip tiny regions
            if bbox_w < 5 or bbox_h < 5:
                continue

            # --- Render glyph text at the polygon location ---
            font = auto_fit_font(text, bbox_w, bbox_h, font_path)
            display_text = shape_arabic(text)

            # Get rendered text size to center within bbox
            tb = glyph_draw.textbbox((0, 0), display_text, font=font)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]

            # Center text in the scaled bbox
            cx = (scaled_bbox[0] + scaled_bbox[2]) // 2
            cy = (scaled_bbox[1] + scaled_bbox[3]) // 2
            pos_x = cx - tw // 2
            pos_y = cy - th // 2

            # Clamp
            pos_x = max(0, min(pos_x, w - tw - 1))
            pos_y = max(0, min(pos_y, h - th - 1))

            glyph_draw.text((pos_x, pos_y), display_text, font=font, fill=(255, 255, 255))

            # --- Add polygon to mask ---
            pts = np.array(scaled_polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask_np, [pts], 255)

            # --- Add polygon to position heatmap ---
            region_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(region_mask, [pts], 255)
            dist = cv2.distanceTransform(region_mask, cv2.DIST_L2, 5)
            if dist.max() > 0:
                dist = dist / dist.max()
            position_np = np.maximum(position_np, dist)

            all_texts.append(text)
            all_annotations_meta.append({
                'text': text,
                'language': ann['language'],
                'original_polygon': ann['polygon'],
                'scaled_polygon': scaled_polygon,
                'scaled_bbox': list(scaled_bbox),
            })

        # If no valid text regions were processed, skip
        if not all_texts:
            return False

        # Dilate mask for padding
        kernel = np.ones((11, 11), np.uint8)
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)

        # Position heatmap to RGB
        position_uint8 = (position_np * 255).astype(np.uint8)
        position_rgb = cv2.cvtColor(position_uint8, cv2.COLOR_GRAY2RGB)

        # Glyph to numpy
        glyph_np = np.array(glyph_img)

        # Canny edge of glyph
        gray = cv2.cvtColor(glyph_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 100)
        edge_kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, edge_kernel, iterations=1)
        canny_np = 255 - np.stack([edges, edges, edges], axis=2)

        # Save sample
        sample_dir = os.path.join(output_dir, f"sample_{sample_id:06d}")
        os.makedirs(sample_dir, exist_ok=True)

        Image.fromarray(target_np).save(os.path.join(sample_dir, 'target.png'))
        Image.fromarray(glyph_np).save(os.path.join(sample_dir, 'glyph.png'))
        Image.fromarray(position_rgb).save(os.path.join(sample_dir, 'position.png'))
        Image.fromarray(mask_np).save(os.path.join(sample_dir, 'mask.png'))
        Image.fromarray(canny_np).save(os.path.join(sample_dir, 'canny.png'))

        # Combined text for prompt and OCR loss
        combined_text = ' '.join(all_texts)
        prompt = random.choice(SCENE_PROMPTS)

        metadata = {
            'text': combined_text,
            'texts': all_texts,
            'num_words': len(all_texts),
            'font_path': font_path,
            'font_size': 0,  # varies per word
            'prompt': prompt,
            'source_image': os.path.basename(image_path),
            'source_annotation': os.path.basename(ann_path),
            'original_size': [orig_w, orig_h],
            'target_size': list(target_size),
            'annotations': all_annotations_meta,
        }

        with open(os.path.join(sample_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return True

    except Exception as e:
        print(f"  Error processing {os.path.basename(image_path)}: {e}")
        return False


def find_image_annotation_pairs(input_dir: str) -> List[Tuple[str, str]]:
    """
    Find img_N.jpg + img_N.txt pairs in the EvArEST directory.
    Handles both flat and nested directory structures.
    """
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    pairs = []

    input_path = Path(input_dir)

    # Collect all images indexed by stem
    images = {}
    for f in input_path.rglob('*'):
        if f.suffix.lower() in image_exts:
            images[f.stem] = str(f)

    # Find matching .txt annotations
    for f in input_path.rglob('*.txt'):
        stem = f.stem
        if stem in images:
            pairs.append((images[stem], str(f)))

    # Sort for deterministic output
    pairs.sort(key=lambda x: x[0])
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description='Convert EvArEST dataset to RepText training format'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing EvArEST data (img_N.jpg + img_N.txt)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for RepText training samples')
    parser.add_argument('--font_dir', type=str, default='./arabic_fonts',
                        help='Directory containing Arabic font files')
    parser.add_argument('--target_size', type=int, nargs=2, default=[512, 512],
                        help='Target image size (width height)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max number of samples (None = all images)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Preview without writing files')
    args = parser.parse_args()

    print("=" * 60)
    print("EvArEST → RepText Dataset Converter")
    print("=" * 60)
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Size:   {args.target_size[0]}x{args.target_size[1]}")
    print()

    # Find fonts
    fonts = get_arabic_fonts(args.font_dir)
    if not fonts:
        fallback_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "./assets/Arial_Unicode.ttf",
        ]
        for fp in fallback_paths:
            if os.path.exists(fp):
                fonts.append(fp)
        if not fonts:
            print("❌ No fonts found! Please provide --font_dir with .ttf files")
            return
    print(f"✓ Found {len(fonts)} font(s)")

    # Find image-annotation pairs
    pairs = find_image_annotation_pairs(args.input_dir)
    if not pairs:
        print(f"\n❌ No image-annotation pairs found in {args.input_dir}")
        print("Expected: img_N.jpg + img_N.txt files in the same directory")
        return
    print(f"✓ Found {len(pairs)} image-annotation pair(s)")

    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        total_arabic = 0
        total_english = 0
        for img, ann in pairs[:10]:
            annotations = parse_annotation_file(ann)
            arabic_count = sum(1 for a in annotations if is_arabic(a['text']))
            english_count = len(annotations) - arabic_count
            total_arabic += arabic_count
            total_english += english_count
            print(f"  {os.path.basename(img)} → {arabic_count} Arabic, {english_count} English word(s)")
        if len(pairs) > 10:
            print(f"  ... and {len(pairs) - 10} more image(s)")
        print(f"\nTotal (first {min(10, len(pairs))} images): {total_arabic} Arabic, {total_english} English words")
        print(f"Would create {min(len(pairs), args.max_samples or len(pairs))} training samples (1 per image)")
        return

    # Convert
    os.makedirs(args.output_dir, exist_ok=True)
    target_size = tuple(args.target_size)

    success = 0
    skipped = 0

    limit = args.max_samples or len(pairs)
    pbar = tqdm(pairs[:limit], desc="Converting", unit="img")

    for sample_id, (img_path, ann_path) in enumerate(pbar):
        ok = convert_image_to_sample(
            img_path, ann_path, fonts, args.output_dir, sample_id, target_size
        )
        if ok:
            success += 1
        else:
            skipped += 1
        pbar.set_postfix(ok=success, skip=skipped)

    # Save dataset info
    info = {
        'total_samples': success,
        'skipped': skipped,
        'total_images': len(pairs),
        'target_size': list(target_size),
        'source': 'EvArEST',
        'num_fonts': len(fonts),
        'approach': 'one_sample_per_image',
    }
    with open(os.path.join(args.output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n✓ Conversion complete!")
    print(f"  Images processed: {len(pairs[:limit])}")
    print(f"  Samples created:  {success}  (1 per image, all words combined)")
    print(f"  Skipped:          {skipped}  (no Arabic text or errors)")
    print(f"  Output:           {args.output_dir}")


if __name__ == '__main__':
    main()
