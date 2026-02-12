import torch
from controlnet_flux import FluxControlNetModel
from pipeline_flux_controlnet import FluxControlNetPipeline

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import re
import os
from masking_utils import create_contour_mask, create_contour_position, create_mask

# --- Arabic text shaping (required for correct rendering) ---
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_SHAPING_AVAILABLE = True
except ImportError:
    ARABIC_SHAPING_AVAILABLE = False
    print("Warning: arabic_reshaper/python-bidi not installed. Arabic shaping will be skipped.")


def shape_arabic(text):
    """Shape Arabic text for correct RTL display and ligatures."""
    if not ARABIC_SHAPING_AVAILABLE:
        return text
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


def contains_arabic(text):
    """Check if text contains Arabic characters."""
    if re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text):
        return True
    return False


def contains_chinese(text):
    if re.search(r'[\u4e00-\u9fff]', text):
        return True
    return False


def canny(img, dilate_edges=True, dilation_kernel_size=3):
    """
    Apply Canny edge detection with optional dilation for thicker edges.

    Thicker edges create stronger gradients that the diffusion model treats
    as physical boundaries, improving Arabic text fidelity.

    Args:
        img: Input image (BGR format from cv2)
        dilate_edges: Whether to dilate edges for thicker lines (recommended for Arabic)
        dilation_kernel_size: Size of dilation kernel (3 = moderate, 5 = heavy)
    """
    low_threshold = 50
    high_threshold = 100
    edges = cv2.Canny(img, low_threshold, high_threshold)
    # Dilate edges to make them thicker -- stronger signal for the diffusion model
    if dilate_edges:
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    edges = edges[:, :, None]
    edges = 255 - np.concatenate([edges, edges, edges], axis=2)
    return edges


# ============================================================================
# 10 Texture-Aware Prompts for Arabic Text Rendering
# ============================================================================
# These prompts describe how text integrates with the surface/scene.
# DO NOT include the actual Arabic text in the prompt -- only the glyph image
# carries the text. The prompt describes the scene + text-surface interaction.
#
# 1. "a large metallic billboard on a city rooftop at sunset, embossed lettering catching golden light, cinematic, realistic"
# 2. "a wooden shop sign hanging above a market stall, text carved deeply into dark oak wood, warm lighting, photorealistic"
# 3. "a glowing neon sign mounted on a brick wall at night, vibrant pink and blue neon tubes forming letters, urban photography"
# 4. "a stone monument in a desert landscape, ancient calligraphy chiseled into smooth sandstone, golden hour, high detail"
# 5. "a frosted glass storefront window, text etched into the glass with light glowing from behind, soft bokeh background"
# 6. "a luxury perfume bottle label, gold foil lettering pressed into matte black paper, studio lighting, product photography"
# 7. "a highway road sign with reflective green surface, white painted text, night scene with car headlights, realistic"
# 8. "a coffee shop chalkboard menu, white chalk text written on dark slate board, cozy warm interior, shallow depth of field"
# 9. "a large banner draped across a building facade, printed text on white fabric, daytime urban scene, professional photo"
# 10. "a ceramic tile mosaic on a mosque wall, intricate geometric patterns surrounding the text area, natural daylight, architectural photography"
# ============================================================================


if __name__ == "__main__":

    base_model = "black-forest-labs/FLUX.1-dev"
    controlnet_model = "Shakker-Labs/RepText"

    controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
    pipe = FluxControlNetPipeline.from_pretrained(
        base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
    ).to("cuda")

    ## set resolution
    width, height = 1024, 1024

    ## set font -- prefer Bold/ExtraBold variants for Arabic
    font_path = "./assets/Arial_Unicode.ttf"  # use your own font (Bold preferred)
    font_size = 80  # it is recommended to use a font size >= 60
    font = ImageFont.truetype(font_path, font_size)

    ## set text content, position, color
    ## For Arabic text, use shape_arabic() to get correct rendering
    text_list = ["تخفيضات كبرى"]
    text_position_list = [(300, 400)]
    text_color_list = [(255, 255, 255)]

    ## --- Glyph Latent Weight (lambda2) ---
    ## Controls how much the glyph image influences the starting latent.
    ## Lower values (0.02-0.05): reduce ghosting/faint layer
    ## Higher values (0.5-0.9): stronger text from the start, may reduce hallucinations
    ## Set to 0.0 to disable glyph latent replication entirely (for debugging)
    glyph_latent_weight = 0.10  # experiment with: 0.0, 0.05, 0.10, 0.50, 0.90

    ## --- Mask Mode ---
    ## "contour" = tight mask following actual letter shapes (recommended)
    ## "bbox" = traditional rectangular bounding box mask
    mask_mode = "contour"
    mask_padding = 3  # pixels of padding around text (3 = tight, 5 = moderate)

    ## set controlnet conditions
    control_image_list = []  # canny list
    control_position_list = []  # position list
    control_mask_list = []  # regional mask list
    control_glyph_all = np.zeros([height, width, 3], dtype=np.uint8)  # all glyphs

    ## handle each line of text
    for text, text_position, text_color in zip(text_list, text_position_list, text_color_list):

        ## Shape Arabic text for correct display
        display_text = shape_arabic(text) if contains_arabic(text) else text

        ### glyph image, render text to black background
        control_image_glyph = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(control_image_glyph)
        draw.text(text_position, display_text, font=font, fill=text_color)

        glyph_np = np.array(control_image_glyph)

        ### get bbox (still needed for fallback and position)
        bbox = draw.textbbox(text_position, display_text, font=font)

        ### position condition -- contour-based or bbox-based
        if mask_mode == "contour":
            pos_np = create_contour_position(glyph_np, padding=mask_padding)
        else:
            pos_np = np.zeros([height, width], dtype=np.uint8)
            pos_np[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255
        control_position = Image.fromarray(pos_np)
        control_position_list.append(control_position)

        ### regional mask -- contour-based or bbox-based
        mask_np = create_mask(
            glyph_np, bbox, width, height,
            mode=mask_mode, padding=mask_padding
        )
        control_mask = Image.fromarray(mask_np)
        control_mask_list.append(control_mask)

        ### accumulate glyph
        control_glyph_all += glyph_np

        ### canny condition -- with edge dilation for thicker lines
        control_image = canny(
            cv2.cvtColor(glyph_np, cv2.COLOR_RGB2BGR),
            dilate_edges=True,
            dilation_kernel_size=3
        )
        control_image = Image.fromarray(cv2.cvtColor(control_image, cv2.COLOR_BGR2RGB))
        control_image_list.append(control_image)

    control_glyph_all = Image.fromarray(control_glyph_all.astype(np.uint8))
    control_glyph_all = control_glyph_all.convert("RGB")

    ## Save debug images to verify mask quality
    if not os.path.exists("./results"):
        os.makedirs("./results")
    control_glyph_all.save("./results/debug_glyph.jpg")
    if control_mask_list:
        control_mask_list[0].save("./results/debug_mask.jpg")
    if control_position_list:
        control_position_list[0].save("./results/debug_position.jpg")
    if control_image_list:
        control_image_list[0].save("./results/debug_canny.jpg")

    # --- PROMPT ---
    # For Arabic text: do NOT include the Arabic text in the prompt.
    # The ControlNet glyph image carries the text -- the prompt describes only the scene.
    # Use texture-aware descriptions for better text-surface integration.
    prompt = "a large metallic billboard on a city rooftop at sunset, embossed lettering catching golden light, cinematic, realistic"

    # For English/Chinese text, you can still add the text to the prompt if desired
    for text in text_list:
        if not contains_chinese(text) and not contains_arabic(text):
            prompt += f", '{text}'"

    print(f"Prompt: {prompt}")

    generator = torch.Generator(device="cuda").manual_seed(42)

    image = pipe(
        prompt,
        control_image=control_image_list,  # canny
        control_position=control_position_list,  # position
        control_mask=control_mask_list,  # regional mask
        control_glyph=control_glyph_all,  # as init latent, optional, set to None if not used
        glyph_latent_weight=glyph_latent_weight,  # lambda2: glyph starting strength
        controlnet_conditioning_scale=1.5,  # increased from 1.0 for better glyph fidelity
        controlnet_conditioning_step=30,  # run ControlNet for 100% of steps (30/30)
        width=width,
        height=height,
        num_inference_steps=30,
        guidance_scale=3.5,
        generator=generator,
    ).images[0]

    if not os.path.exists("./results"):
        os.makedirs("./results")
    image.save(f"./results/result.jpg")