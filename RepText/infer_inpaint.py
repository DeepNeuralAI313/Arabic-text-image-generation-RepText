import torch
from diffusers.utils import load_image
from controlnet_flux import FluxControlNetModel
from pipeline_flux_controlnet_inpaint import FluxControlNetPipeline

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import re
import os

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

    Args:
        img: Input image (BGR format from cv2)
        dilate_edges: Whether to dilate edges for thicker lines (recommended for Arabic)
        dilation_kernel_size: Size of dilation kernel (3 = moderate, 5 = heavy)
    """
    low_threshold = 50
    high_threshold = 100
    edges = cv2.Canny(img, low_threshold, high_threshold)
    if dilate_edges:
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    edges = edges[:, :, None]
    edges = 255 - np.concatenate([edges, edges, edges], axis=2)
    return edges


def resize_img(input_image, max_side=1280, min_side=1024, size=None,
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


if __name__ == "__main__":

    base_model = "black-forest-labs/FLUX.1-dev"
    controlnet_model = "Shakker-Labs/RepText"

    controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
    controlnet_inpaint = FluxControlNetModel.from_pretrained("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", torch_dtype=torch.bfloat16)

    pipe = FluxControlNetPipeline.from_pretrained(
        base_model, controlnet=controlnet, controlnet_inpaint=controlnet_inpaint, torch_dtype=torch.bfloat16
    ).to("cuda")

    control_image_inpaint = load_image("assets/adam-jang-8pOTAtyd_Mc-unsplash.jpg")
    control_image_inpaint = resize_img(control_image_inpaint)
    width, height = control_image_inpaint.size

    ## set font -- prefer Bold/ExtraBold variants for Arabic
    font_path = "./assets/Arial_Unicode.ttf"  # use your own font (Bold preferred)
    font_size = 70  # it is recommended to use a font size >= 60
    font = ImageFont.truetype(font_path, font_size)

    ## set text content, position, color
    text_list = ["تخفيضات كبرى"]
    text_position_list = [(585, 375)]
    text_color_list = [(0, 255, 0)]

    ## --- Glyph Latent Weight (lambda2) ---
    ## Controls how much the glyph image influences the starting latent.
    ## Lower values (0.02-0.05): reduce ghosting/faint layer
    ## Higher values (0.5-0.9): stronger text from the start
    ## Set to 0.0 to disable glyph latent replication entirely (for debugging)
    glyph_latent_weight = 0.10  # experiment with: 0.0, 0.05, 0.10, 0.50, 0.90

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

        ### get bbox
        bbox = draw.textbbox(text_position, display_text, font=font)

        ### position condition
        control_position = np.zeros([height, width], dtype=np.uint8)
        control_position[bbox[1]-5:bbox[3]+5, bbox[0]-5:bbox[2]+5] = 255
        control_position = Image.fromarray(control_position.astype(np.uint8))
        control_position_list.append(control_position)

        ### regional mask -- tight bbox with small padding
        control_mask_np = np.zeros([height, width], dtype=np.uint8)
        control_mask_np[bbox[1]-5:bbox[3]+5, bbox[0]-5:bbox[2]+5] = 255
        control_mask = Image.fromarray(control_mask_np.astype(np.uint8))
        control_mask_list.append(control_mask)

        ### accumulate glyph
        control_glyph = np.array(control_image_glyph)
        control_glyph_all += control_glyph

        ### canny condition -- with edge dilation for thicker lines
        control_image = canny(
            cv2.cvtColor(np.array(control_image_glyph), cv2.COLOR_RGB2BGR),
            dilate_edges=True,
            dilation_kernel_size=3
        )
        control_image = Image.fromarray(cv2.cvtColor(control_image, cv2.COLOR_BGR2RGB))
        control_image_list.append(control_image)

    control_glyph_all = Image.fromarray(control_glyph_all.astype(np.uint8))
    control_glyph_all = control_glyph_all.convert("RGB")

    # --- PROMPT ---
    # For Arabic text: do NOT include the Arabic text in the prompt.
    # The ControlNet glyph image carries the text.
    # Use texture-aware descriptions for better text-surface integration.
    prompt = "a street photo with painted wall text, white painted lettering on rough brick surface, urban scene, filmfotos, film grain"

    # For English/Chinese text, you can still add the text to the prompt
    for text in text_list:
        if not contains_chinese(text) and not contains_arabic(text):
            prompt += f", '{text}'"

    print(f"Prompt: {prompt}")

    generator = torch.Generator(device="cuda").manual_seed(42)

    image = pipe(
        prompt,
        true_guidance_scale=3.5,  # set 1.0 to disable negative guidance
        # for text rendering
        control_image=control_image_list,  # canny
        control_position=control_position_list,  # position
        control_mask=control_mask_list,  # regional mask
        control_glyph=control_glyph_all,  # as init latent, optional, set to None if not used
        glyph_latent_weight=glyph_latent_weight,  # lambda2: glyph starting strength
        controlnet_conditioning_scale=1.5,  # increased from 1.0 for better glyph fidelity
        controlnet_conditioning_step=30,  # run ControlNet for 100% of steps
        # for inpainting
        control_image_inpaint=control_image_inpaint,
        control_mask_inpaint=control_mask,
        controlnet_conditioning_scale_inpaint=1.0,
        width=width,
        height=height,
        num_inference_steps=30,
        guidance_scale=3.5,
        generator=generator,
    ).images[0]

    if not os.path.exists("./results"):
        os.makedirs("./results")
    image.save(f"./results/result_inpaint.jpg")