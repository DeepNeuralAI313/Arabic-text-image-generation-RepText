# RepText Arabic Text Generation -- Status and Improvement Plan

This document tracks all suggestions for improving Arabic text rendering with RepText,
what has been implemented, what is missing, and the concrete steps to move forward.

---

## Current Codebase Overview

The project contains:
- **infer.py** -- Original RepText inference (Chinese/English focused)
- **infer_simple.py** -- Simplified Arabic inference script
- **infer_inpaint.py** -- Inpainting-based inference
- **pipeline_flux_controlnet.py** -- Core FLUX ControlNet pipeline (contains glyph latent replication logic)
- **train_arabic.py** -- Training script with ControlNet + optional OCR loss
- **prepare_arabic_dataset.py** -- Dataset generation with Arabic shaping
- **train_config.yaml** -- Training configuration

---

## Two Core Failure Modes

1. **Ghost / faint correct layer** -- The correct Arabic text appears but is faint, with the model drawing over it
2. **Extra Arabic-like hallucinated text** -- Random Arabic-looking squiggles appear outside or on top of the intended text

Root cause: The diffusion process competes with the ControlNet signal. The model sees the correct "stencil" but treats it as noise, drawing over it with hallucinated text.

---

## Suggestion-by-Suggestion Analysis

### Suggestion 1 -- Remove Arabic Text from Prompt

**Goal:** Stop FLUX/T5 from generating its own pseudo-text by keeping Arabic only in the glyph image, not the prompt.

**Status: PARTIALLY DONE**

What is done:
- `infer_simple.py` passes an empty prompt (`""`) at line 136

What is missing:
- `infer.py` (lines 109-111) and `infer_inpaint.py` still inject non-Chinese text into the prompt:
  ```python
  for text in text_list:
      if not contains_chinese(text):
          prompt += f", '{text}'"
  ```
  This means Arabic text leaks into the prompt, causing T5 to "help" and hallucinate its own text layer.
- No negative prompt is used anywhere (e.g., "extra text, watermark, gibberish letters, additional writing")

What to do:
- Remove the Arabic text injection loop from `infer.py` and `infer_inpaint.py`
- For Arabic, the prompt should describe only the scene (e.g., "a store with a big billboard, cinematic, realistic, evening light")
- Add negative prompt support where the pipeline allows it

---

### Suggestion 2 -- Tune Glyph-Latent Replication Blend (lambda2)

**Goal:** Reduce the "double exposure" effect where glyph latent replication is too visible AND the model generates text on top.

**Status: NOT TUNABLE (hardcoded)**

What is done:
- The glyph latent replication logic exists in `pipeline_flux_controlnet.py` inside `prepare_latents_reptext()` (line 653):
  ```python
  result[glyph_mask] = 0.10 * image_latents[glyph_mask] + 1.0 * noise[glyph_mask]
  ```
  Here lambda2 = 0.10 and lambda1 = 1.0 (they do not sum to 1.0, this is an additive blend).

What is missing:
- lambda2 is hardcoded -- there is no way to change it without editing the source code
- The first suggestion said to try lambda2 = 0.02 to 0.05 or even 0.0 for debugging
- The second suggestion says the OPPOSITE: increase lambda2 to 0.9 or higher to make the correct text the "boss" from the start

These two suggestions contradict each other, so the real answer is: **this needs experimentation**.

What to do:
- Expose lambda2 as a parameter in the pipeline `__call__()` method (e.g., `glyph_latent_weight`)
- Run experiments with multiple values: 0.0, 0.05, 0.10 (current), 0.5, 0.9
- If lambda2 = 0.0 removes ghosting but text disappears, increase gradually
- If lambda2 = 0.9 makes text stronger but introduces artifacts, decrease gradually

---

### Suggestion 3 -- Stricter Regional Masking

**Goal:** Prevent hallucinations outside the intended text area by tightening the mask.

**Status: BASIC IMPLEMENTATION EXISTS**

What is done:
- `infer.py` uses mask padding of +/- 5 px (lines 94-95)
- `prepare_arabic_dataset.py` uses mask padding of 10 px (line 146)
- The regional mask IS multiplied at each ControlNet injection point in the pipeline (lines 1061-1069) -- both `controlnet_block_samples` and `controlnet_single_block_samples` are masked

What is missing:
- For multi-line text, a single large bbox is used instead of per-line tight bboxes
- No "shrink-wrap" masking that follows the actual character contours
- The second suggestion emphasizes that a large mask box gives the AI "permission" to draw extra squiggles inside it

What to do:
- For multi-line text, split into separate bboxes per line instead of one large region
- Reduce mask padding from 10 px to 5 px or less in the dataset preparation
- Consider generating masks from the actual text pixels (dilated by 3-5 px) instead of rectangular bboxes

---

### Suggestion 4 -- Lower CFG, Increase ControlNet Scale

**Goal:** Reduce hallucinations (caused by high CFG) while strengthening glyph fidelity (with higher ControlNet scale).

**Status: CFG IS GOOD, CONTROLNET SCALE NEEDS INCREASE**

Current values:
| Parameter                        | infer.py | infer_simple.py | Suggested |
|----------------------------------|----------|-----------------|-----------|
| guidance_scale (CFG)             | 3.5      | 0.0             | 3.0-5.0   |
| controlnet_conditioning_scale    | 1.0      | 1.0             | 1.2-2.0   |
| controlnet_conditioning_step     | 30       | 20              | see below |

What is done:
- CFG is already in a reasonable range (3.5 in `infer.py`)

What is missing:
- `controlnet_conditioning_scale` is 1.0 everywhere -- needs to be increased to 1.2-2.0
- The second suggestion says to keep ControlNet guidance active for longer (80% of steps instead of 50%)
- Currently `controlnet_conditioning_step = 30` with `num_inference_steps = 30`, meaning guidance runs the full 30 steps. In `infer_simple.py` it is 20 out of 30 steps (67%). The suggestion says to try 80% or more.

What to do:
- Increase `controlnet_conditioning_scale` to 1.5 as a starting point, test up to 2.0
- Ensure `controlnet_conditioning_step` is at least 80% of `num_inference_steps`
- For a 30-step run, use `controlnet_conditioning_step = 24` or higher

---

### Suggestion 5 -- Arabic Glyph Rendering (Shaping + Bidi)

**Goal:** Ensure the Arabic text in the glyph template is properly shaped with correct ligatures and RTL ordering.

**Status: DONE**

What is done:
- `prepare_arabic_dataset.py` has `shape_arabic_text()` (lines 20-37) using `arabic_reshaper` + `python-bidi`
- Font sizes range from 60 to 120 (large enough for Arabic details)
- Arabic fonts are loaded from `arabic_fonts/` directory
- Strong Arabic fonts (Amiri, Noto Naskh Arabic, etc.) are recommended and can be used

No action needed.

---

### Suggestion 6 -- Clean Canny Inputs for Arabic

**Goal:** Provide the ControlNet with crisp, high-contrast canny edge maps from clean glyph templates.

**Status: DONE**

What is done:
- Glyph is rendered as white/colored text on pure black background (line 115 in `prepare_arabic_dataset.py`)
- Canny is applied only on the clean glyph image, not on any textured background
- Canny thresholds are low=50, high=100 -- producing clean edges

What is missing (from second suggestion):
- No "Extra Bold" or "Black" font weight variants are being used
- The second suggestion says thicker strokes create stronger gradients that the diffusion model "settles into" more reliably

What to do:
- Use bolder font variants (Extra Bold, Black weight) when available
- Optionally dilate the canny edges by 1-2 pixels to make them thicker
- Consider increasing canny thresholds slightly (e.g., low=60, high=120) to get only the strongest edges

---

### Suggestion 7 -- Arabic-Capable OCR Loss for Training

**Goal:** Replace the English-only OCR reward loss with one that can actually read Arabic, so the training pushes toward legible Arabic text.

**Status: NOT DONE**

What is done:
- `TextPerceptualLoss` class exists in `train_arabic.py` (lines 55-130)
- `train_config.yaml` has configuration for OCR model name and enable/disable flag

What is missing:
- OCR loss is **disabled** (`use_ocr_loss: false` in `train_config.yaml`)
- The configured OCR model is `microsoft/trocr-base-printed` -- this is English-only and cannot read Arabic
- Even if enabled, the training script does NOT use the full FLUX transformer. It only trains ControlNet with L2 regularization on outputs (lines 279-293), not actual diffusion denoising loss
- This means the model gets no meaningful "did the text render correctly?" feedback during training

What to do:
- Replace the OCR model with an Arabic-capable one:
  - PaddleOCR Arabic model
  - Arabic-capable TrOCR variant
  - Multi-script OCR that explicitly includes Arabic
- Enable OCR loss in the config (`use_ocr_loss: true`)
- Consider adding the full FLUX transformer to the training loop to compute proper denoising loss instead of L2 regularization alone

---

### Suggestion 8 -- Thick/Bold Canny Map ("Lead-Heavy Stencil")

**Goal:** Make the canny edges thick enough that the diffusion model treats them as "physical boundaries" rather than weak suggestions.

**Status: NOT DONE**

What is done:
- Standard canny edges are generated from the glyph
- Standard font weights are used

What is missing:
- No font weight selection logic (no preference for Bold/Extra Bold/Black)
- No dilation or thickening of canny edges after detection
- Thin canny lines are a weak signal that the model can override

What to do:
- Prefer "Extra Bold" or "Black" font variants when rendering glyphs
- After canny edge detection, apply morphological dilation (e.g., `cv2.dilate` with a 3x3 kernel) to thicken the edges
- Test the visual difference in generated images before and after thickening

---

### Suggestion 9 -- Texture-Aware Prompts

**Goal:** Prevent the "pasted-on" look by describing how text interacts with the surface in the prompt.

**Status: NOT DONE**

What is done:
- Prompts describe the scene generically (e.g., "a street sign in city")

What is missing:
- Prompts do not describe how the text integrates with the surface
- No use of words like "embossed," "carved," "engraved," "glowing through," "painted on" to describe text-surface interaction

What to do:
- When writing prompts for Arabic text, describe the physical integration:
  - "The word embossed in the metal sign"
  - "Neon Arabic letters glowing through foggy glass"
  - "Arabic calligraphy carved into stone"
- This forces the model to treat the text as part of the 3D scene rather than a flat overlay
- Note: Do NOT include the actual Arabic text in the prompt (per Suggestion 1), just describe the interaction

---

### Suggestion 10 -- Step-Decay Control (Late ControlNet Cutoff)

**Goal:** Keep ControlNet guidance active for longer so the model does not "improvise" at the end of generation.

**Status: PARTIALLY DONE**

What is done:
- `infer.py` runs ControlNet for the full 30/30 steps
- `infer_simple.py` runs ControlNet for 20/30 steps (67%)

What is missing:
- The suggestion says to run ControlNet for at least 80% of total steps
- `infer_simple.py` at 67% may be cutting off too early, allowing late-stage hallucination

What to do:
- For 30-step inference, set `controlnet_conditioning_step` to 24 or higher (80%+)
- Test the effect of full 30/30 guidance vs 24/30 vs 20/30

---

## Priority Action Plan

### Phase 1 -- Quick Inference Fixes (no retraining needed)

These changes can improve results immediately without any retraining:

1. Remove Arabic text from prompt in `infer.py` and `infer_inpaint.py`
2. Expose lambda2 as a parameter and experiment with values (0.0 to 0.9)
3. Increase `controlnet_conditioning_scale` from 1.0 to 1.5
4. Ensure `controlnet_conditioning_step` is at least 80% of total steps
5. Use texture-aware prompts that describe text-surface interaction
6. Use bolder fonts and thicken canny edges via dilation

### Phase 2 -- Masking Improvements

7. Implement per-line bbox splitting for multi-line text
8. Reduce mask padding to 3-5 px
9. Consider contour-based masks instead of rectangular bboxes

### Phase 3 -- Training Improvements (requires retraining)

10. Replace OCR model with Arabic-capable OCR (PaddleOCR Arabic, Arabic TrOCR)
11. Enable OCR loss in training config
12. Add full FLUX transformer to training loop for proper denoising loss
13. Retrain with bold font variants and thicker glyph templates

---

## File Reference

| File | Role | Key Changes Needed |
|------|------|-------------------|
| infer.py | Original inference | Remove Arabic from prompt |
| infer_simple.py | Arabic inference | Increase controlnet_scale, tune lambda2, adjust step cutoff |
| infer_inpaint.py | Inpainting inference | Remove Arabic from prompt |
| pipeline_flux_controlnet.py | Core pipeline | Expose lambda2 as parameter in prepare_latents_reptext() |
| train_arabic.py | Training | Replace OCR model, enable OCR loss, add full transformer |
| train_config.yaml | Config | Set use_ocr_loss to true, change OCR model name |
| prepare_arabic_dataset.py | Dataset prep | Use bold fonts, thicken canny, tighten masks |
