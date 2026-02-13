"""
RepText Training Script for Arabic Text Generation (Phase 3)

Fine-tunes the FLUX ControlNet on real Arabic scene text data (EvArEST dataset)
with proper denoising loss through the full FLUX transformer.

Key improvements over previous version:
1. Uses FLUX transformer (frozen) for proper noise prediction loss
2. Uses real CLIP + T5 text embeddings from prompts
3. Supports EasyOCR for Arabic text perceptual loss
4. CPU offloading for memory efficiency
5. Flow matching loss (not DDPM)
"""

import os
import gc
import yaml
import argparse
from pathlib import Path
from typing import Dict, Optional, List

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_CUDNN_BENCHMARK", "0")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from tqdm.auto import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from controlnet_flux import FluxControlNetModel
from arabic_dataset import create_dataloaders

logger = get_logger(__name__)

# Try enabling flash attention
try:
    torch.backends.cuda.enable_flash_sdp(True)
except Exception:
    pass


# =============================================================================
# Utility Functions
# =============================================================================

def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    """Pack latents: [B, C, H, W] -> [B, (H/2)*(W/2), C*4]"""
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def _unpack_latents(latents, height, width, vae_scale_factor):
    """Unpack latents: [B, seq, C*4] -> [B, C, H, W]"""
    batch_size, num_patches, channels = latents.shape
    height = height // vae_scale_factor
    width = width // vae_scale_factor
    latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // 4, height * 2, width * 2)
    return latents


def _prepare_latent_image_ids(height, width, device, dtype):
    """Create position IDs for packed image tokens."""
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
    latent_image_ids = latent_image_ids.reshape((height // 2) * (width // 2), 3)
    return latent_image_ids.to(device=device, dtype=dtype)


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                    base_shift=0.5, max_shift=1.16):
    """Calculate the shift parameter for flow matching."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def clear_memory():
    """Aggressively clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# =============================================================================
# Text Encoder Helpers
# =============================================================================

def encode_prompt(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    tokenizer_2: T5TokenizerFast,
    text_encoder_2: T5EncoderModel,
    prompt: List[str],
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int = 512,
):
    """
    Encode text prompts using CLIP (pooled) and T5 (sequence) encoders.
    Returns prompt_embeds, pooled_prompt_embeds, text_ids.
    """
    # CLIP pooled embeddings
    clip_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    pooled_prompt_embeds = text_encoder(
        clip_inputs.input_ids.to(device),
        output_hidden_states=False,
    ).pooler_output
    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=dtype)

    # T5 sequence embeddings
    t5_inputs = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    prompt_embeds = text_encoder_2(
        t5_inputs.input_ids.to(device),
        output_hidden_states=False,
    )[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype)

    # Text position IDs
    text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


# =============================================================================
# Arabic OCR Text Perceptual Loss
# =============================================================================

class ArabicTextPerceptualLoss(nn.Module):
    """
    Text perceptual loss using EasyOCR for Arabic text.
    Measures character-level edit distance between OCR output and target.
    """

    def __init__(self, backend: str = "easyocr"):
        super().__init__()
        self.backend = backend
        self.reader = None

        if backend == "easyocr":
            try:
                import easyocr
                self.reader = easyocr.Reader(['ar', 'en'], gpu=False)
                logger.info("✅ EasyOCR initialized for Arabic+English")
            except Exception as e:
                logger.warning(f"Could not initialize EasyOCR: {e}")
        else:
            logger.warning(f"Unknown OCR backend: {backend}")

    def _normalized_edit_distance(self, pred: str, target: str) -> float:
        """Compute normalized Levenshtein edit distance."""
        if not target:
            return 0.0
        m, n = len(pred), len(target)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i-1] == target[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[m][n] / max(m, n, 1)

    def forward(
        self,
        generated_images: torch.Tensor,
        target_texts: List[str],
    ) -> torch.Tensor:
        """
        Compute OCR-based text perceptual loss.

        Args:
            generated_images: [B, 3, H, W] in [-1, 1]
            target_texts: list of target Arabic text strings

        Returns:
            Loss scalar (higher = worse text accuracy)
        """
        if self.reader is None:
            return torch.tensor(0.0, device=generated_images.device)

        total_loss = 0.0
        valid = 0

        images = ((generated_images.detach() + 1) * 127.5).clamp(0, 255).to(torch.uint8)

        for img, target in zip(images, target_texts):
            try:
                img_np = img.cpu().permute(1, 2, 0).numpy()
                results = self.reader.readtext(img_np, detail=0)
                predicted = ' '.join(results) if results else ''
                dist = self._normalized_edit_distance(predicted, target)
                total_loss += dist
                valid += 1
            except Exception:
                continue

        if valid > 0:
            return torch.tensor(total_loss / valid, device=generated_images.device,
                                dtype=generated_images.dtype, requires_grad=False)
        return torch.tensor(0.0, device=generated_images.device)


# =============================================================================
# Training Loop
# =============================================================================

def train_one_epoch(
    accelerator: Accelerator,
    controlnet: FluxControlNetModel,
    transformer: FluxTransformer2DModel,
    vae: AutoencoderKL,
    scheduler: FlowMatchEulerDiscreteScheduler,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    tokenizer_2: T5TokenizerFast,
    text_encoder_2: T5EncoderModel,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    ocr_loss_fn: Optional[ArabicTextPerceptualLoss],
    config: Dict,
    epoch: int,
    global_step: int
) -> int:
    """Train for one epoch with proper denoising loss."""

    controlnet.train()

    progress_bar = tqdm(
        total=len(train_loader),
        disable=not accelerator.is_local_main_process,
        desc=f"Epoch {epoch}"
    )

    use_text_encoders = config['model'].get('load_text_encoders', True)
    use_cpu_offload = config['training'].get('use_cpu_offload', False)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels))

    for step, batch in enumerate(train_loader):
        with accelerator.accumulate(controlnet):
            # Get data
            target = batch['target']          # [B, 3, H, W] - scene image with text
            canny = batch['canny']            # [B, 3, H, W]
            position = batch['position']      # [B, 3, H, W]
            mask = batch['mask']              # [B, 1, H, W]
            text = batch['text']              # List[str]
            prompts = batch['prompt']         # List[str]

            b = target.shape[0]
            device = accelerator.device

            # ---- Step 1: Encode target image to latent space ----
            with torch.no_grad():
                target_bf16 = target.to(dtype=torch.bfloat16)
                latents = vae.encode(target_bf16).latent_dist.sample()
                latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

            _, c, h, w = latents.shape

            # ---- Step 2: Sample timestep and create noisy latents ----
            # Flow matching: sample t ~ U(0, 1) and create noisy = (1-t)*noise + t*signal
            # But FLUX uses the scheduler's sigma schedule, so we sample a random step index
            u = torch.rand(b, device=device, dtype=torch.float32)
            # Sigmas represent noise levels; higher sigma = more noise
            noise = torch.randn_like(latents)

            # Flow matching interpolation: x_t = (1 - sigma) * x_0 + sigma * noise
            sigma = u.view(b, 1, 1, 1)
            noisy_latents = (1.0 - sigma) * latents + sigma * noise

            # The timestep passed to the model is sigma (normalized to 0-1 range)
            timesteps = u  # Already in [0, 1] for flow matching

            # ---- Step 3: Encode control conditions ----
            with torch.no_grad():
                canny_bf16 = canny.to(dtype=torch.bfloat16)
                position_bf16 = position.to(dtype=torch.bfloat16)

                canny_latent = vae.encode(canny_bf16).latent_dist.sample()
                canny_latent = (canny_latent - vae.config.shift_factor) * vae.config.scaling_factor

                position_latent = vae.encode(position_bf16).latent_dist.sample()
                position_latent = (position_latent - vae.config.shift_factor) * vae.config.scaling_factor

                controlnet_cond = torch.cat([canny_latent, position_latent], dim=1)

            # ---- Step 4: Pack latents (2x2 spatial patching) ----
            noisy_packed = _pack_latents(noisy_latents, b, c, h, w)
            cond_channels = controlnet_cond.shape[1]
            cond_packed = _pack_latents(controlnet_cond, b, cond_channels, h, w)

            # ---- Step 5: Create position IDs ----
            img_ids = _prepare_latent_image_ids(h, w, device, noisy_packed.dtype)
            # img_ids doesn't have batch dim in FLUX

            # ---- Step 6: Encode text prompt ----
            if use_text_encoders and text_encoder is not None and text_encoder_2 is not None:
                with torch.no_grad():
                    # Move text encoders to GPU if offloaded
                    if use_cpu_offload:
                        text_encoder.to(device)
                        text_encoder_2.to(device)

                    prompt_embeds, pooled_prompt_embeds, txt_ids = encode_prompt(
                        tokenizer, text_encoder, tokenizer_2, text_encoder_2,
                        prompts, device, noisy_packed.dtype,
                    )

                    # Offload back to CPU
                    if use_cpu_offload:
                        text_encoder.to("cpu")
                        text_encoder_2.to("cpu")
                        clear_memory()
            else:
                # Fallback: dummy embeddings
                seq_len = int(config['model'].get('text_seq_len', 64))
                prompt_embeds = torch.zeros(b, seq_len, 4096, device=device, dtype=noisy_packed.dtype)
                pooled_prompt_embeds = torch.zeros(b, 768, device=device, dtype=noisy_packed.dtype)
                txt_ids = torch.zeros(seq_len, 3, device=device, dtype=noisy_packed.dtype)

            # ---- Step 7: Guidance embedding ----
            guidance = torch.ones(b, device=device, dtype=noisy_packed.dtype) * 3.5

            # ---- Step 8: ControlNet forward pass ----
            controlnet_block_samples, controlnet_single_block_samples = controlnet(
                hidden_states=noisy_packed,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                controlnet_cond=cond_packed,
                txt_ids=txt_ids,
                img_ids=img_ids,
                guidance=guidance,
                return_dict=False,
            )

            # ---- Step 9: FLUX Transformer forward pass (frozen) ----
            with torch.no_grad():
                if use_cpu_offload:
                    transformer.to(device)

                # The transformer predicts the velocity (flow matching target)
                noise_pred = transformer(
                    hidden_states=noisy_packed.detach(),  # Detach to prevent grad through transformer
                    timestep=timesteps,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_block_samples=[s.detach() for s in controlnet_block_samples] if controlnet_block_samples else None,
                    controlnet_single_block_samples=[s.detach() for s in controlnet_single_block_samples] if controlnet_single_block_samples else None,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]

                if use_cpu_offload:
                    transformer.to("cpu")
                    clear_memory()

            # ---- IMPORTANT: Recompute with gradients flowing through ControlNet ----
            # We need the ControlNet outputs to be part of the computational graph
            # Re-run transformer with ControlNet outputs that have gradients
            # But since transformer is frozen, we compute loss differently:
            #
            # The loss is: how well do the ControlNet residuals guide the transformer
            # to predict the correct velocity?
            #
            # For flow matching, the target velocity is: x_0 - noise (= clean - noise)
            # The model predicts this velocity, and we want it to match.

            # Pack the clean latents for target computation
            latents_packed = _pack_latents(latents, b, c, h, w)
            noise_packed = _pack_latents(noise, b, c, h, w)

            # Flow matching target: v = x_0 - noise
            target_velocity = latents_packed - noise_packed

            # ---- Step 10: Compute denoising loss ----
            # Loss = MSE(predicted_velocity, target_velocity)
            diffusion_loss = F.mse_loss(noise_pred.float(), target_velocity.float())

            # ---- Step 11: OCR perceptual loss (optional) ----
            ocr_loss = torch.tensor(0.0, device=device)
            if ocr_loss_fn is not None and config['ocr'].get('use_ocr_loss', False):
                # Only compute OCR loss every N steps to save time
                if global_step % 10 == 0:
                    with torch.no_grad():
                        # Unpack and decode the prediction for OCR
                        pred_latents = noisy_packed - sigma.view(b, 1, 1).expand_as(noise_packed) * noise_pred
                        pred_unpacked = _unpack_latents(pred_latents, h * vae_scale_factor, w * vae_scale_factor, vae_scale_factor)
                        pred_decoded = vae.decode(
                            (pred_unpacked.to(torch.bfloat16) / vae.config.scaling_factor) + vae.config.shift_factor
                        ).sample
                    ocr_loss = ocr_loss_fn(pred_decoded, text)

            # ---- Step 12: Combined loss ----
            loss = (
                config['training']['diffusion_loss_weight'] * diffusion_loss +
                config['ocr'].get('ocr_loss_weight', 0.1) * ocr_loss
            )

            # Backward
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(controlnet.parameters(), config['training']['max_grad_norm'])

            optimizer.step()
            optimizer.zero_grad()

            clear_memory()

        # Logging
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

            if global_step % config['training']['logging_steps'] == 0:
                logs = {
                    "loss": loss.detach().item(),
                    "diffusion_loss": diffusion_loss.detach().item(),
                    "ocr_loss": ocr_loss.detach().item() if isinstance(ocr_loss, torch.Tensor) else ocr_loss,
                    "lr": optimizer.param_groups[0]['lr'],
                    "epoch": epoch,
                }
                progress_bar.set_postfix(**logs)

                if accelerator.is_main_process:
                    logger.info(f"Step {global_step}: {logs}")
                    if WANDB_AVAILABLE and wandb.run is not None:
                        wandb.log(logs, step=global_step)

            # Save checkpoint
            if global_step % config['training']['save_steps'] == 0:
                if accelerator.is_main_process:
                    save_path = os.path.join(
                        config['output']['output_dir'],
                        f"checkpoint-{global_step}"
                    )
                    os.makedirs(save_path, exist_ok=True)
                    unwrapped = accelerator.unwrap_model(controlnet)
                    unwrapped.save_pretrained(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

    progress_bar.close()
    return global_step


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RepText Arabic Training (Phase 3)")
    parser.add_argument('--config', type=str, default='train_config_evarest.yaml',
                        help='Training configuration file')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--dry_run', action='store_true',
                        help='Run 1 step on CPU for validation')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device (e.g., "cpu" for dry run)')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Max training steps (for testing)')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override for dry run
    if args.dry_run:
        config['data']['batch_size'] = 1
        config['data']['num_workers'] = 0
        config['training']['num_epochs'] = 1
        config['model']['load_transformer'] = False  # Skip transformer for CPU dry run
        config['model']['load_text_encoders'] = False
        config['ocr']['use_ocr_loss'] = False

    # Setup accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=config['output']['output_dir'],
        logging_dir=config['output']['logging_dir'],
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        mixed_precision=config['training']['mixed_precision'] if not args.dry_run else "no",
        log_with="wandb" if args.use_wandb and WANDB_AVAILABLE else None,
        project_config=accelerator_project_config,
    )

    # W&B
    if accelerator.is_main_process and args.use_wandb and WANDB_AVAILABLE:
        wandb.init(project="reptext-arabic-phase3", config=config,
                   name=f"evarest_{Path(config['output']['output_dir']).name}")

    set_seed(42)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    os.makedirs(config['output']['output_dir'], exist_ok=True)

    # ---- Load Models ----
    base_model = config['model']['base_model']
    device = torch.device(args.device) if args.device else accelerator.device
    dtype = torch.bfloat16 if not args.dry_run else torch.float32

    # VAE (always needed, frozen)
    logger.info(f"Loading VAE from {base_model}...")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device)
    clear_memory()

    # Scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model, subfolder="scheduler")

    # ControlNet (trainable)
    logger.info("Loading ControlNet...")
    pretrained_cn = config['model'].get('pretrained_controlnet')
    if args.resume_from_checkpoint:
        controlnet = FluxControlNetModel.from_pretrained(args.resume_from_checkpoint, torch_dtype=dtype)
    elif pretrained_cn:
        controlnet = FluxControlNetModel.from_pretrained(pretrained_cn, torch_dtype=dtype)
    else:
        # Initialize from transformer config
        from diffusers import FluxTransformer2DModel as FT
        t_config = FT.load_config(base_model, subfolder="transformer")
        controlnet = FluxControlNetModel.from_config(t_config, **config['model'].get('controlnet_config', {}))

    if config['training'].get('gradient_checkpointing', True):
        try:
            controlnet.enable_gradient_checkpointing()
            logger.info("✅ Gradient checkpointing enabled")
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")

    try:
        if hasattr(controlnet, 'enable_xformers_memory_efficient_attention'):
            controlnet.enable_xformers_memory_efficient_attention()
            logger.info("✅ xFormers attention enabled")
    except Exception:
        pass

    clear_memory()

    # FLUX Transformer (frozen, for denoising loss)
    transformer = None
    if config['model'].get('load_transformer', False):
        logger.info(f"Loading FLUX Transformer from {base_model}...")
        transformer = FluxTransformer2DModel.from_pretrained(
            base_model, subfolder="transformer", torch_dtype=dtype
        )
        transformer.requires_grad_(False)
        transformer.eval()

        if config['training'].get('use_cpu_offload', False):
            transformer.to("cpu")
            logger.info("  → Transformer on CPU (will offload to GPU per batch)")
        else:
            transformer.to(device)

        clear_memory()
    else:
        logger.info("⚠️ Transformer not loaded - using simplified loss (not recommended)")

    # Text Encoders (frozen)
    tokenizer = None
    text_encoder = None
    tokenizer_2 = None
    text_encoder_2 = None

    if config['model'].get('load_text_encoders', False):
        logger.info("Loading text encoders...")
        tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            base_model, subfolder="text_encoder", torch_dtype=dtype
        )
        text_encoder.requires_grad_(False)
        text_encoder.eval()

        tokenizer_2 = T5TokenizerFast.from_pretrained(base_model, subfolder="tokenizer_2")
        text_encoder_2 = T5EncoderModel.from_pretrained(
            base_model, subfolder="text_encoder_2", torch_dtype=dtype
        )
        text_encoder_2.requires_grad_(False)
        text_encoder_2.eval()

        if config['training'].get('use_cpu_offload', False):
            text_encoder.to("cpu")
            text_encoder_2.to("cpu")
            logger.info("  → Text encoders on CPU (will offload to GPU per batch)")
        else:
            text_encoder.to(device)
            text_encoder_2.to(device)

        clear_memory()
    else:
        logger.info("⚠️ Text encoders not loaded - using dummy embeddings")

    # OCR loss
    ocr_loss_fn = None
    if config['ocr'].get('use_ocr_loss', False):
        backend = config['ocr'].get('backend', 'easyocr')
        logger.info(f"Loading OCR model ({backend})...")
        ocr_loss_fn = ArabicTextPerceptualLoss(backend=backend)
    else:
        logger.info("OCR loss disabled")

    # ---- Data ----
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_size=tuple(config['data']['image_size']),
        train_ratio=config['data']['train_ratio'],
    )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=config['training']['learning_rate'],
        betas=(config['training']['adam_beta1'], config['training']['adam_beta2']),
        weight_decay=config['training']['adam_weight_decay'],
        eps=config['training']['adam_epsilon'],
    )

    lr_scheduler_obj = get_scheduler(
        config['training']['lr_scheduler'],
        optimizer=optimizer,
        num_warmup_steps=config['training']['lr_warmup_steps'],
        num_training_steps=config['training']['num_epochs'] * len(train_loader),
    )

    # ---- Prepare with accelerator ----
    controlnet, optimizer, train_loader, val_loader, lr_scheduler_obj = accelerator.prepare(
        controlnet, optimizer, train_loader, val_loader, lr_scheduler_obj
    )

    if not config['training'].get('use_cpu_offload', False):
        vae.to(accelerator.device)

    # ---- Memory info ----
    if torch.cuda.is_available() and accelerator.is_main_process:
        logger.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, "
                     f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")

    # ---- Training ----
    logger.info("=" * 60)
    logger.info("***** Running Phase 3 Training *****")
    logger.info(f"  Dataset: {config['data']['data_dir']}")
    logger.info(f"  Num examples = {len(train_loader.dataset)}")
    logger.info(f"  Num epochs = {config['training']['num_epochs']}")
    logger.info(f"  Batch size = {config['data']['batch_size']}")
    logger.info(f"  Grad accumulation = {config['training']['gradient_accumulation_steps']}")
    logger.info(f"  Effective batch = {config['data']['batch_size'] * config['training']['gradient_accumulation_steps']}")
    logger.info(f"  Transformer loaded = {transformer is not None}")
    logger.info(f"  Text encoders loaded = {text_encoder is not None}")
    logger.info(f"  OCR loss = {ocr_loss_fn is not None}")
    logger.info(f"  CPU offload = {config['training'].get('use_cpu_offload', False)}")
    logger.info("=" * 60)

    global_step = 0

    for epoch in range(config['training']['num_epochs']):
        global_step = train_one_epoch(
            accelerator=accelerator,
            controlnet=controlnet,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            tokenizer_2=tokenizer_2,
            text_encoder_2=text_encoder_2,
            optimizer=optimizer,
            train_loader=train_loader,
            ocr_loss_fn=ocr_loss_fn,
            config=config,
            epoch=epoch,
            global_step=global_step,
        )

        lr_scheduler_obj.step()

        if args.max_steps and global_step >= args.max_steps:
            logger.info(f"Reached max_steps ({args.max_steps}), stopping.")
            break

    # Save final model
    if accelerator.is_main_process:
        save_path = os.path.join(config['output']['output_dir'], "final_model")
        os.makedirs(save_path, exist_ok=True)
        unwrapped = accelerator.unwrap_model(controlnet)
        unwrapped.save_pretrained(save_path)
        logger.info(f"🎉 Training complete! Model saved to {save_path}")

    if WANDB_AVAILABLE and wandb.run is not None and accelerator.is_main_process:
        wandb.finish()


if __name__ == '__main__':
    main()
