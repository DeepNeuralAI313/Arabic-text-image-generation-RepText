"""
RepText Training Script for Arabic Text Generation
Trains FLUX ControlNet model with text perceptual loss for accurate Arabic text rendering.
"""

import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FluxPipeline, DDPMScheduler, AutoencoderKL
from diffusers.optimization import get_scheduler
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm.auto import tqdm
import wandb

from controlnet_flux import FluxControlNetModel
from arabic_dataset import create_dataloaders


logger = get_logger(__name__)


class TextPerceptualLoss(nn.Module):
    """
    Text perceptual loss using OCR model to ensure text accuracy.
    Measures similarity between rendered text and target text.
    """
    
    def __init__(self, ocr_model_name: str = "microsoft/trocr-base-printed"):
        super().__init__()
        # You can use an Arabic-specific OCR model here
        # For example: "microsoft/trocr-base-handwritten" or Arabic models
        try:
            self.processor = AutoProcessor.from_pretrained(ocr_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(ocr_model_name)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        except Exception as e:
            logger.warning(f"Could not load OCR model: {e}. Text perceptual loss will be disabled.")
            self.processor = None
            self.model = None
    
    def forward(
        self, 
        generated_images: torch.Tensor, 
        target_texts: list
    ) -> torch.Tensor:
        """
        Compute text perceptual loss.
        
        Args:
            generated_images: [B, 3, H, W] in range [-1, 1]
            target_texts: List of target text strings
            
        Returns:
            Loss scalar
        """
        if self.processor is None or self.model is None:
            return torch.tensor(0.0, device=generated_images.device)
        
        # Convert images from [-1, 1] to [0, 255]
        images = ((generated_images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        
        # Process each image
        total_loss = 0.0
        valid_samples = 0
        
        for img, target_text in zip(images, target_texts):
            try:
                # Convert to PIL
                img_pil = transforms.ToPILImage()(img.cpu())
                
                # Process with OCR
                pixel_values = self.processor(img_pil, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(generated_images.device)
                
                # Encode target text
                target_ids = self.processor.tokenizer(
                    target_text,
                    return_tensors="pt",
                    padding=True
                ).input_ids.to(generated_images.device)
                
                # Compute loss
                outputs = self.model(pixel_values=pixel_values, labels=target_ids)
                total_loss += outputs.loss
                valid_samples += 1
                
            except Exception as e:
                logger.warning(f"Error in text perceptual loss computation: {e}")
                continue
        
        if valid_samples > 0:
            return total_loss / valid_samples
        else:
            return torch.tensor(0.0, device=generated_images.device)


def train_one_epoch(
    accelerator: Accelerator,
    controlnet: FluxControlNetModel,
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    text_perceptual_loss: Optional[TextPerceptualLoss],
    config: Dict,
    epoch: int,
    global_step: int
) -> int:
    """Train for one epoch."""
    
    controlnet.train()
    
    progress_bar = tqdm(
        total=len(train_loader),
        disable=not accelerator.is_local_main_process,
        desc=f"Epoch {epoch}"
    )
    
    for step, batch in enumerate(train_loader):
        with accelerator.accumulate(controlnet):
            # Get inputs
            glyph = batch['glyph']  # [B, 3, H, W]
            position = batch['position']  # [B, 3, H, W]
            canny = batch['canny']  # [B, 3, H, W]
            mask = batch['mask']  # [B, 1, H, W]
            text = batch['text']  # List of strings
            
            # Encode glyph to latent space
            with torch.no_grad():
                latents = vae.encode(glyph).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
            
            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample timesteps
            timesteps = torch.randint(
                0, 
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device
            )
            timesteps = timesteps.long()
            
            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Prepare controlnet conditioning
            # Concatenate canny and position as control signals
            controlnet_cond = torch.cat([canny, position], dim=1)  # [B, 6, H, W]
            
            # Encode condition to latent space
            with torch.no_grad():
                controlnet_cond_latent = vae.encode(
                    controlnet_cond[:, :3]  # Use first 3 channels
                ).latent_dist.sample()
                controlnet_cond_latent = controlnet_cond_latent * vae.config.scaling_factor
            
            # Get ControlNet prediction
            down_block_res_samples, mid_block_res_sample = controlnet(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=None,  # We don't use text encoder for glyph
controlnet_cond=controlnet_cond_latent,
                return_dict=False
            )
            
            # For now, compute diffusion loss on the controlnet output
            # In a full implementation, you would pass these to the main FLUX model
            # and compute the loss on the final prediction
            
            # Simplified diffusion loss (MSE between noise and prediction)
            # Note: This is a simplified version. Full implementation would use FLUX transformer
            target = noise
            
            # Use mid_block output as prediction (simplified)
            if mid_block_res_sample is not None:
                # Resize to match target if needed
                if mid_block_res_sample.shape != target.shape:
                    mid_block_res_sample = F.interpolate(
                        mid_block_res_sample,
                        size=target.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Diffusion loss
                diffusion_loss = F.mse_loss(mid_block_res_sample, target)
            else:
                diffusion_loss = torch.tensor(0.0, device=latents.device)
            
            # Text perceptual loss (on decoded images)
            if text_perceptual_loss is not None and config['training']['text_perceptual_loss_weight'] > 0:
                with torch.no_grad():
                    # Decode latents to images
                    pred_latents = noisy_latents - mid_block_res_sample if mid_block_res_sample is not None else noisy_latents
                    pred_images = vae.decode(pred_latents / vae.config.scaling_factor).sample
                
                perceptual_loss = text_perceptual_loss(pred_images, text)
            else:
                perceptual_loss = torch.tensor(0.0, device=latents.device)
            
            # Combined loss
            loss = (
                config['training']['diffusion_loss_weight'] * diffusion_loss +
                config['training']['text_perceptual_loss_weight'] * perceptual_loss
            )
            
            # Backward
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                params_to_clip = controlnet.parameters()
                accelerator.clip_grad_norm_(params_to_clip, config['training']['max_grad_norm'])
            
            optimizer.step()
            optimizer.zero_grad()
        
        # Logging
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            
            if global_step % config['training']['logging_steps'] == 0:
                logs = {
                    "loss": loss.detach().item(),
                    "diffusion_loss": diffusion_loss.detach().item(),
                    "perceptual_loss": perceptual_loss.detach().item() if isinstance(perceptual_loss, torch.Tensor) else perceptual_loss,
                    "lr": optimizer.param_groups[0]['lr'],
                    "epoch": epoch
                }
                progress_bar.set_postfix(**logs)
                
                if accelerator.is_main_process:
                    logger.info(f"Step {global_step}: {logs}")
                    if wandb.run is not None:
                        wandb.log(logs, step=global_step)
            
            # Save checkpoint
            if global_step % config['training']['save_steps'] == 0:
                if accelerator.is_main_process:
                    save_path = os.path.join(
                        config['output']['output_dir'],
                        f"checkpoint-{global_step}"
                    )
                    os.makedirs(save_path, exist_ok=True)
                    
                    unwrapped_controlnet = accelerator.unwrap_model(controlnet)
                    unwrapped_controlnet.save_pretrained(save_path)
                    
                    logger.info(f"Saved checkpoint to {save_path}")
    
    progress_bar.close()
    return global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='train_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=config['output']['output_dir'],
        logging_dir=config['output']['logging_dir']
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        mixed_precision=config['training']['mixed_precision'],
        log_with="wandb" if args.use_wandb else None,
        project_config=accelerator_project_config
    )
    
    # Initialize wandb
    if accelerator.is_main_process and args.use_wandb:
        wandb.init(
            project="reptext-arabic",
            config=config,
            name=f"arabic_reptext_{config['output']['output_dir'].split('/')[-1]}"
        )
    
    # Set seed
    set_seed(42)
    
    # Create output directory
    os.makedirs(config['output']['output_dir'], exist_ok=True)
    
    logger.info(f"Loading base model: {config['model']['base_model']}")
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        config['model']['base_model'],
        subfolder="vae",
        torch_dtype=torch.bfloat16
    )
    vae.requires_grad_(False)
    
    # Initialize ControlNet
    logger.info("Initializing ControlNet...")
    controlnet = FluxControlNetModel.from_pretrained(
        config['model']['base_model'],
        subfolder="transformer",
        **config['model']['controlnet_config']
    ) if args.resume_from_checkpoint is None else FluxControlNetModel.from_pretrained(
        args.resume_from_checkpoint
    )
    
    # Initialize noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        config['model']['base_model'],
        subfolder="scheduler"
    )
    
    # Initialize text perceptual loss
    text_perceptual_loss = None
    if config['ocr']['use_ocr_loss']:
        logger.info(f"Loading OCR model: {config['ocr']['model_name']}")
        text_perceptual_loss = TextPerceptualLoss(config['ocr']['model_name'])
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_size=tuple(config['data']['image_size']),
        train_ratio=config['data']['train_ratio']
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=config['training']['learning_rate'],
        betas=(config['training']['adam_beta1'], config['training']['adam_beta2']),
        weight_decay=config['training']['adam_weight_decay'],
        eps=config['training']['adam_epsilon']
    )
    
    # Initialize LR scheduler
    lr_scheduler = get_scheduler(
        config['training']['lr_scheduler'],
        optimizer=optimizer,
        num_warmup_steps=config['training']['lr_warmup_steps'],
        num_training_steps=config['training']['num_epochs'] * len(train_loader)
    )
    
    # Prepare with accelerator
    controlnet, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_loader, val_loader, lr_scheduler
    )
    
    vae.to(accelerator.device)
    if text_perceptual_loss is not None:
        text_perceptual_loss.to(accelerator.device)
    
    # Training loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_loader.dataset)}")
    logger.info(f"  Num Epochs = {config['training']['num_epochs']}")
    logger.info(f"  Batch size = {config['data']['batch_size']}")
    logger.info(f"  Gradient Accumulation steps = {config['training']['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {config['training']['num_epochs'] * len(train_loader)}")
    
    global_step = 0
    
    for epoch in range(config['training']['num_epochs']):
        global_step = train_one_epoch(
            accelerator=accelerator,
            controlnet=controlnet,
            vae=vae,
            noise_scheduler=noise_scheduler,
            optimizer=optimizer,
            train_loader=train_loader,
            text_perceptual_loss=text_perceptual_loss,
            config=config,
            epoch=epoch,
            global_step=global_step
        )
        
        # Step LR scheduler
        lr_scheduler.step()
    
    # Save final model
    if accelerator.is_main_process:
        save_path = os.path.join(config['output']['output_dir'], "final_model")
        os.makedirs(save_path, exist_ok=True)
        
        unwrapped_controlnet = accelerator.unwrap_model(controlnet)
        unwrapped_controlnet.save_pretrained(save_path)
        
        logger.info(f"Training complete! Model saved to {save_path}")
    
    if args.use_wandb and accelerator.is_main_process:
        wandb.finish()


if __name__ == '__main__':
    main()
