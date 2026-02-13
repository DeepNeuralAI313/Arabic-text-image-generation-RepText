"""
Check Training Status and Test Model
Run this to see if your training is complete and test the model.
"""

import os
import glob
import torch

def check_training_status():
    """Check if training has started and/or completed."""
    output_dir = "./output/arabic_reptext"
    
    print("=" * 60)
    print("RepText Arabic Training Status Check")
    print("=" * 60)
    print()
    
    # Check if training has started
    if not os.path.exists(output_dir):
        print("❌ Training hasn't started yet!")
        print()
        print("To start training, run:")
        print("  cd RepText")
        print("  accelerate launch train_arabic.py --config train_config_48gb.yaml --use_wandb")
        return None
    
    print("✓ Training has started!")
    print()
    
    # Check for checkpoints
    checkpoints = sorted(glob.glob(f"{output_dir}/checkpoint-*"))
    if checkpoints:
        print(f"Found {len(checkpoints)} checkpoint(s):")
        for cp in checkpoints[-5:]:  # Show last 5
            step = os.path.basename(cp).split('-')[-1]
            print(f"  - Checkpoint at step {step}")
        print()
        latest_checkpoint = checkpoints[-1]
    else:
        print("No checkpoints found yet. Training just started.")
        print()
        latest_checkpoint = None
    
    # Check if training completed
    final_model_path = f"{output_dir}/final_model"
    if os.path.exists(final_model_path):
        print("🎉 TRAINING COMPLETED!")
        print(f"Final model saved at: {final_model_path}")
        print()
        return final_model_path
    elif latest_checkpoint:
        print("⏳ Training still in progress...")
        print(f"Latest checkpoint: {latest_checkpoint}")
        print()
        print("You can:")
        print("  1. Wait for training to complete (1-2 days)")
        print("  2. Test with latest checkpoint (partial training)")
        print("  3. Check W&B dashboard: https://wandb.ai/")
        print()
        return latest_checkpoint
    else:
        print("⏳ Training just started, no checkpoints yet.")
        print("Check back in a few hours.")
        return None


def test_model(model_path):
    """Load and test the trained model."""
    if model_path is None:
        print("Cannot test model - no checkpoint available yet.")
        return
    
    print("=" * 60)
    print("Loading Model for Testing")
    print("=" * 60)
    print()
    
    # Check if model files exist
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        print(f"❌ ERROR: config.json not found in {model_path}")
        print()
        print("This checkpoint may be incomplete or corrupted.")
        print("Try using a different checkpoint or wait for training to continue.")
        return
    
    print(f"Loading model from: {model_path}")
    print("This may take a few minutes...")
    print()
    
    try:
        from controlnet_flux import FluxControlNetModel
        from pipeline_flux_controlnet import FluxControlNetPipeline
        
        # Load base model
        base_model = "black-forest-labs/FLUX.1-dev"
        
        # Load your trained ControlNet
        print("Loading ControlNet...")
        controlnet = FluxControlNetModel.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )
        print("✓ ControlNet loaded")
        
        # Load pipeline
        print("Loading FLUX pipeline...")
        pipe = FluxControlNetPipeline.from_pretrained(
            base_model, 
            controlnet=controlnet, 
            torch_dtype=torch.bfloat16
        ).to("cuda")
        print("✓ Pipeline loaded")
        
        print()
        print("🎉 Model loaded successfully!")
        print()
        print("You're ready to generate Arabic text images!")
        print("See infer.py for complete inference examples.")
        
        return pipe
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print()
        print("Common issues:")
        print("  1. Not authenticated with HuggingFace (run: huggingface-cli login)")
        print("  2. FLUX.1-dev access not granted (visit: https://huggingface.co/black-forest-labs/FLUX.1-dev)")
        print("  3. Checkpoint incomplete (training still saving)")
        print()
        raise


if __name__ == "__main__":
    # Check training status
    model_path = check_training_status()
    
    # Ask if user wants to test
    if model_path:
        print()
        response = input("Do you want to load and test this model? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            test_model(model_path)
        else:
            print("Skipping model testing.")
            print(f"To test later, run: python -c \"from check_training import test_model; test_model('{model_path}')\"")
    else:
        print()
        print("No model available to test yet. Check back later!")
