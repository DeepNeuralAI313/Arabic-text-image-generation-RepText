"""
Arabic Dataset Loader for RepText Training
Loads preprocessed Arabic text training data for ControlNet training.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class ArabicRepTextDataset(Dataset):
    """
    Dataset for loading Arabic text training samples for RepText.
    
    Each sample contains:
    - glyph: Rendered Arabic text on black background
    - position: Position heatmap indicating text location
    - mask: Binary mask for text region
    - canny: Canny edge detection of glyph
    - metadata: Text content, font info, etc.
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (1024, 1024),
        split: str = 'train',
        train_ratio: float = 0.9,
        augment: bool = True,
        random_seed: int = 42
    ):
        """
        Args:
            data_dir: Directory containing prepared training samples
            image_size: Target image size (width, height)
            split: 'train' or 'val'
            train_ratio: Ratio of training samples
            augment: Whether to apply data augmentation
            random_seed: Random seed for train/val split
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.split = split
        self.augment = augment
        
        # Find all sample directories
        all_samples = sorted([
            d for d in self.data_dir.iterdir() 
            if d.is_dir() and d.name.startswith('sample_')
        ])
        
        if not all_samples:
            raise ValueError(f"No samples found in {data_dir}")
        
        # Split into train/val
        random.Random(random_seed).shuffle(all_samples)
        split_idx = int(len(all_samples) * train_ratio)
        
        if split == 'train':
            self.samples = all_samples[:split_idx]
        elif split == 'val':
            self.samples = all_samples[split_idx:]
        else:
            raise ValueError(f"Invalid split: {split}")
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        
        # Define transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5], [0.5])
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_image(self, path: Path) -> Image.Image:
        """Load image from path."""
        return Image.open(path).convert('RGB')
    
    def _load_mask(self, path: Path) -> Image.Image:
        """Load mask from path."""
        return Image.open(path).convert('L')
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single sample.
        
        Returns:
            Dictionary containing:
            - glyph: [3, H, W] tensor
            - position: [3, H, W] tensor
            - mask: [1, H, W] tensor
            - canny: [3, H, W] tensor
            - text: str (Arabic text)
        """
        sample_dir = self.samples[idx]
        
        # Load images
        glyph = self._load_image(sample_dir / 'glyph.png')
        position = self._load_image(sample_dir / 'position.png')
        canny = self._load_image(sample_dir / 'canny.png')
        mask = self._load_mask(sample_dir / 'mask.png')
        
        # Load metadata
        with open(sample_dir / 'metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Resize if needed
        if glyph.size != self.image_size:
            glyph = glyph.resize(self.image_size, Image.LANCZOS)
            position = position.resize(self.image_size, Image.LANCZOS)
            canny = canny.resize(self.image_size, Image.LANCZOS)
            mask = mask.resize(self.image_size, Image.LANCZOS)
        
        # Apply augmentation
        if self.augment and self.split == 'train':
            # Random horizontal flip (careful with Arabic RTL text)
            # Skip horizontal flip for text to maintain readability
            pass
        
        # Convert to tensors
        glyph_tensor = self.to_tensor(glyph)  # [3, H, W]
        position_tensor = self.to_tensor(position)  # [3, H, W]
        canny_tensor = self.to_tensor(canny)  # [3, H, W]
        mask_tensor = self.to_tensor(mask)  # [1, H, W]
        
        # Normalize to [-1, 1]
        glyph_tensor = glyph_tensor * 2.0 - 1.0
        position_tensor = position_tensor * 2.0 - 1.0
        canny_tensor = canny_tensor * 2.0 - 1.0
        # Mask stays in [0, 1]
        
        return {
            'glyph': glyph_tensor,
            'position': position_tensor,
            'mask': mask_tensor,
            'canny': canny_tensor,
            'text': metadata['text'],
            'font_size': metadata['font_size']
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching samples.
    """
    glyph = torch.stack([item['glyph'] for item in batch])
    position = torch.stack([item['position'] for item in batch])
    mask = torch.stack([item['mask'] for item in batch])
    canny = torch.stack([item['canny'] for item in batch])
    text = [item['text'] for item in batch]
    font_size = [item['font_size'] for item in batch]
    
    return {
        'glyph': glyph,
        'position': position,
        'mask': mask,
        'canny': canny,
        'text': text,
        'font_size': font_size
    }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (1024, 1024),
    train_ratio: float = 0.9
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = ArabicRepTextDataset(
        data_dir=data_dir,
        image_size=image_size,
        split='train',
        train_ratio=train_ratio,
        augment=True
    )
    
    val_dataset = ArabicRepTextDataset(
        data_dir=data_dir,
        image_size=image_size,
        split='val',
        train_ratio=train_ratio,
        augment=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Test dataset loading
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python arabic_dataset.py <data_dir>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=2,
        num_workers=0
    )
    
    # Test loading a batch
    print("\nTesting train loader...")
    for batch in train_loader:
        print(f"Glyph shape: {batch['glyph'].shape}")
        print(f"Position shape: {batch['position'].shape}")
        print(f"Mask shape: {batch['mask'].shape}")
        print(f"Canny shape: {batch['canny'].shape}")
        print(f"Text samples: {batch['text'][:2]}")
        break
    
    print("\nDataset loading test successful!")
