"""
Dataset class for watermark removal training
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, List
import random


class WatermarkDataset(Dataset):
    """Dataset for watermark removal training"""
    
    def __init__(self, 
                 clean_dir: str,
                 watermarked_dir: str,
                 image_size: Tuple[int, int] = (512, 512),
                 augment: bool = True):
        """
        Initialize dataset
        
        Args:
            clean_dir: Directory containing clean images
            watermarked_dir: Directory containing watermarked images
            image_size: Target image size (height, width)
            augment: Whether to apply data augmentation
        """
        self.clean_dir = Path(clean_dir)
        self.watermarked_dir = Path(watermarked_dir)
        self.image_size = image_size
        self.augment = augment
        
        # Get list of images
        self.clean_images = sorted(list(self.clean_dir.glob('*.png')) + 
                                   list(self.clean_dir.glob('*.jpg')))
        self.watermarked_images = sorted(list(self.watermarked_dir.glob('*.png')) + 
                                        list(self.watermarked_dir.glob('*.jpg')))
        
        # Ensure same number of images
        assert len(self.clean_images) == len(self.watermarked_images), \
            f"Number of clean ({len(self.clean_images)}) and watermarked " \
            f"({len(self.watermarked_images)}) images must match"
        
        print(f"Dataset loaded: {len(self.clean_images)} image pairs")
    
    def __len__(self) -> int:
        return len(self.clean_images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a data sample
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (watermarked_image, clean_image) as tensors
        """
        # Load images
        clean_img = cv2.imread(str(self.clean_images[idx]))
        watermarked_img = cv2.imread(str(self.watermarked_images[idx]))
        
        # Convert BGR to RGB
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        watermarked_img = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2RGB)
        
        # Resize
        clean_img = cv2.resize(clean_img, (self.image_size[1], self.image_size[0]))
        watermarked_img = cv2.resize(watermarked_img, (self.image_size[1], self.image_size[0]))
        
        # Data augmentation
        if self.augment:
            clean_img, watermarked_img = self._augment(clean_img, watermarked_img)
        
        # Normalize to [0, 1]
        clean_img = clean_img.astype(np.float32) / 255.0
        watermarked_img = watermarked_img.astype(np.float32) / 255.0
        
        # Convert to tensors (HWC -> CHW)
        clean_tensor = torch.from_numpy(clean_img.transpose(2, 0, 1))
        watermarked_tensor = torch.from_numpy(watermarked_img.transpose(2, 0, 1))
        
        return watermarked_tensor, clean_tensor
    
    def _augment(self, clean: np.ndarray, watermarked: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation
        
        Args:
            clean: Clean image
            watermarked: Watermarked image
            
        Returns:
            Augmented images
        """
        # Random horizontal flip
        if random.random() > 0.5:
            clean = cv2.flip(clean, 1)
            watermarked = cv2.flip(watermarked, 1)
        
        # Random vertical flip
        if random.random() > 0.5:
            clean = cv2.flip(clean, 0)
            watermarked = cv2.flip(watermarked, 0)
        
        # Random rotation (90, 180, 270 degrees)
        if random.random() > 0.5:
            k = random.randint(1, 3)
            clean = np.rot90(clean, k)
            watermarked = np.rot90(watermarked, k)
        
        return clean, watermarked


class SyntheticWatermarkDataset(Dataset):
    """Dataset that generates synthetic watermarks on-the-fly"""
    
    def __init__(self,
                 image_dir: str,
                 watermark_paths: List[str],
                 image_size: Tuple[int, int] = (512, 512),
                 augment: bool = True):
        """
        Initialize synthetic dataset
        
        Args:
            image_dir: Directory containing clean images
            watermark_paths: List of paths to watermark images
            image_size: Target image size
            augment: Whether to apply data augmentation
        """
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.augment = augment
        
        # Load watermarks
        self.watermarks = []
        for wm_path in watermark_paths:
            wm = cv2.imread(wm_path, cv2.IMREAD_UNCHANGED)
            if wm is not None:
                if wm.shape[2] == 3:
                    # Add alpha channel if not present
                    alpha = np.ones((wm.shape[0], wm.shape[1], 1), dtype=wm.dtype) * 255
                    wm = np.concatenate([wm, alpha], axis=2)
                self.watermarks.append(wm)
        
        # Get list of clean images
        self.images = sorted(list(self.image_dir.glob('*.png')) + 
                           list(self.image_dir.glob('*.jpg')))
        
        print(f"Synthetic dataset: {len(self.images)} images, {len(self.watermarks)} watermarks")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample with synthetic watermark"""
        # Load clean image
        clean_img = cv2.imread(str(self.images[idx]))
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        clean_img = cv2.resize(clean_img, (self.image_size[1], self.image_size[0]))
        
        # Apply synthetic watermark
        watermarked_img = self._apply_watermark(clean_img)
        
        # Data augmentation
        if self.augment:
            clean_img, watermarked_img = self._augment(clean_img, watermarked_img)
        
        # Normalize
        clean_img = clean_img.astype(np.float32) / 255.0
        watermarked_img = watermarked_img.astype(np.float32) / 255.0
        
        # Convert to tensors
        clean_tensor = torch.from_numpy(clean_img.transpose(2, 0, 1))
        watermarked_tensor = torch.from_numpy(watermarked_img.transpose(2, 0, 1))
        
        return watermarked_tensor, clean_tensor
    
    def _apply_watermark(self, image: np.ndarray) -> np.ndarray:
        """Apply random watermark to image"""
        # Select random watermark
        watermark = random.choice(self.watermarks).copy()
        
        # Random opacity
        opacity = random.uniform(0.3, 0.8)
        
        # Random scale
        scale = random.uniform(0.1, 0.3)
        wm_h = int(image.shape[0] * scale)
        wm_w = int(image.shape[1] * scale)
        watermark = cv2.resize(watermark, (wm_w, wm_h))
        
        # Random position
        max_y = image.shape[0] - wm_h
        max_x = image.shape[1] - wm_w
        y = random.randint(0, max_y) if max_y > 0 else 0
        x = random.randint(0, max_x) if max_x > 0 else 0
        
        # Apply watermark
        result = image.copy()
        for c in range(3):
            alpha = watermark[:, :, 3] / 255.0 * opacity
            result[y:y+wm_h, x:x+wm_w, c] = \
                (1 - alpha) * result[y:y+wm_h, x:x+wm_w, c] + \
                alpha * watermark[:, :, c]
        
        return result.astype(np.uint8)
    
    def _augment(self, clean: np.ndarray, watermarked: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation"""
        # Random horizontal flip
        if random.random() > 0.5:
            clean = cv2.flip(clean, 1)
            watermarked = cv2.flip(watermarked, 1)
        
        # Random vertical flip
        if random.random() > 0.5:
            clean = cv2.flip(clean, 0)
            watermarked = cv2.flip(watermarked, 0)
        
        return clean, watermarked


def get_dataloader(dataset: Dataset, 
                   batch_size: int = 4,
                   shuffle: bool = True,
                   num_workers: int = 4) -> DataLoader:
    """
    Create DataLoader
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
