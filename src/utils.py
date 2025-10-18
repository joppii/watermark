"""
Utility functions for watermark removal project
"""
import os
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Union, Tuple, Optional
import yaml
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device(device_preference: str = "auto") -> torch.device:
    """
    Get the appropriate device for PyTorch
    
    Args:
        device_preference: Device preference (auto, cpu, cuda, mps)
        
    Returns:
        PyTorch device
    """
    if device_preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_preference)


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array (RGB format)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load with OpenCV (BGR format)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image(image: np.ndarray, output_path: str, quality: int = 95) -> None:
    """
    Save an image to file
    
    Args:
        image: Image as numpy array (RGB format)
        output_path: Path to save the image
        quality: Image quality (for JPEG)
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Save image
    if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
        cv2.imwrite(output_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif output_path.lower().endswith('.png'):
        cv2.imwrite(output_path, image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    else:
        cv2.imwrite(output_path, image_bgr)
    
    print(f"Image saved to: {output_path}")


def normalize_image(image: np.ndarray, mean: list = None, std: list = None) -> np.ndarray:
    """
    Normalize image for neural network input
    
    Args:
        image: Input image (0-255 range)
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        
    Returns:
        Normalized image
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    # Convert to float and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Apply mean and std normalization
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    image = (image - mean) / std
    
    return image


def denormalize_image(image: np.ndarray, mean: list = None, std: list = None) -> np.ndarray:
    """
    Denormalize image from neural network output
    
    Args:
        image: Normalized image
        mean: Mean values used for normalization
        std: Standard deviation values used for normalization
        
    Returns:
        Denormalized image (0-255 range)
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    # Denormalize
    image = image * std + mean
    
    # Convert to [0, 255] range
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    
    return image


def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                 keep_aspect_ratio: bool = False) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Resize image to target size
    
    Args:
        image: Input image
        target_size: Target size (height, width)
        keep_aspect_ratio: Whether to keep aspect ratio
        
    Returns:
        Resized image and original size
    """
    original_size = image.shape[:2]
    
    if keep_aspect_ratio:
        # Calculate aspect ratio
        h, w = original_size
        target_h, target_w = target_size
        
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        
        resized = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        resized = cv2.resize(image, (target_size[1], target_size[0]), 
                            interpolation=cv2.INTER_LINEAR)
    
    return resized, original_size


def create_comparison_image(original: np.ndarray, processed: np.ndarray) -> np.ndarray:
    """
    Create side-by-side comparison image
    
    Args:
        original: Original image
        processed: Processed image
        
    Returns:
        Comparison image
    """
    # Ensure images have the same size
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
    
    # Create comparison
    comparison = np.hstack([original, processed])
    
    # Add labels
    h, w = original.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Processed", (w + 10, 30), font, 1, (255, 255, 255), 2)
    
    return comparison


def numpy_to_torch(image: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor
    
    Args:
        image: Numpy array (H, W, C)
        
    Returns:
        PyTorch tensor (1, C, H, W)
    """
    # Convert to CHW format
    image = np.transpose(image, (2, 0, 1))
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    # Convert to tensor
    tensor = torch.from_numpy(image).float()
    return tensor


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array
    
    Args:
        tensor: PyTorch tensor (1, C, H, W) or (C, H, W)
        
    Returns:
        Numpy array (H, W, C)
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Convert to numpy
    image = tensor.cpu().detach().numpy()
    
    # Convert to HWC format
    image = np.transpose(image, (1, 2, 0))
    
    return image


def print_device_info():
    """Print information about available devices"""
    print("=" * 50)
    print("Device Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CPU cores: {os.cpu_count()}")
    print("=" * 50)
