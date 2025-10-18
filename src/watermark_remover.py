"""
AI-based watermark removal using U-Net architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import os


class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture for watermark removal"""
    
    def __init__(self, n_channels: int = 3, n_classes: int = 3, 
                 features: list = [64, 128, 256, 512], bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3] * 2 // factor)
        
        # Decoder
        self.up1 = Up(features[3] * 2, features[3] // factor, bilinear)
        self.up2 = Up(features[3], features[2] // factor, bilinear)
        self.up3 = Up(features[2], features[1] // factor, bilinear)
        self.up4 = Up(features[1], features[0], bilinear)
        
        # Output
        self.outc = OutConv(features[0], n_classes)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return torch.sigmoid(logits)


class WatermarkRemover:
    """AI-based watermark removal using U-Net"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 device: Optional[torch.device] = None):
        """
        Initialize watermark remover
        
        Args:
            model_path: Path to pretrained model
            device: PyTorch device
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        self.model = UNet(n_channels=3, n_classes=3)
        self.model.to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("Warning: No pretrained model loaded. Using random initialization.")
            print("For better results, train the model or provide a pretrained checkpoint.")
    
    def load_model(self, model_path: str):
        """
        Load pretrained model
        
        Args:
            model_path: Path to model checkpoint
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"Model loaded from: {model_path}")
    
    def save_model(self, save_path: str, epoch: Optional[int] = None, 
                   optimizer_state: Optional[dict] = None):
        """
        Save model checkpoint
        
        Args:
            save_path: Path to save model
            epoch: Current epoch number
            optimizer_state: Optimizer state dict
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
        }
        
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, save_path)
        print(f"Model saved to: {save_path}")
    
    def remove_watermark(self, image: np.ndarray, 
                        mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Remove watermark from image
        
        Args:
            image: Input image (RGB, normalized [0, 1])
            mask: Optional binary mask of watermark region
            
        Returns:
            Image with watermark removed (RGB, [0, 1])
        """
        self.model.eval()
        
        # Convert to tensor
        if isinstance(image, np.ndarray):
            # HWC to CHW
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        else:
            image_tensor = image
        
        image_tensor = image_tensor.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(image_tensor)
        
        # Convert back to numpy
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # If mask is provided, blend only the masked region
        if mask is not None:
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=-1)
            mask = mask.astype(np.float32) / 255.0
            output = output * mask + image * (1 - mask)
        
        return output
    
    def remove_watermark_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Remove watermarks from batch of images
        
        Args:
            images: Batch of images (B, C, H, W)
            
        Returns:
            Batch of processed images
        """
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(images.to(self.device))
        
        return outputs


class InpaintingRemover:
    """Traditional inpainting-based watermark removal"""
    
    def __init__(self, method: str = 'telea', radius: int = 3):
        """
        Initialize inpainting remover
        
        Args:
            method: Inpainting method ('telea' or 'ns')
            radius: Inpainting radius
        """
        self.method = method
        self.radius = radius
    
    def remove_watermark(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Remove watermark using inpainting
        
        Args:
            image: Input image (RGB, 0-255 or 0-1)
            mask: Binary mask of watermark region (0-255)
            
        Returns:
            Image with watermark removed
        """
        import cv2
        
        # Ensure proper format
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Ensure mask is single channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        # Resize mask to match image size if needed
        if mask.shape[:2] != image_bgr.shape[:2]:
            mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
        
        # Inpaint
        if self.method == 'telea':
            inpainted = cv2.inpaint(image_bgr, mask, self.radius, cv2.INPAINT_TELEA)
        else:
            inpainted = cv2.inpaint(image_bgr, mask, self.radius, cv2.INPAINT_NS)
        
        # Convert back to RGB
        result = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
        
        return result


class HybridRemover:
    """Hybrid approach combining AI and inpainting"""
    
    def __init__(self, ai_remover: WatermarkRemover, 
                 inpainting_remover: InpaintingRemover):
        """
        Initialize hybrid remover
        
        Args:
            ai_remover: AI-based remover
            inpainting_remover: Inpainting-based remover
        """
        self.ai_remover = ai_remover
        self.inpainting_remover = inpainting_remover
    
    def remove_watermark(self, image: np.ndarray, mask: np.ndarray,
                        ai_weight: float = 0.7) -> np.ndarray:
        """
        Remove watermark using hybrid approach
        
        Args:
            image: Input image (normalized [0, 1])
            mask: Binary mask of watermark region
            ai_weight: Weight for AI result (0-1)
            
        Returns:
            Image with watermark removed
        """
        # AI-based removal
        ai_result = self.ai_remover.remove_watermark(image, mask)
        
        # Inpainting-based removal
        inpaint_result = self.inpainting_remover.remove_watermark(image, mask)
        
        # Normalize inpaint result to [0, 1]
        if inpaint_result.max() > 1.0:
            inpaint_result = inpaint_result.astype(np.float32) / 255.0
        
        # Blend results
        result = ai_weight * ai_result + (1 - ai_weight) * inpaint_result
        
        return result
