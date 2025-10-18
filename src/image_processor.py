"""
Image preprocessing and postprocessing module
"""
import cv2
import numpy as np
from typing import Tuple, Optional


class ImageProcessor:
    """Image preprocessing and postprocessing for watermark removal"""
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        """
        Initialize image processor
        
        Args:
            target_size: Target size for processing (height, width)
        """
        self.target_size = target_size
        
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Preprocess image for model input
        
        Args:
            image: Input image (RGB format)
            
        Returns:
            Preprocessed image and metadata
        """
        # Store original information
        metadata = {
            'original_size': image.shape[:2],
            'original_dtype': image.dtype
        }
        
        # Resize to target size
        resized = cv2.resize(image, (self.target_size[1], self.target_size[0]), 
                            interpolation=cv2.INTER_LINEAR)
        
        # Convert to float32 and normalize to [0, 1]
        processed = resized.astype(np.float32) / 255.0
        
        return processed, metadata
    
    def postprocess(self, image: np.ndarray, metadata: dict) -> np.ndarray:
        """
        Postprocess model output
        
        Args:
            image: Model output image (normalized)
            metadata: Metadata from preprocessing
            
        Returns:
            Final output image
        """
        # Denormalize from [0, 1] to [0, 255]
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        
        # Resize back to original size
        original_size = metadata['original_size']
        restored = cv2.resize(image, (original_size[1], original_size[0]), 
                             interpolation=cv2.INTER_LINEAR)
        
        return restored
    
    def enhance_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality after watermark removal
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Apply bilateral filter to reduce noise while preserving edges
        enhanced = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Slight sharpening
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]]) / 9.0
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Blend original and sharpened
        result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        
        return result
    
    def denoise(self, image: np.ndarray, strength: int = 10) -> np.ndarray:
        """
        Denoise image using Non-local Means Denoising
        
        Args:
            image: Input image
            strength: Denoising strength
            
        Returns:
            Denoised image
        """
        denoised = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        return denoised
    
    def adjust_brightness_contrast(self, image: np.ndarray, 
                                   brightness: float = 0, 
                                   contrast: float = 1.0) -> np.ndarray:
        """
        Adjust brightness and contrast
        
        Args:
            image: Input image
            brightness: Brightness adjustment (-100 to 100)
            contrast: Contrast adjustment (0.5 to 3.0)
            
        Returns:
            Adjusted image
        """
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return adjusted
    
    def remove_artifacts(self, image: np.ndarray) -> np.ndarray:
        """
        Remove small artifacts from image
        
        Args:
            image: Input image
            
        Returns:
            Cleaned image
        """
        # Convert to grayscale for morphological operations
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply morphological closing to remove small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Create mask for artifacts
        _, mask = cv2.threshold(closed, 250, 255, cv2.THRESH_BINARY)
        
        # Inpaint artifacts
        result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        return result
    
    def blend_edges(self, original: np.ndarray, processed: np.ndarray, 
                    mask: Optional[np.ndarray] = None, feather: int = 5) -> np.ndarray:
        """
        Blend edges between original and processed image
        
        Args:
            original: Original image
            processed: Processed image
            mask: Binary mask of processed region
            feather: Feather radius for smooth blending
            
        Returns:
            Blended image
        """
        if mask is None:
            # No mask provided, return processed image
            return processed
        
        # Ensure mask is single channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to mask for smooth transition
        mask_float = mask.astype(np.float32) / 255.0
        mask_blurred = cv2.GaussianBlur(mask_float, (feather * 2 + 1, feather * 2 + 1), 0)
        
        # Expand mask to 3 channels
        mask_3ch = np.stack([mask_blurred] * 3, axis=-1)
        
        # Blend images
        blended = (processed * mask_3ch + original * (1 - mask_3ch)).astype(np.uint8)
        
        return blended
