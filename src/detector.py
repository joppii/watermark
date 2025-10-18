"""
Watermark detection module
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional


class WatermarkDetector:
    """Detect watermarks in images using various methods"""
    
    def __init__(self, method: str = "adaptive_threshold"):
        """
        Initialize watermark detector
        
        Args:
            method: Detection method (adaptive_threshold, edge_detection, ai)
        """
        self.method = method
        
    def detect(self, image: np.ndarray, **kwargs) -> Tuple[np.ndarray, List[dict]]:
        """
        Detect watermarks in image
        
        Args:
            image: Input image (RGB format)
            **kwargs: Additional parameters for detection method
            
        Returns:
            Binary mask and list of detected regions
        """
        if self.method == "adaptive_threshold":
            return self._detect_adaptive_threshold(image, **kwargs)
        elif self.method == "edge_detection":
            return self._detect_edge_based(image, **kwargs)
        elif self.method == "color_based":
            return self._detect_color_based(image, **kwargs)
        else:
            raise ValueError(f"Unknown detection method: {self.method}")
    
    def _detect_adaptive_threshold(self, image: np.ndarray, 
                                   threshold: float = 0.5,
                                   min_area: int = 100,
                                   max_area: int = 50000) -> Tuple[np.ndarray, List[dict]]:
        """
        Detect watermarks using adaptive thresholding
        
        Args:
            image: Input image
            threshold: Threshold value
            min_area: Minimum area for watermark region
            max_area: Maximum area for watermark region
            
        Returns:
            Binary mask and detected regions
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        mask = np.zeros_like(gray)
        regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Draw on mask
                cv2.drawContours(mask, [contour], -1, 255, -1)
                
                regions.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'contour': contour
                })
        
        return mask, regions
    
    def _detect_edge_based(self, image: np.ndarray,
                          low_threshold: int = 50,
                          high_threshold: int = 150,
                          min_area: int = 100,
                          max_area: int = 50000) -> Tuple[np.ndarray, List[dict]]:
        """
        Detect watermarks using edge detection
        
        Args:
            image: Input image
            low_threshold: Low threshold for Canny edge detection
            high_threshold: High threshold for Canny edge detection
            min_area: Minimum area for watermark region
            max_area: Maximum area for watermark region
            
        Returns:
            Binary mask and detected regions
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        # Dilate edges to connect nearby components
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours
        mask = np.zeros_like(gray)
        regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                
                regions.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'contour': contour
                })
        
        return mask, regions
    
    def _detect_color_based(self, image: np.ndarray,
                           target_color: Optional[np.ndarray] = None,
                           tolerance: int = 30,
                           min_area: int = 100,
                           max_area: int = 50000) -> Tuple[np.ndarray, List[dict]]:
        """
        Detect watermarks based on color
        
        Args:
            image: Input image
            target_color: Target color in BGR format [B, G, R]
            tolerance: Color tolerance
            min_area: Minimum area for watermark region
            max_area: Maximum area for watermark region
            
        Returns:
            Binary mask and detected regions
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        if target_color is None:
            # Detect bright/white watermarks (common case)
            lower = np.array([0, 0, 200])
            upper = np.array([180, 30, 255])
        else:
            # Convert target color to HSV
            target_hsv = cv2.cvtColor(
                np.uint8([[target_color]]), cv2.COLOR_BGR2HSV
            )[0][0]
            
            lower = np.array([
                max(0, target_hsv[0] - tolerance),
                max(0, target_hsv[1] - tolerance),
                max(0, target_hsv[2] - tolerance)
            ])
            upper = np.array([
                min(180, target_hsv[0] + tolerance),
                min(255, target_hsv[1] + tolerance),
                min(255, target_hsv[2] + tolerance)
            ])
        
        # Create mask
        mask = cv2.inRange(hsv, lower, upper)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours
        filtered_mask = np.zeros_like(mask)
        regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
                
                regions.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'contour': contour
                })
        
        return filtered_mask, regions
    
    def detect_transparent_watermark(self, image: np.ndarray, 
                                     sensitivity: float = 1.5) -> np.ndarray:
        """
        Detect semi-transparent watermarks using statistical methods
        
        Args:
            image: Input image
            sensitivity: Detection sensitivity
            
        Returns:
            Binary mask of detected watermark
        """
        # Convert to LAB color space for better analysis
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Calculate local statistics
        mean = cv2.blur(l_channel.astype(float), (15, 15))
        mean_sq = cv2.blur((l_channel.astype(float) ** 2), (15, 15))
        std = np.sqrt(mean_sq - mean ** 2)
        
        # Detect anomalies
        threshold = np.mean(std) + sensitivity * np.std(std)
        mask = (std > threshold).astype(np.uint8) * 255
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def visualize_detection(self, image: np.ndarray, mask: np.ndarray, 
                           regions: List[dict]) -> np.ndarray:
        """
        Visualize detected watermark regions
        
        Args:
            image: Original image
            mask: Detection mask
            regions: List of detected regions
            
        Returns:
            Visualization image
        """
        # Create copy for visualization
        vis = image.copy()
        
        # Overlay mask in red
        mask_colored = np.zeros_like(image)
        mask_colored[:, :, 0] = mask  # Red channel
        vis = cv2.addWeighted(vis, 0.7, mask_colored, 0.3, 0)
        
        # Draw bounding boxes
        for region in regions:
            x, y, w, h = region['bbox']
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add area text
            area_text = f"{region['area']:.0f}"
            cv2.putText(vis, area_text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return vis
