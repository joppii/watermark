"""
Unit tests for watermark removal
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from image_processor import ImageProcessor
from detector import WatermarkDetector


class TestImageProcessor:
    """Test ImageProcessor class"""
    
    def test_preprocess(self):
        """Test image preprocessing"""
        processor = ImageProcessor(target_size=(256, 256))
        
        # Create dummy image
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Preprocess
        processed, metadata = processor.preprocess(image)
        
        # Check output
        assert processed.shape == (256, 256, 3)
        assert processed.dtype == np.float32
        assert processed.min() >= 0 and processed.max() <= 1
        assert metadata['original_size'] == (512, 512)
    
    def test_postprocess(self):
        """Test image postprocessing"""
        processor = ImageProcessor(target_size=(256, 256))
        
        # Create dummy processed image
        processed = np.random.rand(256, 256, 3).astype(np.float32)
        metadata = {'original_size': (512, 512)}
        
        # Postprocess
        result = processor.postprocess(processed, metadata)
        
        # Check output
        assert result.shape == (512, 512, 3)
        assert result.dtype == np.uint8


class TestWatermarkDetector:
    """Test WatermarkDetector class"""
    
    def test_detect_adaptive_threshold(self):
        """Test adaptive threshold detection"""
        detector = WatermarkDetector(method='adaptive_threshold')
        
        # Create dummy image with white rectangle (watermark)
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        image[100:200, 100:300] = 255
        
        # Detect
        mask, regions = detector.detect(image)
        
        # Check output
        assert mask.shape == (512, 512)
        assert len(regions) >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
