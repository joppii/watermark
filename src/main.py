"""
Main entry point for watermark removal
"""
import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_config, load_image, save_image, get_device,
    create_comparison_image, print_device_info
)
from detector import WatermarkDetector
from image_processor import ImageProcessor
from watermark_remover import WatermarkRemover, InpaintingRemover, HybridRemover


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AI-based Watermark Removal Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with AI model
  python src/main.py -i sample/image.png -o output/result.png
  
  # Use inpainting method
  python src/main.py -i sample/image.png -o output/result.png --method inpainting
  
  # Use hybrid method
  python src/main.py -i sample/image.png -o output/result.png --method hybrid
  
  # Specify custom config
  python src/main.py -i sample/image.png -o output/result.png -c config/custom.yaml
  
  # Auto-detect watermark
  python src/main.py -i sample/image.png -o output/result.png --auto-detect
        """
    )
    
    parser.add_argument('-i', '--input', type=str, required=True,
                       help='Input image path')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='Output image path')
    parser.add_argument('-m', '--mask', type=str, default=None,
                       help='Mask image path (optional, auto-detect if not provided)')
    parser.add_argument('-c', '--config', type=str, default='config/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--method', type=str, default='ai',
                       choices=['ai', 'inpainting', 'hybrid'],
                       help='Removal method (ai, inpainting, hybrid)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to pretrained model (for AI method)')
    parser.add_argument('--auto-detect', action='store_true',
                       help='Automatically detect watermark')
    parser.add_argument('--detection-method', type=str, default='adaptive_threshold',
                       choices=['adaptive_threshold', 'edge_detection', 'color_based'],
                       help='Watermark detection method')
    parser.add_argument('--save-comparison', action='store_true',
                       help='Save comparison image')
    parser.add_argument('--save-mask', action='store_true',
                       help='Save detected mask')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Print device info if verbose
    if args.verbose:
        print_device_info()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        if args.verbose:
            print(f"Configuration loaded from: {args.config}")
    else:
        print(f"Warning: Config file not found: {args.config}")
        config = {}
    
    # Get device
    device_pref = args.device if args.device != 'auto' else config.get('device', 'auto')
    device = get_device(device_pref)
    print(f"Using device: {device}")
    
    # Load input image
    print(f"Loading image: {args.input}")
    image = load_image(args.input)
    print(f"Image shape: {image.shape}")
    
    # Initialize image processor
    target_size = tuple(config.get('image', {}).get('input_size', [512, 512]))
    processor = ImageProcessor(target_size=target_size)
    
    # Preprocess image
    processed_image, metadata = processor.preprocess(image)
    
    # Detect or load mask
    mask = None
    if args.auto_detect or args.mask is None:
        print("Detecting watermark...")
        detector_config = config.get('detection', {})
        detector = WatermarkDetector(method=args.detection_method)
        
        mask, regions = detector.detect(
            image,
            threshold=detector_config.get('threshold', 0.5),
            min_area=detector_config.get('min_area', 100),
            max_area=detector_config.get('max_area', 50000)
        )
        
        print(f"Detected {len(regions)} watermark region(s)")
        
        if args.save_mask:
            mask_path = args.output.replace('.', '_mask.')
            save_image(mask, mask_path)
            print(f"Mask saved to: {mask_path}")
        
        if args.verbose and len(regions) > 0:
            vis = detector.visualize_detection(image, mask, regions)
            vis_path = args.output.replace('.', '_detection.')
            save_image(vis, vis_path)
            print(f"Detection visualization saved to: {vis_path}")
    
    elif args.mask:
        print(f"Loading mask: {args.mask}")
        mask = load_image(args.mask)
        # Convert to grayscale if needed
        if len(mask.shape) == 3:
            import cv2
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    
    # Remove watermark
    print(f"Removing watermark using {args.method} method...")
    
    if args.method == 'ai':
        # AI-based removal
        model_path = args.model or config.get('model', {}).get('pretrained_path')
        remover = WatermarkRemover(model_path=model_path, device=device)
        result = remover.remove_watermark(processed_image, mask)
        
    elif args.method == 'inpainting':
        # Inpainting-based removal
        removal_config = config.get('removal', {})
        remover = InpaintingRemover(
            method=removal_config.get('algorithm', 'telea'),
            radius=removal_config.get('inpainting_radius', 3)
        )
        result = remover.remove_watermark(processed_image, mask)
        
        # Normalize to [0, 1]
        if result.max() > 1.0:
            result = result.astype('float32') / 255.0
    
    elif args.method == 'hybrid':
        # Hybrid method
        model_path = args.model or config.get('model', {}).get('pretrained_path')
        ai_remover = WatermarkRemover(model_path=model_path, device=device)
        
        removal_config = config.get('removal', {})
        inpaint_remover = InpaintingRemover(
            method=removal_config.get('algorithm', 'telea'),
            radius=removal_config.get('inpainting_radius', 3)
        )
        
        hybrid_remover = HybridRemover(ai_remover, inpaint_remover)
        result = hybrid_remover.remove_watermark(processed_image, mask, ai_weight=0.7)
    
    # Postprocess
    result = processor.postprocess(result, metadata)
    
    # Optional: Enhance quality
    if config.get('image', {}).get('enhance_quality', False):
        result = processor.enhance_quality(result)
    
    # Save result
    output_config = config.get('output', {})
    quality = output_config.get('quality', 95)
    save_image(result, args.output, quality=quality)
    
    # Save comparison if requested
    if args.save_comparison or output_config.get('save_comparison', False):
        comparison = create_comparison_image(image, result)
        comparison_path = args.output.replace('.', '_comparison.')
        save_image(comparison, comparison_path, quality=quality)
        print(f"Comparison saved to: {comparison_path}")
    
    print("\n✓ Watermark removal completed successfully!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
