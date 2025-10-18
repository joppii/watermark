"""
Data preparation script for watermark removal training
"""
import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil


def create_synthetic_watermarks(input_dir: str, 
                                output_clean_dir: str,
                                output_watermarked_dir: str,
                                watermark_paths: list,
                                num_variations: int = 3):
    """
    Create synthetic watermarked dataset from clean images
    
    Args:
        input_dir: Directory with clean images
        output_clean_dir: Output directory for clean images
        output_watermarked_dir: Output directory for watermarked images
        watermark_paths: Paths to watermark images
        num_variations: Number of variations per image
    """
    os.makedirs(output_clean_dir, exist_ok=True)
    os.makedirs(output_watermarked_dir, exist_ok=True)
    
    # Load watermarks
    watermarks = []
    for wm_path in watermark_paths:
        wm = cv2.imread(wm_path, cv2.IMREAD_UNCHANGED)
        if wm is not None:
            if len(wm.shape) == 2:
                # Grayscale to RGBA
                wm = cv2.cvtColor(wm, cv2.COLOR_GRAY2BGR)
                alpha = np.ones((wm.shape[0], wm.shape[1], 1), dtype=wm.dtype) * 255
                wm = np.concatenate([wm, alpha], axis=2)
            elif wm.shape[2] == 3:
                # Add alpha channel
                alpha = np.ones((wm.shape[0], wm.shape[1], 1), dtype=wm.dtype) * 255
                wm = np.concatenate([wm, alpha], axis=2)
            watermarks.append(wm)
    
    print(f"Loaded {len(watermarks)} watermarks")
    
    # Get list of images
    input_path = Path(input_dir)
    images = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
    
    print(f"Found {len(images)} images")
    print(f"Creating {num_variations} variations per image...")
    
    count = 0
    for img_path in tqdm(images, desc="Processing images"):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        for var_idx in range(num_variations):
            # Select random watermark
            watermark = watermarks[np.random.randint(0, len(watermarks))].copy()
            
            # Random parameters
            opacity = np.random.uniform(0.3, 0.8)
            scale = np.random.uniform(0.1, 0.3)
            
            # Resize watermark
            wm_h = int(img.shape[0] * scale)
            wm_w = int(img.shape[1] * scale)
            watermark = cv2.resize(watermark, (wm_w, wm_h))
            
            # Random position
            max_y = img.shape[0] - wm_h
            max_x = img.shape[1] - wm_w
            y = np.random.randint(0, max_y) if max_y > 0 else 0
            x = np.random.randint(0, max_x) if max_x > 0 else 0
            
            # Apply watermark
            watermarked = img.copy()
            for c in range(3):
                alpha = watermark[:, :, 3] / 255.0 * opacity
                watermarked[y:y+wm_h, x:x+wm_w, c] = \
                    (1 - alpha) * watermarked[y:y+wm_h, x:x+wm_w, c] + \
                    alpha * watermark[:, :, c]
            
            # Save images
            filename = f"{img_path.stem}_{var_idx:03d}.png"
            cv2.imwrite(os.path.join(output_clean_dir, filename), img)
            cv2.imwrite(os.path.join(output_watermarked_dir, filename), watermarked)
            count += 1
    
    print(f"\n✓ Created {count} image pairs")
    print(f"Clean images: {output_clean_dir}")
    print(f"Watermarked images: {output_watermarked_dir}")


def create_text_watermarks(text: str, 
                          output_path: str,
                          num_variations: int = 10):
    """
    Create text watermark variations
    
    Args:
        text: Watermark text
        output_path: Output directory
        num_variations: Number of variations to create
    """
    os.makedirs(output_path, exist_ok=True)
    
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_PLAIN,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
    ]
    
    colors = [
        (255, 255, 255),  # White
        (200, 200, 200),  # Light gray
        (0, 0, 0),        # Black
        (50, 50, 50),     # Dark gray
    ]
    
    print(f"Creating {num_variations} text watermark variations...")
    
    for i in range(num_variations):
        # Random parameters
        font = fonts[np.random.randint(0, len(fonts))]
        color = colors[np.random.randint(0, len(colors))]
        scale = np.random.uniform(1.0, 3.0)
        thickness = np.random.randint(1, 4)
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, scale, thickness
        )
        
        # Create image with padding
        padding = 20
        img_size = (text_height + baseline + padding * 2, 
                   text_width + padding * 2)
        
        # Create transparent image
        img = np.zeros((img_size[0], img_size[1], 4), dtype=np.uint8)
        
        # Draw text
        cv2.putText(
            img, text,
            (padding, text_height + padding),
            font, scale, color + (255,), thickness
        )
        
        # Save
        output_file = os.path.join(output_path, f"text_watermark_{i:03d}.png")
        cv2.imwrite(output_file, img)
    
    print(f"✓ Created {num_variations} text watermarks in {output_path}")


def split_dataset(input_dir: str, 
                 train_dir: str,
                 val_dir: str,
                 val_split: float = 0.1):
    """
    Split dataset into train and validation sets
    
    Args:
        input_dir: Input directory
        train_dir: Training output directory
        val_dir: Validation output directory
        val_split: Validation split ratio
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all images
    input_path = Path(input_dir)
    images = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
    
    # Shuffle
    np.random.shuffle(images)
    
    # Split
    split_idx = int(len(images) * (1 - val_split))
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Copy files
    print(f"Splitting {len(images)} images...")
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")
    
    for img_path in tqdm(train_images, desc="Copying train"):
        shutil.copy(str(img_path), train_dir)
    
    for img_path in tqdm(val_images, desc="Copying val"):
        shutil.copy(str(img_path), val_dir)
    
    print(f"✓ Dataset split complete")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Prepare training data for watermark removal"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Synthetic watermark generation
    synthetic_parser = subparsers.add_parser('synthetic', 
                                            help='Create synthetic watermarks')
    synthetic_parser.add_argument('-i', '--input', type=str, required=True,
                                 help='Input directory with clean images')
    synthetic_parser.add_argument('--output-clean', type=str, 
                                 default='data/train/clean',
                                 help='Output directory for clean images')
    synthetic_parser.add_argument('--output-watermarked', type=str,
                                 default='data/train/watermarked',
                                 help='Output directory for watermarked images')
    synthetic_parser.add_argument('-w', '--watermarks', type=str, nargs='+',
                                 required=True,
                                 help='Paths to watermark images')
    synthetic_parser.add_argument('-n', '--num-variations', type=int, default=3,
                                 help='Number of variations per image')
    
    # Text watermark creation
    text_parser = subparsers.add_parser('text', help='Create text watermarks')
    text_parser.add_argument('-t', '--text', type=str, required=True,
                            help='Watermark text')
    text_parser.add_argument('-o', '--output', type=str, 
                            default='data/watermarks',
                            help='Output directory')
    text_parser.add_argument('-n', '--num-variations', type=int, default=10,
                            help='Number of variations')
    
    # Dataset split
    split_parser = subparsers.add_parser('split', help='Split dataset')
    split_parser.add_argument('-i', '--input', type=str, required=True,
                             help='Input directory')
    split_parser.add_argument('--train-dir', type=str, default='data/train',
                             help='Training output directory')
    split_parser.add_argument('--val-dir', type=str, default='data/val',
                             help='Validation output directory')
    split_parser.add_argument('--val-split', type=float, default=0.1,
                             help='Validation split ratio')
    
    args = parser.parse_args()
    
    if args.command == 'synthetic':
        create_synthetic_watermarks(
            args.input,
            args.output_clean,
            args.output_watermarked,
            args.watermarks,
            args.num_variations
        )
    elif args.command == 'text':
        create_text_watermarks(
            args.text,
            args.output,
            args.num_variations
        )
    elif args.command == 'split':
        split_dataset(
            args.input,
            args.train_dir,
            args.val_dir,
            args.val_split
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
