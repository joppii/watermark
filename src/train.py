"""
Training script for watermark removal model
"""
import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from watermark_remover import UNet
from dataset import WatermarkDataset, SyntheticWatermarkDataset, get_dataloader
from utils import get_device


class WatermarkRemovalTrainer:
    """Trainer for watermark removal model"""
    
    def __init__(self, config: dict):
        """
        Initialize trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = get_device(config.get('device', 'auto'))
        
        # Create model
        model_config = config['model']
        self.model = UNet(
            n_channels=model_config.get('input_channels', 3),
            n_classes=model_config.get('output_channels', 3),
            features=model_config.get('features', [64, 128, 256, 512])
        ).to(self.device)
        
        # Loss function
        self.criterion = nn.L1Loss()  # L1 loss for better edge preservation
        self.mse_criterion = nn.MSELoss()  # Additional MSE loss
        
        # Optimizer
        training_config = config['training']
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=training_config.get('learning_rate', 0.0001)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir='runs/watermark_removal')
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, (watermarked, clean) in enumerate(pbar):
            # Move to device
            watermarked = watermarked.to(self.device)
            clean = clean.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(watermarked)
            
            # Calculate loss
            l1_loss = self.criterion(output, clean)
            mse_loss = self.mse_criterion(output, clean)
            loss = l1_loss + 0.1 * mse_loss  # Weighted combination
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update stats
            total_loss += loss.item()
            self.global_step += 1
            
            # Log to tensorboard
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Loss/train_batch', loss.item(), self.global_step)
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for watermarked, clean in tqdm(val_loader, desc="Validating"):
                watermarked = watermarked.to(self.device)
                clean = clean.to(self.device)
                
                output = self.model(watermarked)
                
                l1_loss = self.criterion(output, clean)
                mse_loss = self.mse_criterion(output, clean)
                loss = l1_loss + 0.1 * mse_loss
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def save_checkpoint(self, save_path: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")
        
        if is_best:
            best_path = save_path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch}")
    
    def train(self, train_loader, val_loader, num_epochs: int):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
        """
        print("=" * 60)
        print(f"Starting training on {self.device}")
        print(f"Total epochs: {num_epochs}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print("=" * 60)
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            self.writer.add_scalar('Loss/val_epoch', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', 
                                  self.optimizer.param_groups[0]['lr'], epoch)
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            save_interval = self.config['training'].get('save_interval', 10)
            if (epoch + 1) % save_interval == 0 or is_best:
                save_path = f"models/pretrained/checkpoint_epoch_{epoch}.pth"
                self.save_checkpoint(save_path, is_best)
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation loss: {self.best_loss:.4f}")
        print("=" * 60)
        
        self.writer.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Watermark Removal Model")
    
    parser.add_argument('-c', '--config', type=str, default='config/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--clean-dir', type=str, required=True,
                       help='Directory containing clean images')
    parser.add_argument('--watermarked-dir', type=str, default=None,
                       help='Directory containing watermarked images (optional for synthetic)')
    parser.add_argument('--watermarks', type=str, nargs='+', default=None,
                       help='Paths to watermark images for synthetic dataset')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic watermark generation')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    # Create datasets
    if args.synthetic:
        print("Using synthetic watermark dataset...")
        if not args.watermarks:
            raise ValueError("--watermarks required for synthetic dataset")
        
        # Split data
        from torch.utils.data import random_split
        full_dataset = SyntheticWatermarkDataset(
            args.clean_dir,
            args.watermarks,
            image_size=tuple(config['image']['input_size']),
            augment=True
        )
        
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
    else:
        print("Using paired watermark dataset...")
        if not args.watermarked_dir:
            raise ValueError("--watermarked-dir required for paired dataset")
        
        # Create datasets
        from torch.utils.data import random_split
        full_dataset = WatermarkDataset(
            args.clean_dir,
            args.watermarked_dir,
            image_size=tuple(config['image']['input_size']),
            augment=True
        )
        
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = get_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = get_dataloader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Create trainer
    trainer = WatermarkRemovalTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(train_loader, val_loader, config['training']['epochs'])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
