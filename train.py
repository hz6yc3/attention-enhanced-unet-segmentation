"""
Training Script for U-Net Road Segmentation
=============================================

This script trains U-Net models (baseline and attention variants)
for satellite road segmentation.

Usage:
    python train.py --model baseline --epochs 100
    python train.py --model attention --epochs 100

Reference Paper:
    "A Comprehensive Review of U-Net and Its Variants"
    IET Image Processing, 2025
    https://arxiv.org/abs/2502.06895
"""

import os
import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_config, Config
from models import get_model
from utils.dataset import get_dataloaders
from utils.losses import get_loss_function
from utils.metrics import MetricTracker, dice_coefficient, iou_score


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    device: str,
    scaler=None,
    use_amp: bool = True
) -> dict:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scaler: GradScaler for mixed precision
        use_amp: Whether to use automatic mixed precision
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    metric_tracker = MetricTracker()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        
        with torch.no_grad():
            pred_probs = torch.sigmoid(outputs)
            metric_tracker.update(pred_probs, masks)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{metric_tracker.get_metrics()["dice"]:.4f}'
        })
    
    metrics = metric_tracker.get_metrics()
    metrics['loss'] = running_loss / len(dataloader)
    
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    criterion,
    device: str
) -> dict:
    """
    Validate the model.
    
    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to run on
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    metric_tracker = MetricTracker()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Validation", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        running_loss += loss.item()
        
        pred_probs = torch.sigmoid(outputs)
        metric_tracker.update(pred_probs, masks)
    
    metrics = metric_tracker.get_metrics()
    metrics['loss'] = running_loss / len(dataloader)
    
    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_dice: float,
    checkpoint_path: str
):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_dice': best_dice
    }, checkpoint_path)


def load_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    checkpoint_path: str
):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['best_dice']


def train(config: Config, args):
    """
    Main training function.
    
    Args:
        config: Configuration object
        args: Command line arguments
    """
    # Set random seed
    set_seed(config.training.seed)
    
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=config.data.data_root,
        batch_size=config.training.batch_size,
        image_size=config.data.image_size,
        num_workers=config.training.num_workers,
        use_augmentation=config.data.use_augmentation,
        use_generated=config.data.use_generated,
        val_split=config.data.val_split,
        seed=config.training.seed
    )
    
    if train_loader is None or len(train_loader) == 0:
        print("\n" + "="*60)
        print("ERROR: No training data found!")
        print("="*60)
        print("\nPlease download the dataset:")
        print("1. Go to: https://www.kaggle.com/datasets/sanadalali/satellite-images-for-road-segmentation")
        print("2. Download and extract to the 'data' folder")
        print("3. Expected structure:")
        print("   data/")
        print("   ├── test_set_images/")
        print("   │   ├── test_1/test_1.png")
        print("   │   └── ...")
        print("   └── training/")
        print("       ├── images/satImage_XXX.png")
        print("       ├── groundtruth/satImage_XXX.png")
        print("       ├── images_generated/ (optional)")
        print("       └── groundtruth_generated/ (optional)")
        return
    
    # Create model
    print(f"\nCreating {config.model.model_type} model...")
    model = get_model(
        model_type=config.model.model_type,
        in_channels=config.data.num_channels,
        num_classes=config.data.num_classes,
        encoder_channels=config.model.encoder_channels,
        bottleneck_channels=config.model.bottleneck_channels,
        use_batch_norm=config.model.use_batch_norm,
        dropout_rate=config.model.dropout_rate
    )
    model = model.to(device)
    
    total_params, trainable_params = model.get_num_parameters()
    print(f"Model parameters: {trainable_params:,}")
    
    # Loss function
    criterion = get_loss_function(
        loss_type=config.training.loss_function,
        bce_weight=config.training.bce_weight,
        dice_weight=config.training.dice_weight,
        focal_alpha=config.training.focal_alpha,
        focal_gamma=config.training.focal_gamma
    )
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Learning rate scheduler
    if config.training.scheduler_type == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience,
            min_lr=config.training.min_lr,
            verbose=True
        )
    elif config.training.scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs,
            eta_min=config.training.min_lr
        )
    else:
        scheduler = None
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config.training.use_amp and device.type == 'cuda' else None
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config.model.model_type}_{timestamp}"
    log_dir = os.path.join(config.output.log_dir, experiment_name)
    writer = SummaryWriter(log_dir)
    
    checkpoint_dir = config.output.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training for {config.training.num_epochs} epochs...")
    print(f"Logging to: {log_dir}")
    print("-" * 60)
    
    best_dice = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, config.training.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.training.num_epochs}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, use_amp=config.training.use_amp
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        if scheduler:
            if config.training.scheduler_type == 'reduce_on_plateau':
                scheduler.step(val_metrics['dice'])
            else:
                scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        writer.add_scalars('Loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)
        
        writer.add_scalars('Dice', {
            'train': train_metrics['dice'],
            'val': val_metrics['dice']
        }, epoch)
        
        writer.add_scalars('IoU', {
            'train': train_metrics['iou'],
            'val': val_metrics['iou']
        }, epoch)
        
        writer.add_scalar('Learning Rate', current_lr, epoch)
        
        # Print epoch summary
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | Dice: {train_metrics['dice']:.4f} | IoU: {train_metrics['iou']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f}")
        print(f"  LR: {current_lr:.2e}")
        
        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            best_epoch = epoch
            patience_counter = 0
            
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_dice,
                os.path.join(checkpoint_dir, f'{config.model.model_type}_best.pt')
            )
            print(f"  ★ New best model saved! (Dice: {best_dice:.4f})")
        else:
            patience_counter += 1
        
        # Save latest model
        save_checkpoint(
            model, optimizer, scheduler, epoch, best_dice,
            os.path.join(checkpoint_dir, f'{config.model.model_type}_latest.pt')
        )
        
        # Early stopping
        if patience_counter >= config.training.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best validation Dice: {best_dice:.4f} at epoch {best_epoch}")
            break
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation Dice: {best_dice:.4f} at epoch {best_epoch}")
    print(f"Model saved to: {os.path.join(checkpoint_dir, f'{config.model.model_type}_best.pt')}")
    print("="*60)
    
    writer.close()
    
    return best_dice


def main():
    parser = argparse.ArgumentParser(description='Train U-Net for Road Segmentation')
    
    parser.add_argument('--model', type=str, default='baseline',
                       choices=['baseline', 'attention'],
                       help='Model type: baseline or attention')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to data directory (overrides config)')
    parser.add_argument('--loss', type=str, default=None,
                       choices=['bce', 'dice', 'bce_dice', 'focal'],
                       help='Loss function (overrides config)')
    parser.add_argument('--no-aug', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--no-generated', action='store_true',
                       help='Use only original images (not Pix2Pix generated)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config(model_type=args.model)
    
    # Override with command line arguments
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.data:
        config.data.data_root = args.data
    if args.loss:
        config.training.loss_function = args.loss
    if args.no_aug:
        config.data.use_augmentation = False
    if args.no_generated:
        config.data.use_generated = False
    
    # Print configuration
    print("="*60)
    print("U-Net Road Segmentation Training")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Model: {config.model.model_type}")
    print(f"  Image size: {config.data.image_size}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Loss function: {config.training.loss_function}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Data augmentation: {config.data.use_augmentation}")
    print(f"  Use generated images: {config.data.use_generated}")
    
    # Train
    train(config, args)


if __name__ == '__main__':
    main()
