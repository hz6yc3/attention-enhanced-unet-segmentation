"""
Evaluation Script for U-Net Road Segmentation
===============================================

This script evaluates trained U-Net models on the test set
and generates comprehensive metrics and visualizations.

Usage:
    python evaluate.py --model baseline --checkpoint outputs/checkpoints/baseline_best.pt
    python evaluate.py --model attention --checkpoint outputs/checkpoints/attention_best.pt

Reference Paper:
    "A Comprehensive Review of U-Net and Its Variants"
    IET Image Processing, 2025
"""

import os
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import get_config
from models import get_model
from utils.dataset import get_dataloaders, get_val_transforms, RoadSegmentationDataset
from utils.losses import get_loss_function
from utils.metrics import (
    MetricTracker, 
    dice_coefficient, 
    iou_score, 
    pixel_accuracy,
    precision_recall
)


def denormalize(image: torch.Tensor) -> np.ndarray:
    """
    Denormalize image for visualization.
    
    Args:
        image: Normalized image tensor (C, H, W)
    
    Returns:
        Denormalized numpy array (H, W, C) in range [0, 1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    image = image.cpu() * std + mean
    image = torch.clamp(image, 0, 1)
    image = image.permute(1, 2, 0).numpy()
    
    return image


def visualize_predictions(
    images: torch.Tensor,
    masks: torch.Tensor,
    predictions: torch.Tensor,
    num_samples: int = 5,
    save_path: str = None,
    threshold: float = 0.5
):
    """
    Visualize predictions compared to ground truth.
    
    Args:
        images: Input images (B, C, H, W)
        masks: Ground truth masks (B, 1, H, W)
        predictions: Predicted masks (B, 1, H, W)
        num_samples: Number of samples to visualize
        save_path: Path to save the figure
        threshold: Threshold for binarizing predictions
    """
    num_samples = min(num_samples, images.shape[0])
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Denormalize image
        img = denormalize(images[i])
        
        # Get mask and prediction
        mask = masks[i, 0].cpu().numpy()
        pred = (predictions[i, 0] > threshold).float().cpu().numpy()
        pred_prob = predictions[i, 0].cpu().numpy()
        
        # Calculate metrics for this sample
        dice = dice_coefficient(
            predictions[i:i+1], 
            masks[i:i+1].cpu(), 
            threshold=threshold
        ).item()
        iou = iou_score(
            predictions[i:i+1], 
            masks[i:i+1].cpu(), 
            threshold=threshold
        ).item()
        
        # Plot
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_prob, cmap='hot')
        axes[i, 2].set_title('Prediction (Probability)')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(pred, cmap='gray')
        axes[i, 3].set_title(f'Prediction (Dice: {dice:.4f}, IoU: {iou:.4f})')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.close()


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create overlay of mask on image."""
    overlay = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[mask > 0.5] = [1, 0, 0]  # Red for road
    overlay = (1 - alpha) * overlay + alpha * mask_colored
    return np.clip(overlay, 0, 1)


def visualize_comparison(
    images: torch.Tensor,
    masks: torch.Tensor,
    baseline_preds: torch.Tensor,
    attention_preds: torch.Tensor,
    num_samples: int = 3,
    save_path: str = None,
    threshold: float = 0.5
):
    """
    Visualize comparison between baseline and attention models.
    """
    num_samples = min(num_samples, images.shape[0])
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        img = denormalize(images[i])
        mask = masks[i, 0].cpu().numpy()
        baseline = (baseline_preds[i, 0] > threshold).float().cpu().numpy()
        attention = (attention_preds[i, 0] > threshold).float().cpu().numpy()
        
        # Metrics
        baseline_dice = dice_coefficient(baseline_preds[i:i+1], masks[i:i+1].cpu()).item()
        attention_dice = dice_coefficient(attention_preds[i:i+1], masks[i:i+1].cpu()).item()
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(baseline, cmap='gray')
        axes[i, 2].set_title(f'Baseline (Dice: {baseline_dice:.4f})')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(attention, cmap='gray')
        axes[i, 3].set_title(f'Attention (Dice: {attention_dice:.4f})')
        axes[i, 3].axis('off')
        
        # Difference
        diff = np.abs(attention.astype(float) - baseline.astype(float))
        axes[i, 4].imshow(diff, cmap='hot')
        axes[i, 4].set_title('Difference')
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader,
    criterion,
    device: str,
    visualization_dir: str = None,
    num_vis_samples: int = 5
) -> dict:
    """
    Evaluate model on test set.
    
    Args:
        model: Model to evaluate
        dataloader: Test dataloader
        criterion: Loss function
        device: Device to run on
        visualization_dir: Directory to save visualizations
        num_vis_samples: Number of samples to visualize
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    metric_tracker = MetricTracker()
    running_loss = 0.0
    
    all_images = []
    all_masks = []
    all_predictions = []
    
    print("Evaluating model...")
    pbar = tqdm(dataloader, desc="Evaluation")
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        running_loss += loss.item()
        
        pred_probs = torch.sigmoid(outputs)
        metric_tracker.update(pred_probs, masks)
        
        # Store samples for visualization
        if len(all_images) < num_vis_samples:
            all_images.append(images.cpu())
            all_masks.append(masks.cpu())
            all_predictions.append(pred_probs.cpu())
    
    # Aggregate metrics
    metrics = metric_tracker.get_metrics()
    metrics['loss'] = running_loss / len(dataloader)
    
    # Generate visualizations
    if visualization_dir and len(all_images) > 0:
        os.makedirs(visualization_dir, exist_ok=True)
        
        images = torch.cat(all_images, dim=0)[:num_vis_samples]
        masks = torch.cat(all_masks, dim=0)[:num_vis_samples]
        predictions = torch.cat(all_predictions, dim=0)[:num_vis_samples]
        
        visualize_predictions(
            images, masks, predictions,
            num_samples=num_vis_samples,
            save_path=os.path.join(visualization_dir, 'predictions.png')
        )
    
    return metrics, (all_images, all_masks, all_predictions)


def main():
    parser = argparse.ArgumentParser(description='Evaluate U-Net for Road Segmentation')
    
    parser.add_argument('--model', type=str, default='baseline',
                       choices=['baseline', 'attention'],
                       help='Model type')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--output', type=str, default='outputs/evaluation',
                       help='Output directory for results')
    parser.add_argument('--num-vis', type=int, default=5,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Setup
    config = get_config(model_type=args.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading {args.model} model from {args.checkpoint}...")
    model = get_model(
        model_type=args.model,
        in_channels=config.data.num_channels,
        num_classes=config.data.num_classes,
        encoder_channels=config.model.encoder_channels,
        bottleneck_channels=config.model.bottleneck_channels,
        use_batch_norm=config.model.use_batch_norm,
        dropout_rate=0  # No dropout during evaluation
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    total_params, _ = model.get_num_parameters()
    print(f"Model parameters: {total_params:,}")
    
    # Load data
    print("\nLoading validation data for evaluation...")
    # Note: Test set has no ground truth, so we evaluate on validation set
    _, val_loader, test_loader = get_dataloaders(
        data_root=args.data,
        batch_size=8,
        image_size=config.data.image_size,
        num_workers=4,
        use_augmentation=False,
        use_generated=False,  # Evaluate on original images only
        val_split=0.2  # Use 20% for validation
    )
    
    # Use validation loader for evaluation (test set has no ground truth)
    eval_loader = val_loader
    
    if eval_loader is None or len(eval_loader) == 0:
        print("Error: No validation data found!")
        return
    
    print(f"Evaluating on {len(eval_loader.dataset)} validation images")
    
    # Loss function
    criterion = get_loss_function(config.training.loss_function)
    
    # Evaluate
    output_dir = os.path.join(args.output, args.model)
    os.makedirs(output_dir, exist_ok=True)
    
    metrics, (images, masks, predictions) = evaluate_model(
        model, eval_loader, criterion, device,
        visualization_dir=output_dir,
        num_vis_samples=args.num_vis
    )
    
    # Print results
    print("\n" + "="*60)
    print(f"Evaluation Results - {args.model.upper()} U-Net")
    print("="*60)
    print(f"  Loss:      {metrics['loss']:.4f}")
    print(f"  Dice:      {metrics['dice']:.4f}")
    print(f"  IoU:       {metrics['iou']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print("="*60)
    
    # Save results
    results = {
        'model': args.model,
        'checkpoint': args.checkpoint,
        'num_parameters': total_params,
        'metrics': metrics
    }
    
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print(f"Visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
