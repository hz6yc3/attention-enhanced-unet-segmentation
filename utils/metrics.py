"""
Evaluation Metrics for Segmentation
====================================

This module implements evaluation metrics commonly used for
semantic segmentation tasks, as discussed in the reference paper:

"A Comprehensive Review of U-Net and Its Variants"
- Dice Coefficient
- IoU (Jaccard Index)
- Pixel Accuracy

Reference: https://arxiv.org/abs/2502.06895
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
    threshold: Optional[float] = 0.5
) -> torch.Tensor:
    """
    Compute Dice Coefficient (F1 Score for segmentation).
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Args:
        pred: Predicted mask (B, 1, H, W) with values in [0, 1]
        target: Ground truth mask (B, 1, H, W) with values in {0, 1}
        smooth: Smoothing factor to avoid division by zero
        threshold: Threshold to binarize predictions (None for soft Dice)
    
    Returns:
        Dice coefficient as a scalar tensor
    """
    if threshold is not None:
        pred = (pred > threshold).float()
    
    # Flatten
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return dice


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
    threshold: Optional[float] = 0.5
) -> torch.Tensor:
    """
    Compute Intersection over Union (Jaccard Index).
    
    IoU = |A ∩ B| / |A ∪ B|
    
    Args:
        pred: Predicted mask (B, 1, H, W) with values in [0, 1]
        target: Ground truth mask (B, 1, H, W) with values in {0, 1}
        smooth: Smoothing factor to avoid division by zero
        threshold: Threshold to binarize predictions
    
    Returns:
        IoU score as a scalar tensor
    """
    if threshold is not None:
        pred = (pred > threshold).float()
    
    # Flatten
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


def pixel_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Compute pixel-wise accuracy.
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        pred: Predicted mask (B, 1, H, W) with values in [0, 1]
        target: Ground truth mask (B, 1, H, W) with values in {0, 1}
        threshold: Threshold to binarize predictions
    
    Returns:
        Pixel accuracy as a scalar tensor
    """
    pred = (pred > threshold).float()
    correct = (pred == target).float()
    accuracy = correct.sum() / correct.numel()
    
    return accuracy


def precision_recall(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute precision and recall.
    
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    
    Args:
        pred: Predicted mask (B, 1, H, W)
        target: Ground truth mask (B, 1, H, W)
        threshold: Threshold to binarize predictions
        smooth: Smoothing factor
    
    Returns:
        Tuple of (precision, recall)
    """
    pred = (pred > threshold).float()
    
    # Flatten
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    
    return precision, recall


class MetricTracker:
    """
    Track multiple metrics during training/evaluation.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.dice_scores = []
        self.iou_scores = []
        self.accuracies = []
        self.precisions = []
        self.recalls = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
        """
        Update metrics with a batch of predictions.
        
        Args:
            pred: Predicted masks (B, 1, H, W)
            target: Ground truth masks (B, 1, H, W)
            threshold: Threshold for binarization
        """
        with torch.no_grad():
            # Apply sigmoid if predictions are logits
            if pred.min() < 0 or pred.max() > 1:
                pred = torch.sigmoid(pred)
            
            dice = dice_coefficient(pred, target, threshold=threshold)
            iou = iou_score(pred, target, threshold=threshold)
            acc = pixel_accuracy(pred, target, threshold=threshold)
            prec, rec = precision_recall(pred, target, threshold=threshold)
            
            self.dice_scores.append(dice.item())
            self.iou_scores.append(iou.item())
            self.accuracies.append(acc.item())
            self.precisions.append(prec.item())
            self.recalls.append(rec.item())
    
    def get_metrics(self) -> dict:
        """Get average of all tracked metrics."""
        return {
            'dice': np.mean(self.dice_scores) if self.dice_scores else 0.0,
            'iou': np.mean(self.iou_scores) if self.iou_scores else 0.0,
            'accuracy': np.mean(self.accuracies) if self.accuracies else 0.0,
            'precision': np.mean(self.precisions) if self.precisions else 0.0,
            'recall': np.mean(self.recalls) if self.recalls else 0.0
        }
    
    def __str__(self) -> str:
        metrics = self.get_metrics()
        return (f"Dice: {metrics['dice']:.4f} | "
                f"IoU: {metrics['iou']:.4f} | "
                f"Acc: {metrics['accuracy']:.4f} | "
                f"Prec: {metrics['precision']:.4f} | "
                f"Rec: {metrics['recall']:.4f}")


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics module...")
    
    # Create dummy predictions and targets
    pred = torch.rand(4, 1, 128, 128)
    target = (torch.rand(4, 1, 128, 128) > 0.5).float()
    
    # Compute metrics
    dice = dice_coefficient(pred, target)
    iou = iou_score(pred, target)
    acc = pixel_accuracy(pred, target)
    prec, rec = precision_recall(pred, target)
    
    print(f"Dice Coefficient: {dice.item():.4f}")
    print(f"IoU Score: {iou.item():.4f}")
    print(f"Pixel Accuracy: {acc.item():.4f}")
    print(f"Precision: {prec.item():.4f}")
    print(f"Recall: {rec.item():.4f}")
    
    # Test metric tracker
    print("\nTesting MetricTracker...")
    tracker = MetricTracker()
    for _ in range(5):
        pred = torch.rand(4, 1, 128, 128)
        target = (torch.rand(4, 1, 128, 128) > 0.5).float()
        tracker.update(pred, target)
    
    print(f"Average metrics: {tracker}")
    
    print("\nMetrics module test completed!")
