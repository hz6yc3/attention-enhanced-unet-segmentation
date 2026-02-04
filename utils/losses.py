"""
Loss Functions for Segmentation
================================

This module implements various loss functions for binary segmentation:
- Binary Cross-Entropy Loss
- Dice Loss
- Combined BCE + Dice Loss
- Focal Loss

These loss functions are commonly used in U-Net based segmentation
as discussed in the reference paper.

Reference: "A Comprehensive Review of U-Net and Its Variants"
https://arxiv.org/abs/2502.06895
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    
    Dice Loss = 1 - Dice Coefficient
    
    The Dice coefficient measures the overlap between prediction and target,
    making this loss particularly suitable for imbalanced segmentation tasks.
    """
    
    def __init__(self, smooth: float = 1e-6):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice Loss.
        
        Args:
            pred: Predictions (B, 1, H, W), raw logits or probabilities
            target: Ground truth (B, 1, H, W), values in {0, 1}
        
        Returns:
            Dice loss value
        """
        # Apply sigmoid if predictions are logits
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice_coeff = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        return 1 - dice_coeff


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Dice Loss.
    
    This combination leverages:
    - BCE: Pixel-wise classification accuracy
    - Dice: Region-based overlap optimization
    
    Total Loss = bce_weight * BCE + dice_weight * Dice
    """
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6
    ):
        """
        Args:
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
            smooth: Smoothing factor for Dice loss
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined BCE + Dice loss.
        
        Args:
            pred: Predictions (B, 1, H, W), raw logits
            target: Ground truth (B, 1, H, W), values in {0, 1}
        
        Returns:
            Combined loss value
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL = -alpha * (1 - p)^gamma * log(p)
    
    Focal loss down-weights easy examples and focuses training on hard examples,
    which is particularly useful when the background dominates the image.
    
    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            pred: Predictions (B, 1, H, W), raw logits
            target: Ground truth (B, 1, H, W), values in {0, 1}
        
        Returns:
            Focal loss value
        """
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Get probabilities
        pred_prob = torch.sigmoid(pred)
        
        # Compute focal weight
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Compute focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss for imbalanced segmentation.
    
    Tversky index generalizes Dice coefficient with adjustable weights
    for false positives and false negatives.
    
    TI = TP / (TP + alpha*FP + beta*FN)
    
    When alpha = beta = 0.5, it becomes Dice coefficient.
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1e-6
    ):
        """
        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Tversky Loss.
        """
        # Apply sigmoid if predictions are logits
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1 - tversky


def get_loss_function(
    loss_type: str = 'bce_dice',
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0
) -> nn.Module:
    """
    Get loss function by name.
    
    Args:
        loss_type: One of 'bce', 'dice', 'bce_dice', 'focal', 'tversky'
        bce_weight: Weight for BCE in combined loss
        dice_weight: Weight for Dice in combined loss
        focal_alpha: Alpha parameter for Focal loss
        focal_gamma: Gamma parameter for Focal loss
    
    Returns:
        Loss function module
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'bce_dice':
        return BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)
    elif loss_type == 'focal':
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    elif loss_type == 'tversky':
        return TverskyLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                        f"Choose from: bce, dice, bce_dice, focal, tversky")


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    # Create dummy predictions and targets
    pred = torch.randn(4, 1, 128, 128)  # Logits
    target = (torch.rand(4, 1, 128, 128) > 0.5).float()
    
    # Test each loss function
    losses = ['bce', 'dice', 'bce_dice', 'focal', 'tversky']
    
    for loss_name in losses:
        loss_fn = get_loss_function(loss_name)
        loss_value = loss_fn(pred, target)
        print(f"{loss_name:12s} loss: {loss_value.item():.4f}")
    
    print("\nLoss functions test completed!")
