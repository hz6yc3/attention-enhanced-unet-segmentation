"""
U-Net Models for Road Segmentation
===================================

This module provides implementations of:
1. Baseline U-Net
2. Attention U-Net (with attention gates on skip connections)

Reference Paper:
"A Comprehensive Review of U-Net and Its Variants: Advances and 
Applications in Medical Image Segmentation"
IET Image Processing, 2025
https://arxiv.org/abs/2502.06895
"""

from .unet import UNet
from .attention_unet import AttentionUNet

__all__ = ['UNet', 'AttentionUNet']


def get_model(model_type: str = 'baseline', **kwargs):
    """
    Factory function to get model by type.
    
    Args:
        model_type: 'baseline' for standard U-Net, 'attention' for Attention U-Net
        **kwargs: Additional arguments passed to model constructor
    
    Returns:
        Model instance
    """
    if model_type.lower() == 'baseline':
        return UNet(**kwargs)
    elif model_type.lower() == 'attention':
        return AttentionUNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from: baseline, attention")
