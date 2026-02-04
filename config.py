"""
Configuration file for U-Net Road Segmentation Project
========================================================

This file contains all hyperparameters and configuration settings
for the satellite road segmentation project.

Reference Paper:
    "A Comprehensive Review of U-Net and Its Variants: Advances and 
    Applications in Medical Image Segmentation"
    IET Image Processing, 2025
    https://arxiv.org/abs/2502.06895
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    """Configuration for dataset and preprocessing."""
    
    # Dataset paths
    # Expected structure:
    # data/
    # ├── test_set_images/test_1/test_1.png, ...
    # └── training/
    #     ├── images/satImage_XXX.png (100 original)
    #     ├── groundtruth/satImage_XXX.png (100 masks)
    #     ├── images_generated/ (1000+ generated)
    #     └── groundtruth_generated/
    data_root: str = "data"
    
    # Image settings
    image_size: Tuple[int, int] = (256, 256)  # (H, W)
    num_channels: int = 3  # RGB input
    num_classes: int = 1   # Binary segmentation (road vs background)
    
    # Data split ratio for validation (from training data)
    val_split: float = 0.15  # 15% of training for validation
    
    # Whether to use generated images (Pix2Pix augmented)
    use_generated: bool = True  # Set to False for original images only
    
    # Augmentation settings
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5
    rotation_limit: int = 15
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2


@dataclass
class ModelConfig:
    """Configuration for U-Net model architecture."""
    
    # Model type: "baseline" or "attention"
    model_type: str = "baseline"
    
    # Encoder configuration
    encoder_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    
    # Bottleneck channels
    bottleneck_channels: int = 1024
    
    # Use batch normalization
    use_batch_norm: bool = True
    
    # Dropout rate (0 to disable)
    dropout_rate: float = 0.1
    
    # For Attention U-Net: attention reduction ratio
    attention_reduction_ratio: int = 2


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Training settings
    batch_size: int = 8
    num_epochs: int = 100
    early_stopping_patience: int = 15
    
    # Learning rate settings
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Learning rate scheduler
    scheduler_type: str = "reduce_on_plateau"  # "reduce_on_plateau", "cosine", "step"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_lr: float = 1e-7
    
    # Loss function: "bce", "dice", "bce_dice", "focal"
    loss_function: str = "bce_dice"
    bce_weight: float = 0.5  # Weight for BCE in combined loss
    dice_weight: float = 0.5  # Weight for Dice in combined loss
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Mixed precision training
    use_amp: bool = True
    
    # Gradient clipping
    gradient_clip_value: float = 1.0
    
    # Number of workers for data loading
    num_workers: int = 4
    
    # Random seed for reproducibility
    seed: int = 42


@dataclass
class OutputConfig:
    """Configuration for outputs and logging."""
    
    # Output directories
    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    visualization_dir: str = "outputs/visualizations"
    
    # Logging settings
    log_interval: int = 10  # Log every N batches
    save_best_only: bool = True
    
    # Visualization settings
    num_visualization_samples: int = 5


@dataclass
class Config:
    """Main configuration class combining all settings."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Device settings
    device: str = "cuda"  # "cuda" or "cpu"
    
    def __post_init__(self):
        """Create output directories if they don't exist."""
        os.makedirs(self.output.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output.log_dir, exist_ok=True)
        os.makedirs(self.output.visualization_dir, exist_ok=True)


def get_config(model_type: str = "baseline") -> Config:
    """
    Get configuration with specified model type.
    
    Args:
        model_type: Either "baseline" for standard U-Net or "attention" for Attention U-Net
    
    Returns:
        Config object with all settings
    """
    config = Config()
    config.model.model_type = model_type
    return config


# Experiment configurations for comparison
EXPERIMENTS = {
    "baseline": {
        "model_type": "baseline",
        "description": "Standard U-Net with skip connections"
    },
    "attention": {
        "model_type": "attention", 
        "description": "U-Net with attention gates on skip connections"
    },
    "baseline_augmented": {
        "model_type": "baseline",
        "use_augmentation": True,
        "description": "Standard U-Net with enhanced data augmentation"
    }
}


if __name__ == "__main__":
    # Print default configuration
    config = get_config()
    print("=" * 60)
    print("U-Net Road Segmentation Configuration")
    print("=" * 60)
    print(f"\nData Config:")
    print(f"  Image Size: {config.data.image_size}")
    print(f"  Num Classes: {config.data.num_classes}")
    print(f"\nModel Config:")
    print(f"  Model Type: {config.model.model_type}")
    print(f"  Encoder Channels: {config.model.encoder_channels}")
    print(f"  Bottleneck Channels: {config.model.bottleneck_channels}")
    print(f"\nTraining Config:")
    print(f"  Batch Size: {config.training.batch_size}")
    print(f"  Learning Rate: {config.training.learning_rate}")
    print(f"  Num Epochs: {config.training.num_epochs}")
    print(f"  Loss Function: {config.training.loss_function}")
