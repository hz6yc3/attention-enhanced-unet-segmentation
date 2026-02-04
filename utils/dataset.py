"""
Dataset module for Satellite Road Segmentation
===============================================

This module provides dataset classes and data loading utilities
for the satellite road segmentation task.

Dataset: Satellite Images for Road Segmentation (Kaggle)
https://www.kaggle.com/datasets/sanadalali/satellite-images-for-road-segmentation

Dataset Structure:
------------------
data/
├── test_set_images/
│   ├── test_1/test_1.png
│   ├── test_2/test_2.png
│   └── ... (50 test images)
└── training/
    ├── images/
    │   └── satImage_001.png to satImage_100.png (100 original images)
    ├── groundtruth/
    │   └── satImage_001.png to satImage_100.png (100 masks)
    ├── images_generated/
    │   └── generated_image_0000.png ... (1000+ generated images)
    └── groundtruth_generated/
        └── generated_image_0000.png ... (1000+ generated masks)
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RoadSegmentationDataset(Dataset):
    """
    PyTorch Dataset for satellite road segmentation.
    
    This dataset loads RGB satellite images and their corresponding
    binary segmentation masks (road vs. background).
    
    Args:
        image_paths: List of paths to input images
        mask_paths: List of paths to mask images (can be None for test set)
        transform: Albumentations transform pipeline
        image_size: Target size for images (height, width)
    """
    
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: Optional[List[str]] = None,
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (256, 256)
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.image_size = image_size
        
        # Verify paths exist
        valid_indices = []
        for i, img_path in enumerate(self.image_paths):
            if os.path.exists(img_path):
                valid_indices.append(i)
        
        self.image_paths = [self.image_paths[i] for i in valid_indices]
        if self.mask_paths:
            self.mask_paths = [self.mask_paths[i] for i in valid_indices]
        
        if len(self.image_paths) == 0:
            print("Warning: No valid images found!")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            image: Tensor of shape (3, H, W) with normalized values
            mask: Tensor of shape (1, H, W) with values in {0, 1}
        """
        # Load image
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        
        # Load mask
        if self.mask_paths and os.path.exists(self.mask_paths[idx]):
            mask = np.array(Image.open(self.mask_paths[idx]).convert('L'))
            # Binarize mask (ensure values are 0 or 1)
            mask = (mask > 127).astype(np.float32)
        else:
            # Create dummy mask for test set
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Default transform: resize and normalize
            image = np.array(Image.fromarray(image).resize(
                (self.image_size[1], self.image_size[0]), 
                Image.BILINEAR
            ))
            mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(
                (self.image_size[1], self.image_size[0]), 
                Image.NEAREST
            )) / 255.0
            
            # Convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).float()
        
        # Ensure mask has correct shape
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        return image, mask
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata about a sample."""
        return {
            'image_path': self.image_paths[idx],
            'mask_path': self.mask_paths[idx] if self.mask_paths else None
        }


def get_train_transforms(image_size: Tuple[int, int] = (256, 256)) -> A.Compose:
    """
    Get augmentation pipeline for training.
    
    Augmentations inspired by best practices for satellite imagery:
    - Geometric: flips, rotation
    - Photometric: brightness, contrast adjustments
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,
            p=0.5
        ),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=1
            ),
        ], p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transforms(image_size: Tuple[int, int] = (256, 256)) -> A.Compose:
    """Get transform pipeline for validation/testing (no augmentation)."""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def collect_dataset_paths(
    data_root: str,
    use_generated: bool = True
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Collect all image and mask paths from the dataset.
    
    Dataset structure:
    data_root/
    ├── test_set_images/
    │   ├── test_1/test_1.png
    │   └── ...
    └── training/
        ├── images/satImage_XXX.png
        ├── groundtruth/satImage_XXX.png
        ├── images_generated/generated_image_XXXX.png
        └── groundtruth_generated/generated_image_XXXX.png
    
    Args:
        data_root: Root directory of the dataset
        use_generated: Whether to include generated images for training
    
    Returns:
        Tuple of (train_images, train_masks, test_images, test_masks)
    """
    data_root = Path(data_root)
    
    train_images = []
    train_masks = []
    test_images = []
    test_masks = []  # Will be empty for this dataset
    
    # ========== Training Data ==========
    training_dir = data_root / 'training'
    
    # Original training images
    images_dir = training_dir / 'images'
    groundtruth_dir = training_dir / 'groundtruth'
    
    if images_dir.exists():
        for img_path in sorted(images_dir.glob('*.png')):
            mask_path = groundtruth_dir / img_path.name
            if mask_path.exists():
                train_images.append(str(img_path))
                train_masks.append(str(mask_path))
    
    # Generated images (optional, for data augmentation)
    if use_generated:
        images_gen_dir = training_dir / 'images_generated'
        groundtruth_gen_dir = training_dir / 'groundtruth_generated'
        
        if images_gen_dir.exists():
            for img_path in sorted(images_gen_dir.glob('*.png')):
                mask_path = groundtruth_gen_dir / img_path.name
                if mask_path.exists():
                    train_images.append(str(img_path))
                    train_masks.append(str(mask_path))
    
    # ========== Test Data ==========
    test_dir = data_root / 'test_set_images'
    
    if test_dir.exists():
        # Test images are in nested folders: test_1/test_1.png
        for test_folder in sorted(test_dir.iterdir()):
            if test_folder.is_dir() and test_folder.name.startswith('test_'):
                img_path = test_folder / f"{test_folder.name}.png"
                if img_path.exists():
                    test_images.append(str(img_path))
                    test_masks.append(None)  # No ground truth for test
    
    print(f"Dataset paths collected:")
    print(f"  Original training images: {len([p for p in train_images if 'satImage' in p])}")
    print(f"  Generated training images: {len([p for p in train_images if 'generated' in p])}")
    print(f"  Total training: {len(train_images)}")
    print(f"  Test images: {len(test_images)}")
    
    return train_images, train_masks, test_images, test_masks


def get_dataloaders(
    data_root: str,
    batch_size: int = 8,
    image_size: Tuple[int, int] = (256, 256),
    num_workers: int = 4,
    use_augmentation: bool = True,
    use_generated: bool = True,
    val_split: float = 0.15,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_root: Root directory containing the dataset
        batch_size: Batch size for dataloaders
        image_size: Target image size (H, W)
        num_workers: Number of worker processes for data loading
        use_augmentation: Whether to use data augmentation for training
        use_generated: Whether to include generated images in training
        val_split: Fraction of training data to use for validation
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set seed for reproducibility
    random.seed(seed)
    
    # Collect all paths
    train_images, train_masks, test_images, test_masks = collect_dataset_paths(
        data_root, use_generated=use_generated
    )
    
    if len(train_images) == 0:
        print("\n" + "="*60)
        print("ERROR: No training images found!")
        print("="*60)
        print(f"\nLooking in: {data_root}")
        print("\nExpected structure:")
        print("  data/")
        print("  ├── test_set_images/")
        print("  │   ├── test_1/test_1.png")
        print("  │   └── ...")
        print("  └── training/")
        print("      ├── images/satImage_XXX.png")
        print("      ├── groundtruth/satImage_XXX.png")
        print("      ├── images_generated/ (optional)")
        print("      └── groundtruth_generated/ (optional)")
        print("\nPlease check your data directory structure.")
        print("="*60 + "\n")
    
    # Split training data into train and validation
    indices = list(range(len(train_images)))
    random.shuffle(indices)
    
    val_size = int(len(indices) * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    # Create path lists for each split
    train_img_paths = [train_images[i] for i in train_indices]
    train_mask_paths = [train_masks[i] for i in train_indices]
    val_img_paths = [train_images[i] for i in val_indices]
    val_mask_paths = [train_masks[i] for i in val_indices]
    
    # Get transforms
    train_transform = get_train_transforms(image_size) if use_augmentation else get_val_transforms(image_size)
    val_transform = get_val_transforms(image_size)
    
    # Create datasets
    train_dataset = RoadSegmentationDataset(
        image_paths=train_img_paths,
        mask_paths=train_mask_paths,
        transform=train_transform,
        image_size=image_size
    )
    
    val_dataset = RoadSegmentationDataset(
        image_paths=val_img_paths,
        mask_paths=val_mask_paths,
        transform=val_transform,
        image_size=image_size
    )
    
    test_dataset = RoadSegmentationDataset(
        image_paths=test_images,
        mask_paths=None,  # No ground truth for test
        transform=val_transform,
        image_size=image_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    ) if len(train_dataset) > 0 else None
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    ) if len(val_dataset) > 0 else None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    ) if len(test_dataset) > 0 else None
    
    print(f"\nDataloader sizes:")
    print(f"  Train: {len(train_dataset)} images ({len(train_loader) if train_loader else 0} batches)")
    print(f"  Val:   {len(val_dataset)} images ({len(val_loader) if val_loader else 0} batches)")
    print(f"  Test:  {len(test_dataset)} images ({len(test_loader) if test_loader else 0} batches)")
    
    return train_loader, val_loader, test_loader


def get_dataloaders_original_only(
    data_root: str,
    batch_size: int = 8,
    image_size: Tuple[int, int] = (256, 256),
    num_workers: int = 4,
    use_augmentation: bool = True,
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders using ONLY original images (no generated).
    
    Use this for a cleaner experiment setup where you control augmentation.
    
    Args:
        Same as get_dataloaders
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    return get_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        use_augmentation=use_augmentation,
        use_generated=False,  # Don't use generated images
        val_split=val_split,
        seed=seed
    )


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset module...")
    print("="*60)
    
    # Test with the actual dataset structure
    data_root = "data"
    
    if os.path.exists(data_root):
        # Test path collection
        train_imgs, train_masks, test_imgs, test_masks = collect_dataset_paths(
            data_root, use_generated=True
        )
        
        if len(train_imgs) > 0:
            print(f"\nSample training paths:")
            print(f"  Image: {train_imgs[0]}")
            print(f"  Mask:  {train_masks[0]}")
            
            if len(test_imgs) > 0:
                print(f"\nSample test path:")
                print(f"  Image: {test_imgs[0]}")
            
            # Test dataloader creation
            print("\nCreating dataloaders...")
            train_loader, val_loader, test_loader = get_dataloaders(
                data_root=data_root,
                batch_size=4,
                image_size=(256, 256),
                num_workers=0,
                use_augmentation=True,
                use_generated=True
            )
            
            if train_loader and len(train_loader) > 0:
                # Get a batch
                images, masks = next(iter(train_loader))
                print(f"\nBatch shapes:")
                print(f"  Images: {images.shape}")
                print(f"  Masks: {masks.shape}")
                print(f"  Image value range: [{images.min():.3f}, {images.max():.3f}]")
                print(f"  Mask unique values: {torch.unique(masks).tolist()}")
    else:
        print(f"\nData directory '{data_root}' not found.")
        print("Please download the dataset and place it in the 'data' folder.")
    
    print("\n" + "="*60)
    print("Dataset module test completed!")
