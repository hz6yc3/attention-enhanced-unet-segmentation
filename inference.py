"""
Inference Script for U-Net Road Segmentation
==============================================

This script runs inference on single images or directories
using trained U-Net models.

Usage:
    python inference.py --model baseline --checkpoint outputs/checkpoints/baseline_best.pt --input image.jpg
    python inference.py --model attention --checkpoint outputs/checkpoints/attention_best.pt --input images/

Reference Paper:
    "A Comprehensive Review of U-Net and Its Variants"
    IET Image Processing, 2025
"""

import os
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import get_config
from models import get_model


def get_inference_transform(image_size=(256, 256)):
    """Get transform for inference."""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def load_image(image_path: str, transform) -> tuple:
    """
    Load and preprocess image.
    
    Returns:
        Tuple of (preprocessed tensor, original image)
    """
    original = Image.open(image_path).convert('RGB')
    original_np = np.array(original)
    
    transformed = transform(image=original_np)
    tensor = transformed['image'].unsqueeze(0)
    
    return tensor, original_np


def denormalize(image: torch.Tensor) -> np.ndarray:
    """Denormalize image for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    image = image.cpu() * std + mean
    image = torch.clamp(image, 0, 1)
    image = image.permute(1, 2, 0).numpy()
    
    return image


def create_overlay(image: np.ndarray, mask: np.ndarray, 
                   color=(1, 0, 0), alpha: float = 0.4) -> np.ndarray:
    """
    Create overlay of prediction mask on original image.
    
    Args:
        image: Original image (H, W, 3) in range [0, 1]
        mask: Binary mask (H, W) in range [0, 1]
        color: RGB color for the mask
        alpha: Transparency of the overlay
    
    Returns:
        Overlay image (H, W, 3)
    """
    # Resize mask to match image if needed
    if mask.shape[:2] != image.shape[:2]:
        from PIL import Image as PILImage
        mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((image.shape[1], image.shape[0]), PILImage.NEAREST)
        mask = np.array(mask_pil) / 255.0
    
    overlay = image.copy()
    mask_colored = np.zeros_like(image)
    for i, c in enumerate(color):
        mask_colored[:, :, i] = mask * c
    
    overlay = np.where(
        mask[:, :, np.newaxis] > 0.5,
        (1 - alpha) * overlay + alpha * mask_colored,
        overlay
    )
    
    return np.clip(overlay, 0, 1)


@torch.no_grad()
def predict(model, image_tensor, device, threshold=0.5):
    """
    Run inference on a single image.
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor (1, 3, H, W)
        device: Device to run on
        threshold: Threshold for binarization
    
    Returns:
        Tuple of (probability map, binary mask)
    """
    image_tensor = image_tensor.to(device)
    output = model(image_tensor)
    prob = torch.sigmoid(output)
    mask = (prob > threshold).float()
    
    return prob[0, 0].cpu().numpy(), mask[0, 0].cpu().numpy()


def visualize_result(
    original: np.ndarray,
    prob_map: np.ndarray,
    binary_mask: np.ndarray,
    save_path: str = None,
    show: bool = True
):
    """
    Visualize inference result.
    
    Args:
        original: Original image
        prob_map: Probability map from model
        binary_mask: Binary segmentation mask
        save_path: Path to save visualization
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Probability map
    axes[1].imshow(prob_map, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Probability Map')
    axes[1].axis('off')
    
    # Binary mask
    axes[2].imshow(binary_mask, cmap='gray')
    axes[2].set_title('Segmentation Mask')
    axes[2].axis('off')
    
    # Overlay
    # Normalize original for overlay
    orig_normalized = original.astype(np.float32) / 255.0
    
    # Resize mask to original size
    from PIL import Image as PILImage
    mask_pil = PILImage.fromarray((binary_mask * 255).astype(np.uint8))
    mask_resized = mask_pil.resize((original.shape[1], original.shape[0]), PILImage.NEAREST)
    mask_resized = np.array(mask_resized) / 255.0
    
    overlay = create_overlay(orig_normalized, mask_resized, color=(1, 0.3, 0))
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def process_directory(
    model,
    input_dir: str,
    output_dir: str,
    transform,
    device,
    threshold: float = 0.5
):
    """Process all images in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in valid_extensions]
    
    print(f"Found {len(image_files)} images to process")
    
    for image_file in image_files:
        print(f"Processing: {image_file.name}")
        
        # Load and predict
        tensor, original = load_image(str(image_file), transform)
        prob_map, binary_mask = predict(model, tensor, device, threshold)
        
        # Save visualization
        save_path = output_path / f"{image_file.stem}_result.png"
        visualize_result(original, prob_map, binary_mask, 
                        save_path=str(save_path), show=False)
        
        # Save mask
        mask_path = output_path / f"{image_file.stem}_mask.png"
        mask_image = Image.fromarray((binary_mask * 255).astype(np.uint8))
        mask_image.save(mask_path)
    
    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Inference with U-Net Road Segmentation')
    
    parser.add_argument('--model', type=str, default='baseline',
                       choices=['baseline', 'attention'],
                       help='Model type')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image path or directory')
    parser.add_argument('--output', type=str, default='outputs/inference',
                       help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary mask')
    parser.add_argument('--image-size', type=int, default=256,
                       help='Input image size for model')
    parser.add_argument('--show', action='store_true',
                       help='Display results')
    
    args = parser.parse_args()
    
    # Setup
    config = get_config(model_type=args.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading {args.model} model from {args.checkpoint}...")
    model = get_model(
        model_type=args.model,
        in_channels=3,
        num_classes=1,
        encoder_channels=config.model.encoder_channels,
        bottleneck_channels=config.model.bottleneck_channels,
        use_batch_norm=config.model.use_batch_norm,
        dropout_rate=0
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Get transform
    transform = get_inference_transform((args.image_size, args.image_size))
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        print(f"\nProcessing single image: {args.input}")
        tensor, original = load_image(args.input, transform)
        prob_map, binary_mask = predict(model, tensor, device, args.threshold)
        
        os.makedirs(args.output, exist_ok=True)
        save_path = os.path.join(args.output, f"{input_path.stem}_result.png")
        visualize_result(original, prob_map, binary_mask, 
                        save_path=save_path, show=args.show)
        
    elif input_path.is_dir():
        # Directory of images
        print(f"\nProcessing directory: {args.input}")
        process_directory(model, args.input, args.output, transform, 
                         device, args.threshold)
    else:
        print(f"Error: Input path not found: {args.input}")


if __name__ == '__main__':
    main()
