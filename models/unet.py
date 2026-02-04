"""
Baseline U-Net Model
====================

Implementation of the standard U-Net architecture for binary segmentation.

The U-Net architecture consists of:
1. Encoder (Contracting Path): Captures context through downsampling
2. Bottleneck: Highest level of abstraction
3. Decoder (Expansive Path): Enables precise localization through upsampling
4. Skip Connections: Connect encoder and decoder at each level

Original Paper:
"U-Net: Convolutional Networks for Biomedical Image Segmentation"
Ronneberger et al., MICCAI 2015

Reference for this implementation:
"A Comprehensive Review of U-Net and Its Variants"
https://arxiv.org/abs/2502.06895
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DoubleConv(nn.Module):
    """
    Double Convolution Block: (Conv2d -> BatchNorm -> ReLU) x 2
    
    This is the fundamental building block used in both encoder and decoder.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            mid_channels: Number of intermediate channels (defaults to out_channels)
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout probability (0 to disable)
        """
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
        
        layers = []
        
        # First convolution block
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=not use_batch_norm))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(mid_channels))
        layers.append(nn.ReLU(inplace=True))
        
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        
        # Second convolution block
        layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=not use_batch_norm))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        self.double_conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    """
    Encoder Block: MaxPool -> DoubleConv
    
    Downsamples the feature map and increases the number of channels.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_batch_norm=use_batch_norm, dropout_rate=dropout_rate)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class DecoderBlock(nn.Module):
    """
    Decoder Block: Upsample -> Concatenate -> DoubleConv
    
    Upsamples the feature map and combines with skip connection from encoder.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        bilinear: bool = True
    ):
        """
        Args:
            in_channels: Number of input channels (from previous decoder level)
            out_channels: Number of output channels
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout probability
            bilinear: If True, use bilinear upsampling; else use transposed conv
        """
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(
                in_channels, 
                out_channels, 
                mid_channels=in_channels // 2,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(
                in_channels, 
                out_channels,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate
            )
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Feature map from previous decoder level
            x2: Skip connection from corresponding encoder level
        
        Returns:
            Decoded feature map
        """
        x1 = self.up(x1)
        
        # Handle size mismatch due to pooling
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net Architecture for Binary Segmentation.
    
    Architecture Overview:
    ----------------------
    Input (3 x H x W)
        │
        ▼
    [Encoder 1] → 64 channels  ──────────────────────┐
        │                                             │
        ▼                                             │
    [Encoder 2] → 128 channels ─────────────────┐    │
        │                                        │    │
        ▼                                        │    │
    [Encoder 3] → 256 channels ────────────┐    │    │
        │                                   │    │    │
        ▼                                   │    │    │
    [Encoder 4] → 512 channels ───────┐    │    │    │
        │                              │    │    │    │
        ▼                              │    │    │    │
    [Bottleneck] → 1024 channels      │    │    │    │
        │                              │    │    │    │
        ▼                              ▼    │    │    │
    [Decoder 4] ← concat ─────────────┘    │    │    │
        │                                   │    │    │
        ▼                                   ▼    │    │
    [Decoder 3] ← concat ──────────────────┘    │    │
        │                                        │    │
        ▼                                        ▼    │
    [Decoder 2] ← concat ───────────────────────┘    │
        │                                             │
        ▼                                             ▼
    [Decoder 1] ← concat ────────────────────────────┘
        │
        ▼
    [Output Conv] → 1 channel
        │
        ▼
    Output (1 x H x W)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_channels: List[int] = [64, 128, 256, 512],
        bottleneck_channels: int = 1024,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1,
        bilinear: bool = True
    ):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes (1 for binary segmentation)
            encoder_channels: List of channel sizes for each encoder level
            bottleneck_channels: Number of channels in bottleneck
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout probability
            bilinear: If True, use bilinear upsampling
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.encoder_channels = encoder_channels
        
        # Initial convolution
        self.inc = DoubleConv(in_channels, encoder_channels[0], use_batch_norm=use_batch_norm)
        
        # Encoder path
        self.encoders = nn.ModuleList()
        for i in range(len(encoder_channels) - 1):
            self.encoders.append(
                EncoderBlock(
                    encoder_channels[i], 
                    encoder_channels[i + 1],
                    use_batch_norm=use_batch_norm,
                    dropout_rate=dropout_rate
                )
            )
        
        # Bottleneck
        self.bottleneck = EncoderBlock(
            encoder_channels[-1], 
            bottleneck_channels,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate
        )
        
        # Decoder path
        decoder_channels = encoder_channels[::-1]  # Reverse order
        self.decoders = nn.ModuleList()
        
        # First decoder takes bottleneck output
        factor = 2 if bilinear else 1
        self.decoders.append(
            DecoderBlock(
                bottleneck_channels,
                decoder_channels[0] // factor,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate,
                bilinear=bilinear
            )
        )
        
        # Remaining decoders
        for i in range(len(decoder_channels) - 1):
            self.decoders.append(
                DecoderBlock(
                    decoder_channels[i],
                    decoder_channels[i + 1] // factor if i < len(decoder_channels) - 2 else decoder_channels[i + 1],
                    use_batch_norm=use_batch_norm,
                    dropout_rate=dropout_rate,
                    bilinear=bilinear
                )
            )
        
        # Output convolution
        self.outc = nn.Conv2d(encoder_channels[0], num_classes, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Output tensor of shape (B, num_classes, H, W)
        """
        # Store encoder outputs for skip connections
        encoder_features = []
        
        # Initial convolution
        x = self.inc(x)
        encoder_features.append(x)
        
        # Encoder path
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path (reverse order of encoder features)
        encoder_features = encoder_features[::-1]
        
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, encoder_features[i])
        
        # Output convolution
        logits = self.outc(x)
        
        return logits
    
    def get_num_parameters(self) -> Tuple[int, int]:
        """Get number of total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


if __name__ == "__main__":
    # Test the model
    print("Testing U-Net model...")
    
    # Create model
    model = UNet(
        in_channels=3,
        num_classes=1,
        encoder_channels=[64, 128, 256, 512],
        bottleneck_channels=1024,
        use_batch_norm=True,
        dropout_rate=0.1
    )
    
    # Print model summary
    total_params, trainable_params = model.get_num_parameters()
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Verify output is same spatial size as input
    assert output.shape[2:] == x.shape[2:], "Output spatial size mismatch!"
    print("\nU-Net test passed!")
