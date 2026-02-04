"""
Attention U-Net Model
======================

Implementation of U-Net with Attention Gates on skip connections.

The Attention U-Net extends the standard U-Net by adding attention gates
that learn to focus on relevant spatial regions. This allows the model
to suppress irrelevant features and highlight salient features.

Key Enhancement:
    Before concatenating encoder features with decoder features in each
    skip connection, an attention gate filters the encoder features to
    focus on relevant regions.

Reference Papers:
1. "Attention U-Net: Learning Where to Look for the Pancreas"
   Oktay et al., MIDL 2018
   
2. "A Comprehensive Review of U-Net and Its Variants"
   https://arxiv.org/abs/2502.06895
   (Discusses attention mechanisms as one of four key enhancement mechanisms)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DoubleConv(nn.Module):
    """
    Double Convolution Block: (Conv2d -> BatchNorm -> ReLU) x 2
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0
    ):
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


class AttentionGate(nn.Module):
    """
    Attention Gate Module
    
    The attention gate learns to focus on relevant spatial regions by
    computing attention coefficients that weight the encoder features.
    
    Architecture:
    -------------
    g (gating signal from decoder) ──────┐
                                          │
    x (encoder features) ────────────────┼──→ [1x1 Conv] ──→ ReLU ──→ [1x1 Conv] ──→ Sigmoid ──→ α
                                          │                                                      │
                                          └──→ [Interpolate if needed] ────────────────────────┘
                                                                                                 │
    x * α ──────────────────────────────────────────────────────────────────────────────────────┘
    
    The attention coefficient α ∈ [0, 1] indicates the importance of each
    spatial location in the encoder features.
    """
    
    def __init__(
        self,
        gate_channels: int,      # Channels from gating signal (decoder)
        feature_channels: int,   # Channels from encoder features
        inter_channels: int = None  # Intermediate channels
    ):
        """
        Args:
            gate_channels: Number of channels in gating signal (from decoder)
            feature_channels: Number of channels in encoder features
            inter_channels: Number of intermediate channels (defaults to feature_channels // 2)
        """
        super().__init__()
        
        if inter_channels is None:
            inter_channels = feature_channels // 2
            if inter_channels == 0:
                inter_channels = 1
        
        # Transform gating signal to intermediate channels
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        # Transform encoder features to intermediate channels
        self.W_x = nn.Sequential(
            nn.Conv2d(feature_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        # Compute attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Apply attention to encoder features.
        
        Args:
            x: Encoder features (B, C_x, H_x, W_x)
            g: Gating signal from decoder (B, C_g, H_g, W_g)
        
        Returns:
            Attention-weighted encoder features (B, C_x, H_x, W_x)
        """
        # Upsample gating signal to match encoder feature size
        g_upsampled = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Transform both inputs
        g_transformed = self.W_g(g_upsampled)
        x_transformed = self.W_x(x)
        
        # Combine and compute attention
        combined = self.relu(g_transformed + x_transformed)
        attention = self.psi(combined)
        
        # Apply attention to encoder features
        return x * attention


class EncoderBlock(nn.Module):
    """Encoder Block: MaxPool -> DoubleConv"""
    
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


class AttentionDecoderBlock(nn.Module):
    """
    Decoder Block with Attention Gate
    
    Upsample -> Attention Gate on Skip Connection -> Concatenate -> DoubleConv
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        bilinear: bool = True
    ):
        """
        Args:
            in_channels: Channels from previous decoder level
            skip_channels: Channels from corresponding encoder level (skip connection)
            out_channels: Output channels
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout probability
            bilinear: If True, use bilinear upsampling
        """
        super().__init__()
        
        # Upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            up_channels = in_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            up_channels = in_channels // 2
        
        # Attention gate
        self.attention = AttentionGate(
            gate_channels=up_channels,
            feature_channels=skip_channels,
            inter_channels=skip_channels // 2
        )
        
        # Double convolution after concatenation
        self.conv = DoubleConv(
            up_channels + skip_channels, 
            out_channels,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate
        )
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Feature map from previous decoder level (gating signal)
            x2: Skip connection from corresponding encoder level
        
        Returns:
            Decoded feature map
        """
        # Upsample
        x1 = self.up(x1)
        
        # Apply attention gate to encoder features
        x2_attended = self.attention(x2, x1)
        
        # Handle size mismatch
        diff_y = x2_attended.size()[2] - x1.size()[2]
        diff_x = x2_attended.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate attended encoder features with decoder features
        x = torch.cat([x2_attended, x1], dim=1)
        
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net for Binary Segmentation
    
    This model extends the standard U-Net by incorporating attention gates
    into the skip connections. The attention mechanism allows the model to
    learn which spatial regions are important for the segmentation task.
    
    Key Difference from Standard U-Net:
    ------------------------------------
    In standard U-Net, encoder features are directly concatenated with
    decoder features. In Attention U-Net, encoder features first pass
    through an attention gate that is conditioned on the decoder features
    (gating signal). This filters out irrelevant information.
    
    Benefits:
    ---------
    1. Better focus on relevant anatomical/object regions
    2. Suppression of noisy or irrelevant background features
    3. Improved segmentation accuracy, especially for small objects
    4. Interpretability through visualization of attention maps
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
            num_classes: Number of output classes (1 for binary)
            encoder_channels: Channel sizes for each encoder level
            bottleneck_channels: Channels in bottleneck
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
        
        # Decoder path with attention gates
        decoder_in_channels = [bottleneck_channels] + encoder_channels[::-1][:-1]
        decoder_skip_channels = encoder_channels[::-1]
        decoder_out_channels = encoder_channels[::-1]
        
        self.decoders = nn.ModuleList()
        for i in range(len(encoder_channels)):
            out_ch = decoder_out_channels[i] // 2 if i < len(encoder_channels) - 1 else decoder_out_channels[i]
            self.decoders.append(
                AttentionDecoderBlock(
                    in_channels=decoder_in_channels[i],
                    skip_channels=decoder_skip_channels[i],
                    out_channels=out_ch if bilinear else decoder_out_channels[i],
                    use_batch_norm=use_batch_norm,
                    dropout_rate=dropout_rate,
                    bilinear=bilinear
                )
            )
        
        # Output convolution
        final_channels = encoder_channels[0] // 2 if bilinear else encoder_channels[0]
        self.outc = nn.Conv2d(final_channels, num_classes, kernel_size=1)
        
        # Store attention maps for visualization
        self.attention_maps = []
        
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
        Forward pass through Attention U-Net.
        
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
    print("Testing Attention U-Net model...")
    
    # Create model
    model = AttentionUNet(
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
    print("\nAttention U-Net test passed!")
    
    # Compare with baseline U-Net
    from unet import UNet
    baseline = UNet(
        in_channels=3,
        num_classes=1,
        encoder_channels=[64, 128, 256, 512],
        bottleneck_channels=1024
    )
    baseline_params, _ = baseline.get_num_parameters()
    
    print(f"\nParameter Comparison:")
    print(f"  Baseline U-Net: {baseline_params:,}")
    print(f"  Attention U-Net: {total_params:,}")
    print(f"  Attention overhead: {total_params - baseline_params:,} ({100*(total_params-baseline_params)/baseline_params:.1f}%)")
