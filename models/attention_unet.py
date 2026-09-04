"""
Attention U-Net
===============

Baseline U-Net with an additive attention gate on every skip connection. The
gate uses the decoder feature at the same level as the gating signal and
re-weights the encoder feature before concatenation. Default configuration has
31.39M trainable parameters (about 1.1% more than the baseline).

Reference:
    Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas",
    MIDL 2018.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import DoubleConv


class AttentionGate(nn.Module):
    """Additive attention gate: alpha = sigmoid(psi(ReLU(W_g g + W_x x)))."""

    def __init__(self, gate_channels: int, feature_channels: int, inter_channels: int = None):
        super().__init__()
        inter_channels = inter_channels or max(1, feature_channels // 2)
        self.W_g = nn.Sequential(nn.Conv2d(gate_channels, inter_channels, 1, bias=True),
                                 nn.BatchNorm2d(inter_channels))
        self.W_x = nn.Sequential(nn.Conv2d(feature_channels, inter_channels, 1, bias=True),
                                 nn.BatchNorm2d(inter_channels))
        self.psi = nn.Sequential(nn.Conv2d(inter_channels, 1, 1, bias=True),
                                 nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
        self.last_attention = None  # (B, 1, H, W) coefficients from the latest forward pass

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        alpha = self.psi(self.relu(self.W_g(g) + self.W_x(x)))
        self.last_attention = alpha.detach()
        return x * alpha


class AttentionUNet(nn.Module):
    """U-Net with attention gates on all skip connections (same signature as UNet)."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_channels: List[int] = (64, 128, 256, 512),
        bottleneck_channels: int = 1024,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.1,
        **_unused,
    ):
        super().__init__()
        encoder_channels = list(encoder_channels)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.encoder_channels = encoder_channels

        self.inc = DoubleConv(in_channels, encoder_channels[0], use_batch_norm)
        self.encoders = nn.ModuleList()
        for i in range(len(encoder_channels) - 1):
            self.encoders.append(nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(encoder_channels[i], encoder_channels[i + 1], use_batch_norm, dropout_rate),
            ))
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(encoder_channels[-1], bottleneck_channels, use_batch_norm, dropout_rate),
        )

        decoder_channels = encoder_channels[::-1]
        self.upconvs = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs.append(nn.ConvTranspose2d(bottleneck_channels, decoder_channels[0], 2, stride=2))
        self.attention_gates.append(AttentionGate(decoder_channels[0], decoder_channels[0]))
        self.decoders.append(DoubleConv(decoder_channels[0] * 2, decoder_channels[0], use_batch_norm, dropout_rate))
        for i in range(len(decoder_channels) - 1):
            self.upconvs.append(nn.ConvTranspose2d(decoder_channels[i], decoder_channels[i + 1], 2, stride=2))
            self.attention_gates.append(AttentionGate(decoder_channels[i + 1], decoder_channels[i + 1]))
            self.decoders.append(DoubleConv(decoder_channels[i + 1] * 2, decoder_channels[i + 1],
                                            use_batch_norm, dropout_rate))

        self.outc = nn.Conv2d(encoder_channels[0], num_classes, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [self.inc(x)]
        for enc in self.encoders:
            feats.append(enc(feats[-1]))
        x = self.bottleneck(feats[-1])
        feats = feats[::-1]
        for i, (up, gate, dec) in enumerate(zip(self.upconvs, self.attention_gates, self.decoders)):
            x = up(x)
            skip = gate(feats[i], x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:])
            x = dec(torch.cat([skip, x], dim=1))
        return self.outc(x)

    def get_attention_maps(self) -> List[torch.Tensor]:
        """Attention coefficients from the most recent forward pass, coarsest level first."""
        return [g.last_attention for g in self.attention_gates]

    def get_num_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def count_parameters(self) -> int:
        return self.get_num_parameters()[1]


if __name__ == "__main__":
    m = AttentionUNet()
    print(f"AttentionUNet trainable parameters: {m.get_num_parameters()[1]:,}")
    print(m(torch.zeros(1, 3, 256, 256)).shape)
