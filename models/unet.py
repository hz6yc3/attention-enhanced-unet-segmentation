"""
Baseline U-Net
==============

Encoder-decoder with skip connections, transposed-convolution upsampling and
(Conv -> BN -> ReLU) x 2 blocks. This is the definition used for the reported
experiments (ported from the Colab notebook); the default configuration has
31.04M trainable parameters.

Reference:
    Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image
    Segmentation", MICCAI 2015.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2 with optional spatial dropout in between."""

    def __init__(self, in_channels: int, out_channels: int,
                 use_batch_norm: bool = True, dropout_rate: float = 0.0):
        super().__init__()
        norm = (lambda c: nn.BatchNorm2d(c)) if use_batch_norm else (lambda c: nn.Identity())
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=not use_batch_norm),
            norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=not use_batch_norm),
            norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net.

    Args:
        in_channels: input channels (3 for RGB)
        num_classes: output channels (1 for binary segmentation, raw logits)
        encoder_channels: channels at each encoder level
        bottleneck_channels: channels at the bottleneck
        use_batch_norm: batch normalisation in every DoubleConv
        dropout_rate: Dropout2d rate inside DoubleConv blocks (not in the stem)
    """

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
        self.decoders = nn.ModuleList()
        self.upconvs.append(nn.ConvTranspose2d(bottleneck_channels, decoder_channels[0], 2, stride=2))
        self.decoders.append(DoubleConv(decoder_channels[0] * 2, decoder_channels[0], use_batch_norm, dropout_rate))
        for i in range(len(decoder_channels) - 1):
            self.upconvs.append(nn.ConvTranspose2d(decoder_channels[i], decoder_channels[i + 1], 2, stride=2))
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
        for i, (up, dec) in enumerate(zip(self.upconvs, self.decoders)):
            x = up(x)
            if x.shape[2:] != feats[i].shape[2:]:
                x = F.interpolate(x, size=feats[i].shape[2:])
            x = dec(torch.cat([feats[i], x], dim=1))
        return self.outc(x)

    def get_num_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def count_parameters(self) -> int:
        return self.get_num_parameters()[1]


if __name__ == "__main__":
    m = UNet()
    print(f"UNet trainable parameters: {m.get_num_parameters()[1]:,}")
    print(m(torch.zeros(1, 3, 256, 256)).shape)
