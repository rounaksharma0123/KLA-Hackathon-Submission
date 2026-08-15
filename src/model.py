import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallResidualUNet(nn.Module):
    """Small grayscale U-Net that predicts a residual correction over bicubic input."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32, residual_scale: float = 0.2):
        super().__init__()
        self.residual_scale = residual_scale

        self.enc1 = ConvBlock(in_channels, base_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.enc2 = ConvBlock(base_channels * 2, base_channels * 2)

        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4)

        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.out = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.enc1(x)
        skip2 = self.enc2(F.relu(self.down1(skip1), inplace=True))
        hidden = self.bottleneck(F.relu(self.down2(skip2), inplace=True))

        hidden = F.interpolate(hidden, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.dec2(torch.cat([hidden, skip2], dim=1))
        hidden = F.interpolate(hidden, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.dec1(torch.cat([hidden, skip1], dim=1))

        residual = self.out(hidden)
        return torch.clamp(x + self.residual_scale * residual, 0.0, 1.0)


def build_model(base_channels: int = 32) -> nn.Module:
    return SmallResidualUNet(base_channels=base_channels)
