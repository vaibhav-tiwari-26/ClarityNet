"""
ClarityNet model architectures — Stage 1 (Poisson-Gaussian denoiser)
and Stage 2 (sub-pixel CNN super-resolution).

Kept self-contained in this file so `run.py` has a single import
source for both network definitions.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Stage 1: Physics-informed denoising (U-Net, residual + attention)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + residual)


class AttentionGate(nn.Module):
    """Simple channel attention gate applied at skip connections."""

    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class UNetDenoiser(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        c = base_channels

        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, c, 3, padding=1), ResidualBlock(c))
        self.down1 = nn.Conv2d(c, c * 2, 4, stride=2, padding=1)

        self.enc2 = nn.Sequential(ResidualBlock(c * 2))
        self.down2 = nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1)

        self.bottleneck = nn.Sequential(ResidualBlock(c * 4), ResidualBlock(c * 4))

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1)
        self.attn2 = AttentionGate(c * 2)
        self.dec2 = ResidualBlock(c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1)
        self.attn1 = AttentionGate(c)
        self.dec1 = ResidualBlock(c)

        self.out_conv = nn.Conv2d(c, in_channels, 3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))

        d2 = self.up2(b) + self.attn2(e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2) + self.attn1(e1)
        d1 = self.dec1(d1)

        return x + self.out_conv(d1)


# ---------------------------------------------------------------------------
# Stage 2: Detail-preserving super-resolution (ESPCN-style, sub-pixel conv)
# ---------------------------------------------------------------------------

class ESPCN(nn.Module):
    def __init__(self, in_channels=1, upscale_factor=2, base_channels=64):
        super().__init__()
        self.upscale_factor = upscale_factor

        self.feature_extract = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.sub_pixel_conv = nn.Conv2d(
            base_channels // 2,
            in_channels * (upscale_factor ** 2),
            3,
            padding=1,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.feature_extract(x)
        x = self.sub_pixel_conv(x)
        return self.pixel_shuffle(x)


def build_pipeline(device="cpu", upscale_factor=2):
    """Instantiate both stages. Weight loading happens in run.py."""
    denoiser = UNetDenoiser(in_channels=1, base_channels=32).to(device)
    sr = ESPCN(in_channels=1, upscale_factor=upscale_factor, base_channels=64).to(device)
    return denoiser, sr
