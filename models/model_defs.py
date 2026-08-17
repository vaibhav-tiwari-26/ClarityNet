import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        hidden = max(ch // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, attention=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.attn = SEBlock(out_ch) if attention else nn.Identity()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return self.attn(x)


class UNetDenoiser(nn.Module):
    """Stage 1: residual U-Net with SE attention for Poisson-Gaussian denoising."""

    def __init__(self, in_ch=1, base=64):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.mid = ConvBlock(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4, attention=True)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2, attention=True)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base, attention=True)
        self.out = nn.Conv2d(base, in_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m = self.mid(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return x - self.out(d1)


class ESPCNSR(nn.Module):
    """Stage 2: ESPCN-style sub-pixel convolution super-resolution."""

    def __init__(self, in_ch=1, upscale=2, base=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, base, 5, padding=2)
        self.conv2 = nn.Conv2d(base, base * 2, 3, padding=1)
        self.conv3 = nn.Conv2d(base * 2, base * 2, 3, padding=1)
        self.conv4 = nn.Conv2d(base * 2, in_ch * upscale * upscale, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = self.pixel_shuffle(self.conv4(x))
        return x


def build_stage1(in_ch=1):
    return UNetDenoiser(in_ch=in_ch)


def build_stage2(in_ch=1, upscale=2):
    return ESPCNSR(in_ch=in_ch, upscale=upscale)
