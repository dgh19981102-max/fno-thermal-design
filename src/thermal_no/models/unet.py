"""U-Net baseline (Ronneberger et al., 2015), a standard CNN surrogate for fields.

Coordinates are appended as extra input channels (like the FNO) because the
problem is not translation invariant (the heat sink sits at a fixed place).

The network is fully convolutional, so it *runs* on a 128x128 grid, but its
3x3 kernels then span half the physical distance they were trained on, so
there is no reason to expect it to be accurate there. We report it anyway.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fno import grid_channels


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
    )


class UNet(nn.Module):
    def __init__(self, in_ch: int = 2, out_ch: int = 1, base: int = 32, depth: int = 3):
        super().__init__()
        chs = [base * 2**i for i in range(depth + 1)]           # e.g. 32, 64, 128, 256
        self.enc = nn.ModuleList()
        c = in_ch + 2
        for ch in chs[:-1]:
            self.enc.append(conv_block(c, ch))
            c = ch
        self.bottleneck = conv_block(c, chs[-1])
        self.dec = nn.ModuleList()
        c = chs[-1]
        for ch in reversed(chs[:-1]):
            self.dec.append(conv_block(c + ch, ch))
            c = ch
        self.head = nn.Conv2d(c, out_ch, 1)
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        if H % 2**self.depth or W % 2**self.depth:
            raise ValueError(f"grid must be divisible by {2**self.depth}")
        x = torch.cat([x, grid_channels(x)], dim=1)
        skips = []
        for block in self.enc:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for block, skip in zip(self.dec, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)
