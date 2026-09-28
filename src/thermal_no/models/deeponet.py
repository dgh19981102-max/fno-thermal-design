"""DeepONet baseline (Lu et al., Nature Machine Intelligence 2021).

    G(u)(y) = sum_{i=1..p} b_i(u) * t_i(y) + b_0

    branch net b(u): encodes the input functions (k, f) sampled on a fixed grid
                     -> p coefficients. Here a small strided CNN.
    trunk net  t(y): encodes a query location y = (x, y) -> p basis functions.

The trunk can be queried at any location, so the *output* can be produced on any
grid. The branch, however, needs its inputs on the grid it was trained on
(branch_res). For a 128x128 input we block-average it down to 64x64 first, so
the "super-resolution" test only refines the output, not the input.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepONet(nn.Module):
    def __init__(self, in_ch: int = 2, out_ch: int = 1, p: int = 128, width: int = 128,
                 branch_res: int = 64):
        super().__init__()
        assert out_ch == 1
        self.branch_res = branch_res
        chs = [in_ch, 32, 64, 128, 128]
        layers = []
        for a, b in zip(chs[:-1], chs[1:]):                       # 64 -> 32 -> 16 -> 8 -> 4
            layers += [nn.Conv2d(a, b, 3, stride=2, padding=1), nn.GELU()]
        self.branch_cnn = nn.Sequential(*layers)
        feat = chs[-1] * (branch_res // 16) ** 2
        self.branch_fc = nn.Sequential(nn.Linear(feat, width), nn.GELU(), nn.Linear(width, p))
        self.trunk = nn.Sequential(
            nn.Linear(2, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, p), nn.GELU(),
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        u = x
        if (H, W) != (self.branch_res, self.branch_res):
            u = F.adaptive_avg_pool2d(x, self.branch_res)
        b = self.branch_fc(self.branch_cnn(u).flatten(1))           # (B, p)
        ys = (torch.arange(H, device=x.device, dtype=x.dtype) + 0.5) / H
        xs = (torch.arange(W, device=x.device, dtype=x.dtype) + 0.5) / W
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        t = self.trunk(torch.stack([gx, gy], -1).reshape(-1, 2))    # (H*W, p)
        out = b @ t.T + self.bias                                   # (B, H*W)
        return out.reshape(B, 1, H, W)
