"""Fourier Neural Operator (Li et al., ICLR 2021) for 2D fields.

Architecture
    input (B, C_in, N, N) + (x, y) coordinates
      -> lifting P: 1x1 conv to `width` channels
      -> L Fourier layers:  v <- sigma( K v + W v )
             K v : spectral convolution (global, resolution independent)
             W v : 1x1 conv (local, pointwise linear map)
      -> projection Q: 1x1 conv -> 128 -> 1 channel

Why it is resolution independent
    K is parameterised by weights R on the lowest `modes` Fourier frequencies,
    which are physical frequencies of the unit square, not grid-index offsets.
    On a finer grid the same R multiplies the same frequencies, so a model
    trained at 64x64 can be evaluated at 128x128 ("zero-shot super-resolution").
    Whether that is accurate is an empirical question - see experiment 03.

Non-periodic boundaries
    The FFT assumes periodic signals, but our boundary conditions are not
    periodic. As in the reference implementation we zero-pad the domain on two
    sides before the Fourier layers and crop afterwards, giving the network room
    to represent non-periodic behaviour. The padding is a fixed *fraction* of the
    grid so that it covers the same physical length at every resolution.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """Spectral convolution: FFT -> keep low modes -> multiply by learned R -> IFFT.

    For every retained frequency (kx, ky), each output channel is a learned
    complex linear combination of the input channels:
        out_hat[b, o, kx, ky] = sum_i in_hat[b, i, kx, ky] * R[i, o, kx, ky]
    rfft2 keeps only non-negative ky (real signals are Hermitian-symmetric), so
    we need two weight blocks: positive kx (top rows) and negative kx (bottom rows).
    """

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        self.w_pos = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.w_neg = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    @staticmethod
    def _mul(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # (B, I, X, Y) x (I, O, X, Y) -> (B, O, X, Y)
        return torch.einsum("bixy,ioxy->boxy", a, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        m1, m2 = self.modes1, self.modes2
        if m1 > H // 2 or m2 > W // 2 + 1:
            raise ValueError(f"grid {H}x{W} too coarse for {m1}x{m2} modes")
        x_hat = torch.fft.rfft2(x)                                   # (B, I, H, W//2+1)
        out_hat = torch.zeros(B, self.out_ch, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_hat[:, :, :m1, :m2] = self._mul(x_hat[:, :, :m1, :m2], self.w_pos)
        out_hat[:, :, -m1:, :m2] = self._mul(x_hat[:, :, -m1:, :m2], self.w_neg)
        return torch.fft.irfft2(out_hat, s=(H, W))                   # back to (B, O, H, W)


def grid_channels(x: torch.Tensor) -> torch.Tensor:
    """(B, 2, H, W) tensor of cell-centre coordinates in [0, 1] (channel 0 = x, 1 = y)."""
    B, _, H, W = x.shape
    ys = (torch.arange(H, device=x.device, dtype=x.dtype) + 0.5) / H
    xs = (torch.arange(W, device=x.device, dtype=x.dtype) + 0.5) / W
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy]).unsqueeze(0).expand(B, -1, -1, -1)


class FNO2d(nn.Module):
    def __init__(self, in_ch: int = 2, out_ch: int = 1, modes: int = 12, width: int = 32,
                 n_layers: int = 4, pad_frac: float = 0.125):
        super().__init__()
        self.pad_frac = pad_frac
        self.lift = nn.Conv2d(in_ch + 2, width, 1)
        self.spectral = nn.ModuleList(SpectralConv2d(width, width, modes, modes) for _ in range(n_layers))
        self.pointwise = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(n_layers))
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, out_ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        x = torch.cat([x, grid_channels(x)], dim=1)
        x = self.lift(x)
        ph, pw = round(H * self.pad_frac), round(W * self.pad_frac)
        x = F.pad(x, (0, pw, 0, ph))                  # pad right (x) and top (y)
        n = len(self.spectral)
        for i, (K, Wl) in enumerate(zip(self.spectral, self.pointwise)):
            x = K(x) + Wl(x)
            if i < n - 1:                             # no activation after the last layer
                x = F.gelu(x)
        x = x[..., :H, :W]                            # crop the padding
        return self.proj(x)
