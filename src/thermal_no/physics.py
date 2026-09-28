"""Differentiable (PyTorch) version of the finite-volume operator in solver.py.

residual(T, k, f) = A(k) T - b(f), evaluated with array shifts instead of a
sparse matrix so that it runs batched on the GPU and supports autograd with
respect to both T and k.

It is used in two places:
    * the physics-informed (PINO-style) loss: ||A(k) T_pred - b||, which uses
      exactly the same discretisation as the data generator;
    * the adjoint gradient in optimize.py: dJ/dk = -d/dk [lambda^T R(k, T)].
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .solver import DEFAULT_SINK_WIDTH, sink_mask


def _hmean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 2.0 * a * b / (a + b)


def fv_residual(
    T: torch.Tensor,
    k: torch.Tensor,
    f: torch.Tensor,
    sink_width: float = DEFAULT_SINK_WIDTH,
) -> torch.Tensor:
    """Cell-wise residual A(k) T - f h^2.

    T, k, f : tensors of shape (..., N, N) (axis -2 = y, axis -1 = x).
    Returns a tensor of the same shape.
    """
    n = T.shape[-1]
    kx = _hmean(k[..., :, :-1], k[..., :, 1:])
    ky = _hmean(k[..., :-1, :], k[..., 1:, :])
    qx = kx * (T[..., :, :-1] - T[..., :, 1:])  # flux from column j to j+1
    qy = ky * (T[..., :-1, :] - T[..., 1:, :])  # flux from row i to i+1

    out = F.pad(qx, (0, 1)) - F.pad(qx, (1, 0)) + F.pad(qy, (0, 0, 0, 1)) - F.pad(qy, (0, 0, 1, 0))

    mask = torch.as_tensor(sink_mask(n, sink_width), device=T.device)
    sink_flux = torch.zeros_like(T)
    sink_flux[..., 0, :] = torch.where(mask, 2.0 * k[..., 0, :] * T[..., 0, :], torch.zeros_like(T[..., 0, :]))
    return out + sink_flux - f / n**2


def physics_loss(
    T: torch.Tensor,
    k: torch.Tensor,
    f: torch.Tensor,
    sink_width: float = DEFAULT_SINK_WIDTH,
) -> torch.Tensor:
    """Scale-free physics loss: per-sample ||A T - b||^2 / ||b||^2, averaged.

    Normalising by ||b|| makes the loss comparable across samples whose heat
    loads differ by an order of magnitude.
    """
    r = fv_residual(T, k, f, sink_width)
    n = T.shape[-1]
    b = f / n**2
    num = r.flatten(1).pow(2).sum(1)
    den = b.flatten(1).pow(2).sum(1).clamp_min(1e-30)
    return (num / den).mean()
