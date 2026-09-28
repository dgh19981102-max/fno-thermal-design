"""Density-based layout optimisation of high-conductivity material.

Design problem
    minimise    max_x T(x)                  (approximated by a p-norm, see below)
    over        rho(x) in [0, 1]            (fraction of high-k material per cell)
    subject to  mean(rho) = V               (limited amount of expensive material)
    with        k = k_min + (k_max - k_min) * rho^p     (SIMP interpolation)

Parameterisation (identical for both optimisers, so the comparison is fair)
    z (free, N x N)  --filter-->  z_f  --sigmoid(z_f + mu)-->  rho  --SIMP-->  k
    * the cone filter smooths the design and suppresses checkerboards;
    * mu is chosen by bisection so that mean(rho) = V exactly; its gradient
      d mu / d z is added through one differentiable Newton step (implicit
      function theorem), so the volume constraint never needs a penalty.

Objective
    max(T) is not differentiable, so we minimise the p-norm
        J = ( mean_x T(x)^P )^(1/P),  which tends to max T as P -> inf.
    All reported numbers use the true max(T) from the reference solver.

Gradients
    * solver:    adjoint method. For A(k) T = b and J(T):
                     A^T lambda = dJ/dT,     dJ/dk = -lambda^T (dA/dk) T.
                 A is symmetric, so the LU factors from the forward solve are
                 reused and the gradient costs one extra triangular solve.
                 The term lambda^T (dA/dk) T is obtained by autograd through the
                 differentiable residual in physics.py.
    * surrogate: plain autograd through the neural network.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from . import solver
from .physics import fv_residual


# ----------------------------------------------------------------- solver as a torch op
class SolveFV(torch.autograd.Function):
    """T = A(k)^{-1} b(f) for a single (N, N) problem, with an adjoint backward."""

    @staticmethod
    def forward(ctx, k: torch.Tensor, f: torch.Tensor, sink_width: float):
        k_np = k.detach().cpu().double().numpy()
        T_np, lu = solver.solve(k_np, f.detach().cpu().double().numpy(), sink_width, return_factor=True)
        T = torch.from_numpy(T_np).to(k)
        ctx.lu, ctx.sink_width = lu, sink_width
        ctx.save_for_backward(k, f, T)
        return T

    @staticmethod
    def backward(ctx, grad_T: torch.Tensor):
        k, f, T = ctx.saved_tensors
        n = k.shape[-1]
        lam = ctx.lu.solve(grad_T.detach().cpu().double().numpy().ravel()).reshape(n, n)   # A^T lam = dJ/dT
        lam = torch.from_numpy(lam).to(k)
        with torch.enable_grad():
            kk = k.detach().double().requires_grad_(True)
            r = fv_residual(T.detach().double(), kk, f.detach().double(), ctx.sink_width)
            (g_k,) = torch.autograd.grad(-(lam.double() * r).sum(), kk)
        return g_k.to(k), None, None


def solver_T(k: torch.Tensor, f: torch.Tensor, sink_width: float = solver.DEFAULT_SINK_WIDTH) -> torch.Tensor:
    """Batched wrapper: k, f of shape (S, N, N) -> T (S, N, N), differentiable in k."""
    return torch.stack([SolveFV.apply(k[i], f[i], sink_width) for i in range(k.shape[0])])


# ----------------------------------------------------------------- design parameterisation
def cone_kernel(radius: float, dtype=torch.float32) -> torch.Tensor:
    r = int(np.ceil(radius))
    ax = torch.arange(-r, r + 1, dtype=dtype)
    Y, X = torch.meshgrid(ax, ax, indexing="ij")
    w = (radius - torch.sqrt(X**2 + Y**2)).clamp_min(0)
    return (w / w.sum()).reshape(1, 1, *w.shape)


def density_filter(z: torch.Tensor, radius: float) -> torch.Tensor:
    """Cone filter with replicate padding. z: (S, N, N)."""
    if radius <= 1:
        return z
    ker = cone_kernel(radius, z.dtype).to(z.device)
    p = ker.shape[-1] // 2
    return F.conv2d(F.pad(z.unsqueeze(1), (p, p, p, p), mode="replicate"), ker).squeeze(1)


def volume_shift(zf: torch.Tensor, volume: float, iters: int = 60) -> torch.Tensor:
    """mu (S,) with mean(sigmoid(zf + mu)) = volume; differentiable w.r.t. zf."""
    with torch.no_grad():
        lo = torch.full((zf.shape[0],), -30.0, dtype=zf.dtype, device=zf.device)
        hi = -lo
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            too_big = torch.sigmoid(zf + mid[:, None, None]).mean((1, 2)) > volume
            hi = torch.where(too_big, mid, hi)
            lo = torch.where(too_big, lo, mid)
        mu0 = 0.5 * (lo + hi)
    # One Newton step from the (converged) root: the value stays mu0 but autograd
    # now sees d mu / d zf = -sigma'(zf + mu) / sum(sigma'), i.e. the implicit gradient.
    s = torch.sigmoid(zf + mu0[:, None, None])
    g = s.mean((1, 2)) - volume
    dg = (s * (1 - s)).mean((1, 2)).clamp_min(1e-12)
    return mu0 - g / dg


@dataclass
class DesignSpace:
    volume_fraction: float = 0.3
    k_min: float = 1.0
    k_max: float = 20.0
    simp_p: float = 3.0
    filter_radius: float = 2.0

    def density(self, z: torch.Tensor) -> torch.Tensor:
        zf = density_filter(z, self.filter_radius)
        mu = volume_shift(zf, self.volume_fraction)
        return torch.sigmoid(zf + mu[:, None, None])

    def conductivity(self, rho: torch.Tensor) -> torch.Tensor:
        return self.k_min + (self.k_max - self.k_min) * rho**self.simp_p


def pnorm(T: torch.Tensor, P: float) -> torch.Tensor:
    """(mean T^P)^(1/P) per sample, computed stably."""
    Tp = T.clamp_min(0).flatten(1)
    m = Tp.max(1, keepdim=True).values.detach().clamp_min(1e-12)
    return m.squeeze(1) * ((Tp / m) ** P).mean(1) ** (1.0 / P)


# ----------------------------------------------------------------- optimisation loop
@dataclass
class OptResult:
    rho: np.ndarray                 # (S, N, N) final designs
    k: np.ndarray                   # (S, N, N)
    objective: np.ndarray           # (iters, S) objective as seen by the optimiser
    wall_time_s: float
    snapshots: list = field(default_factory=list)   # intermediate k fields (for refinement)


def optimise(T_fn, f: torch.Tensor, space: DesignSpace, z0: torch.Tensor, iters: int, lr: float,
             P: float, snapshot_every: int = 0) -> OptResult:
    """Adam on z. T_fn maps k (S, N, N) -> T (S, N, N) and must be differentiable.

    f  : (N, N) heat source, shared by all S starting designs in z0 (S, N, N).
    """
    z = z0.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    fS = f.expand_as(z0)
    hist, snaps = [], []
    t0 = time.perf_counter()
    for it in range(iters):
        rho = space.density(z)
        k = space.conductivity(rho)
        J = pnorm(T_fn(k, fS), P)
        opt.zero_grad()
        J.sum().backward()              # starts are independent, so summing is fine
        opt.step()
        hist.append(J.detach().cpu().numpy())
        if snapshot_every and it % snapshot_every == 0:
            snaps.append(k.detach().cpu().numpy())
    with torch.no_grad():
        rho = space.density(z)
        k = space.conductivity(rho)
    wall = time.perf_counter() - t0
    return OptResult(rho.cpu().numpy(), k.cpu().numpy(), np.array(hist), wall, snaps)


def initial_designs(n: int, n_starts: int, seed: int, dtype=torch.float32) -> torch.Tensor:
    """Start 0 is the uniform design; the others add smooth random perturbations."""
    g = torch.Generator().manual_seed(seed)
    z = torch.zeros(n_starts, n, n, dtype=dtype)
    if n_starts > 1:
        noise = torch.randn(n_starts - 1, n, n, generator=g, dtype=dtype)
        z[1:] = 2.0 * density_filter(noise, 4.0) / density_filter(noise, 4.0).std()
    return z


def true_max_T(k: np.ndarray, f: np.ndarray, sink_width: float = solver.DEFAULT_SINK_WIDTH) -> np.ndarray:
    """Reference-solver max temperature for each design in k (S, N, N)."""
    return np.array([solver.solve(kk, f, sink_width).max() for kk in k])
