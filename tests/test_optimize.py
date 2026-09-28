"""Tests for the optimisation machinery: adjoint gradient, volume constraint, p-norm."""
import numpy as np
import torch

from thermal_no import data as D
from thermal_no import solver
from thermal_no.optimize import DesignSpace, initial_designs, pnorm, solver_T, volume_shift


def _problem(n=16, seed=0):
    rng = np.random.default_rng(seed)
    cfg = D.InputConfig(n=n)
    return torch.tensor(D.sample_conductivity(cfg, rng)), torch.tensor(D.sample_sources(cfg, rng))


def test_solver_op_matches_solver():
    k, f = _problem()
    T = solver_T(k[None], f[None])[0]
    assert np.allclose(T.numpy(), solver.solve(k.numpy(), f.numpy()))


def test_adjoint_gradient_matches_finite_differences():
    k, f = _problem()
    k = k.requires_grad_(True)
    J = pnorm(solver_T(k[None], f[None]), 8.0).sum()
    (g,) = torch.autograd.grad(J, k)
    rng = np.random.default_rng(1)
    for _ in range(5):
        i, j = rng.integers(0, 16, 2)
        eps = 1e-6 * k[i, j].item()
        kp, km = k.detach().clone(), k.detach().clone()
        kp[i, j] += eps
        km[i, j] -= eps
        fd = (pnorm(solver_T(kp[None], f[None]), 8.0) - pnorm(solver_T(km[None], f[None]), 8.0)).item() / (2 * eps)
        assert abs(fd - g[i, j].item()) <= 1e-5 * abs(fd) + 1e-10, (fd, g[i, j].item())


def test_volume_constraint_exact_and_gradient_implicit():
    space = DesignSpace(volume_fraction=0.3)
    z = initial_designs(16, 3, seed=0, dtype=torch.float64)
    rho = space.density(z)
    assert torch.allclose(rho.mean((1, 2)), torch.full((3,), 0.3, dtype=torch.float64), atol=1e-10)
    # d/dz of the volume must vanish (the constraint holds for every z)
    zz = z.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(space.density(zz).mean(), zz)
    assert g.abs().max() < 1e-10
    # gradcheck of the implicit volume shift
    zf = torch.randn(2, 6, 6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a: volume_shift(a, 0.4), (zf,))


def test_pnorm_approaches_max():
    T = torch.rand(2, 32, 32, dtype=torch.float64)
    m = T.flatten(1).max(1).values
    assert torch.all(pnorm(T, 200.0) <= m + 1e-12)
    assert torch.all(pnorm(T, 200.0) > 0.95 * m)
