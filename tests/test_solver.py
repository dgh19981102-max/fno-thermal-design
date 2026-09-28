"""Correctness tests for the finite-volume solver.

1. Method of manufactured solutions (MMS) with variable k and mixed BCs:
   second-order convergence.
2. Layered 1D problem with piecewise-constant k: agreement with the analytic
   solution.
3. Discrete energy balance: heat generated == heat through the sink.
4. The PyTorch residual reproduces the sparse operator exactly.
"""
import numpy as np
import torch

from thermal_no import solver
from thermal_no.physics import fv_residual


from thermal_no.verification import mms_rms_error as mms_error


def test_mms_second_order_convergence():
    ns = [16, 32, 64, 128]
    errs = [mms_error(n) for n in ns]
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(rates > 1.85), f"observed rates {rates}"
    assert errs[-1] < 1e-3


def test_layered_1d_analytic():
    # k depends on y only (3 layers), f = 1, whole bottom is the sink:
    #   -(k T')' = 1, T(0) = 0, T'(1) = 0  ->  heat flux q(y) = 1 - y,
    #   T(y) = int_0^y (1 - s) / k(s) ds.
    n = 120  # layer interfaces at y = 1/3, 2/3 fall on cell faces
    y = solver.cell_centers(n)
    k_layers = np.array([1.0, 10.0, 3.0])
    layer = np.minimum((y * 3).astype(int), 2)
    k = np.repeat(k_layers[layer][:, None], n, axis=1)
    f = np.ones((n, n))
    T = solver.solve(k, f, sink_width=2.0)

    def T_exact(yy):
        edges = [0.0, 1 / 3, 2 / 3, 1.0]
        total = 0.0
        for a, b, kk in zip(edges[:-1], edges[1:], k_layers):
            lo, hi = a, min(b, yy)
            if hi > lo:
                total += ((hi - hi**2 / 2) - (lo - lo**2 / 2)) / kk
        return total

    Te = np.array([T_exact(v) for v in y])
    assert np.allclose(T[:, n // 2], Te, rtol=0, atol=2e-4 * Te.max())
    assert np.allclose(T, T[:, :1], atol=1e-10)  # no x-variation


def test_energy_balance_partial_sink():
    rng = np.random.default_rng(0)
    n = 64
    k = np.exp(rng.normal(size=(n, n)))
    f = rng.uniform(0, 2, size=(n, n))
    T = solver.solve(k, f)
    q_in = solver.total_heat_generated(f)
    q_out = solver.sink_heat_flow(T, k)
    assert abs(q_in - q_out) / q_in < 1e-10


def test_torch_residual_matches_sparse_operator():
    rng = np.random.default_rng(1)
    n = 32
    k = np.exp(rng.normal(size=(n, n)))
    f = rng.uniform(0, 1, size=(n, n))
    T = solver.solve(k, f)
    # residual of the exact discrete solution must vanish
    r = fv_residual(torch.tensor(T), torch.tensor(k), torch.tensor(f))
    assert r.abs().max().item() < 1e-12
    # and A @ v must match for an arbitrary v
    v = rng.normal(size=(n, n))
    A = solver.assemble(k, solver.sink_mask(n))
    Av = (A @ v.ravel()).reshape(n, n)
    r2 = fv_residual(torch.tensor(v), torch.tensor(k), torch.zeros(n, n, dtype=torch.float64))
    assert np.allclose(r2.numpy(), Av, atol=1e-12)


def test_matrix_symmetric():
    rng = np.random.default_rng(2)
    k = np.exp(rng.normal(size=(16, 16)))
    A = solver.assemble(k, solver.sink_mask(16))
    assert abs(A - A.T).max() < 1e-14
