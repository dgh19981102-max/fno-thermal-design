"""Finite-volume solver for 2D steady heat conduction.

Problem (unit square, cell-centred grid of N x N cells, h = 1/N):

    -div( k(x, y) grad T ) = f(x, y)

Boundary conditions ("volume-to-point" heat-sink setup):
    * T = 0 on a heat-sink segment of the bottom edge (y = 0),
      |x - 0.5| <= sink_width / 2
    * zero heat flux (adiabatic) on the rest of the boundary.

Discretisation:
    Each cell balances the heat leaving through its four faces against the heat
    generated inside it. The flux through the face shared by cells P and Q is

        q_PQ = k_face * (T_P - T_Q) / h * h = k_face * (T_P - T_Q),

    with the harmonic mean k_face = 2 k_P k_Q / (k_P + k_Q). The harmonic mean
    is the exact effective conductivity of two slabs in series, so flux is
    continuous across material jumps. At a sink face the distance from the cell
    centre to the wall is h/2, giving q = 2 k_P (T_P - 0).

    Cell balance:  sum_faces q = f_P * h^2   ->   A(k) T = b.

A is symmetric positive definite (as long as the sink has at least one cell),
which the adjoint gradient in optimize.py relies on.

Array convention: fields have shape (N, N) with axis 0 = y (row 0 is the
bottom edge) and axis 1 = x. Flattening is row-major: idx = i * N + j.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

DEFAULT_SINK_WIDTH = 0.25


def cell_centers(n: int) -> np.ndarray:
    """Return the 1D cell-centre coordinates of an n-cell grid on [0, 1]."""
    h = 1.0 / n
    return (np.arange(n) + 0.5) * h


def sink_mask(n: int, sink_width: float = DEFAULT_SINK_WIDTH) -> np.ndarray:
    """Boolean mask (length n) of bottom-row cells whose lower face is the sink."""
    x = cell_centers(n)
    mask = np.abs(x - 0.5) <= sink_width / 2 + 1e-12
    if not mask.any():
        raise ValueError("Sink is narrower than one cell; increase n or sink_width.")
    return mask


def harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 2.0 * a * b / (a + b)


def assemble(k: np.ndarray, sink: np.ndarray) -> sp.csr_matrix:
    """Assemble the sparse FV matrix A(k) for an (N, N) conductivity field.

    Parameters
    ----------
    k    : (N, N) positive conductivity field.
    sink : (N,) boolean mask of bottom cells in contact with the T = 0 sink.
    """
    n = k.shape[0]
    assert k.shape == (n, n), "Only square grids are supported."
    idx = np.arange(n * n).reshape(n, n)

    rows, cols, vals = [], [], []
    diag = np.zeros((n, n))

    # Horizontal faces (between column j and j+1).
    kx = harmonic_mean(k[:, :-1], k[:, 1:])
    diag[:, :-1] += kx
    diag[:, 1:] += kx
    p, q = idx[:, :-1].ravel(), idx[:, 1:].ravel()
    rows += [p, q]
    cols += [q, p]
    vals += [-kx.ravel(), -kx.ravel()]

    # Vertical faces (between row i and i+1).
    ky = harmonic_mean(k[:-1, :], k[1:, :])
    diag[:-1, :] += ky
    diag[1:, :] += ky
    p, q = idx[:-1, :].ravel(), idx[1:, :].ravel()
    rows += [p, q]
    cols += [q, p]
    vals += [-ky.ravel(), -ky.ravel()]

    # Dirichlet sink faces on the bottom row (half-cell distance).
    diag[0, sink] += 2.0 * k[0, sink]

    rows.append(idx.ravel())
    cols.append(idx.ravel())
    vals.append(diag.ravel())
    A = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n * n, n * n),
    )
    return A


def rhs(f: np.ndarray) -> np.ndarray:
    """Right-hand side b = f * h^2 (heat generated per cell, per unit depth)."""
    n = f.shape[0]
    return (f / n**2).ravel()


def solve(
    k: np.ndarray,
    f: np.ndarray,
    sink_width: float = DEFAULT_SINK_WIDTH,
    return_factor: bool = False,
):
    """Solve for the temperature field T (shape (N, N)).

    If return_factor is True, also return the sparse LU factorisation so that
    the adjoint system (A is symmetric) can be solved at almost no extra cost.
    """
    n = k.shape[0]
    A = assemble(np.asarray(k, dtype=np.float64), sink_mask(n, sink_width))
    lu = spla.splu(A.tocsc())
    T = lu.solve(rhs(np.asarray(f, dtype=np.float64))).reshape(n, n)
    if return_factor:
        return T, lu
    return T


def sink_heat_flow(T: np.ndarray, k: np.ndarray, sink_width: float = DEFAULT_SINK_WIDTH) -> float:
    """Total heat leaving through the sink (per unit depth)."""
    n = k.shape[0]
    m = sink_mask(n, sink_width)
    return float(np.sum(2.0 * k[0, m] * T[0, m]))


def total_heat_generated(f: np.ndarray) -> float:
    n = f.shape[0]
    return float(np.sum(f) / n**2)
