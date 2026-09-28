"""Reference problems used to verify the solver (shared by tests and experiments)."""
from __future__ import annotations

import numpy as np

from . import solver


def mms_fields(n: int):
    """Manufactured solution with variable k and mixed boundary conditions.

    T = cos(pi x) (1 - cos(pi y)) satisfies T = 0 on y = 0 and zero normal
    derivative on the other three edges, i.e. the "whole bottom edge is a sink"
    case (solve with sink_width >= 1). f is derived analytically:
        f = -(k_x T_x + k T_xx + k_y T_y + k T_yy).
    """
    x = solver.cell_centers(n)
    X, Y = np.meshgrid(x, x)
    pi = np.pi
    k = 1.0 + 0.5 * np.sin(2 * pi * X) * np.sin(2 * pi * Y)
    kx = pi * np.cos(2 * pi * X) * np.sin(2 * pi * Y)
    ky = pi * np.sin(2 * pi * X) * np.cos(2 * pi * Y)
    T = np.cos(pi * X) * (1 - np.cos(pi * Y))
    Tx = -pi * np.sin(pi * X) * (1 - np.cos(pi * Y))
    Ty = pi * np.cos(pi * X) * np.sin(pi * Y)
    Txx = -pi**2 * np.cos(pi * X) * (1 - np.cos(pi * Y))
    Tyy = pi**2 * np.cos(pi * X) * np.cos(pi * Y)
    f = -(kx * Tx + k * Txx + ky * Ty + k * Tyy)
    return k, f, T


def mms_rms_error(n: int) -> float:
    k, f, T_exact = mms_fields(n)
    T = solver.solve(k, f, sink_width=2.0)
    return float(np.sqrt(np.mean((T - T_exact) ** 2)))


def smooth_sink_problem(n: int):
    """A smooth, resolution-independent instance of the real (partial-sink) problem."""
    x = solver.cell_centers(n)
    X, Y = np.meshgrid(x, x)
    k = 5.0 + 4.0 * np.sin(2 * np.pi * X) * np.cos(3 * np.pi * Y)
    f = np.zeros_like(X)
    for cx, cy, w, p in [(0.3, 0.7, 0.06, 1.0), (0.7, 0.5, 0.05, 0.8), (0.5, 0.25, 0.07, 0.6)]:
        f += p / (2 * np.pi * w**2) * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * w**2))
    return k, f
