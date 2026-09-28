"""Random input generation and dataset construction.

Inputs of one sample:
    k : conductivity field in [k_min, k_max], drawn from one of two families
        - "smooth":    k = k_min * (k_max / k_min) ** sigmoid(a * g)
        - "two_phase": k = k_min + (k_max - k_min) * rho**3, with rho a smoothed
                       indicator of a thresholded random field. This family looks
                       like the density-based designs produced in optimize.py,
                       so the surrogate sees such layouts during training.
      where g is a Gaussian random field with physical correlation length ell.
    f : sum of n_src Gaussian heat sources (integrated power P_s each).

Random fields are generated in Fourier space with a filter that depends on the
*physical* wavenumber, so a 64x64 and a 128x128 field follow the same
distribution. This matters for the zero-shot super-resolution test.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import solver


@dataclass
class InputConfig:
    n: int = 64
    k_min: float = 1.0
    k_max: float = 20.0
    two_phase_prob: float = 0.5
    length_scale: tuple = (0.05, 0.2)
    n_sources: tuple = (1, 3)            # inclusive range
    source_region: str = "interior"      # "interior" | "edge"
    source_margin: float = 0.1
    source_width: tuple = (0.03, 0.07)
    source_power: tuple = (0.5, 1.5)
    sink_width: float = solver.DEFAULT_SINK_WIDTH


# Named dataset variants used in the experiments.
VARIANTS = {
    "in_dist": {},
    "ood_many_sources": {"n_sources": (4, 6)},
    "ood_edge_sources": {"source_region": "edge"},
    "ood_high_contrast": {"k_max": 100.0},
}


def gaussian_random_field(n: int, ell: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean, unit-variance GRF with a Gaussian spectrum of length scale ell."""
    kappa = np.fft.fftfreq(n, d=1.0 / n)  # cycles per unit length
    KX, KY = np.meshgrid(kappa, kappa)
    spectrum = np.exp(-0.5 * (2 * np.pi * ell) ** 2 * (KX**2 + KY**2))
    noise = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    g = np.real(np.fft.ifft2(noise * np.sqrt(spectrum)))
    g -= g.mean()
    return g / (g.std() + 1e-12)


def sample_conductivity(cfg: InputConfig, rng: np.random.Generator) -> np.ndarray:
    ell = rng.uniform(*cfg.length_scale)
    g = gaussian_random_field(cfg.n, ell, rng)
    if rng.uniform() < cfg.two_phase_prob:
        vol = rng.uniform(0.2, 0.5)                   # fraction of high-k material
        thr = np.quantile(g, 1.0 - vol)
        sharp = rng.uniform(4.0, 12.0)
        rho = 1.0 / (1.0 + np.exp(-sharp * (g - thr)))
        return cfg.k_min + (cfg.k_max - cfg.k_min) * rho**3
    a = rng.uniform(1.0, 3.0)
    s = 1.0 / (1.0 + np.exp(-a * g))
    return cfg.k_min * (cfg.k_max / cfg.k_min) ** s


def _source_centre(cfg: InputConfig, rng: np.random.Generator) -> np.ndarray:
    m = cfg.source_margin
    if cfg.source_region == "interior":
        return rng.uniform(m, 1 - m, size=2)
    if cfg.source_region == "edge":
        c = rng.uniform(0.02, 0.98, size=2)
        axis = rng.integers(2)
        c[axis] = rng.choice([rng.uniform(0.02, m), rng.uniform(1 - m, 0.98)])
        return c
    raise ValueError(cfg.source_region)


def sample_sources(cfg: InputConfig, rng: np.random.Generator) -> np.ndarray:
    x = solver.cell_centers(cfg.n)
    X, Y = np.meshgrid(x, x)
    f = np.zeros((cfg.n, cfg.n))
    for _ in range(rng.integers(cfg.n_sources[0], cfg.n_sources[1] + 1)):
        cx, cy = _source_centre(cfg, rng)
        w = rng.uniform(*cfg.source_width)
        p = rng.uniform(*cfg.source_power)
        f += p / (2 * np.pi * w**2) * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * w**2))
    return f


def generate(num: int, cfg: InputConfig, seed: int, verbose: bool = True) -> dict:
    """Generate `num` (k, f, T) samples with the reference solver."""
    rng = np.random.default_rng(seed)
    ks = np.empty((num, cfg.n, cfg.n), np.float32)
    fs = np.empty_like(ks)
    Ts = np.empty_like(ks)
    t_solve = np.empty(num)
    for i in range(num):
        k = sample_conductivity(cfg, rng)
        f = sample_sources(cfg, rng)
        t0 = time.perf_counter()
        T = solver.solve(k, f, cfg.sink_width)
        t_solve[i] = time.perf_counter() - t0
        ks[i], fs[i], Ts[i] = k, f, T
        if verbose and (i + 1) % max(1, num // 10) == 0:
            print(f"  {i + 1}/{num} samples")
    return {"k": ks, "f": fs, "T": Ts, "solve_time": t_solve, "config": asdict(cfg)}


def generate_paired_superres(num: int, cfg_hi: InputConfig, factor: int, seed: int) -> dict:
    """Generate inputs at high resolution plus their block-averaged coarse copy.

    Both resolutions are solved with the reference solver, so the same physical
    problems can be compared at 64x64 (training resolution) and 128x128.
    """
    hi = generate(num, cfg_hi, seed, verbose=False)
    n_lo = cfg_hi.n // factor

    def coarsen(a):
        return a.reshape(num, n_lo, factor, n_lo, factor).mean(axis=(2, 4))

    k_lo, f_lo = coarsen(hi["k"]), coarsen(hi["f"])
    T_lo = np.stack([solver.solve(k, f, cfg_hi.sink_width) for k, f in zip(k_lo, f_lo)]).astype(np.float32)
    return {
        "k": hi["k"], "f": hi["f"], "T": hi["T"],
        "k_lo": k_lo.astype(np.float32), "f_lo": f_lo.astype(np.float32), "T_lo": T_lo,
        "solve_time": hi["solve_time"], "config": asdict(cfg_hi),
    }


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(path, **arrays, config=np.array(repr(data.get("config", {}))))


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k != "config"}
