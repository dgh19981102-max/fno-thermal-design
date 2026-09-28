"""Experiment 05 - inference time of the surrogates vs the reference solver.

Solver: assembly + sparse LU + solve per sample (SciPy/SuperLU, single CPU thread
of work, float64), at 64x64 and 128x128.
Surrogates: forward pass at several batch sizes on the selected device (GPU on
Colab) and on the CPU. GPU timings synchronise the device before stopping the clock.
Speed-ups are always stated together with the hardware they were measured on.
"""
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from thermal_no import data as D
from thermal_no import solver
from thermal_no.config import cli
from thermal_no.train import load_model
from thermal_no.utils import hardware_info, pick_device, write_json


def time_solver(k, f, repeats=1):
    ts = []
    for kk, ff in zip(k, f):
        t0 = time.perf_counter()
        for _ in range(repeats):
            solver.solve(kk, ff)
        ts.append((time.perf_counter() - t0) / repeats)
    return float(np.median(ts))


@torch.no_grad()
def time_model(model, k, f, batch, device, n_rep=20):
    model = model.to(device).eval()
    kb = torch.as_tensor(k[:batch], dtype=torch.float32, device=device)
    fb = torch.as_tensor(f[:batch], dtype=torch.float32, device=device)
    for _ in range(3):                                   # warm-up (cuFFT plans, allocator)
        model(kb, fb)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        model(kb, fb)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) / len(kb)                # seconds per sample


def main():
    args, cfg = cli(__doc__.splitlines()[1])
    device = pick_device(args.device)
    data_dir, res, art = Path(cfg["data_dir"]), Path(cfg["results_dir"]), Path(cfg["artifacts_dir"])
    sc = cfg["speed"]
    test, sr = D.load(data_dir / "test.npz"), D.load(data_dir / "superres.npz")
    n = min(sc["n_samples"], len(test["k"]))
    grids = {"64": (test["k"][:n], test["f"][:n]), "128": (sr["k"][:n], sr["f"][:n])}

    out = {"hardware": hardware_info(device), "solver_s_per_sample": {}, "models": {}}
    for g, (k, f) in grids.items():
        out["solver_s_per_sample"][g] = time_solver(k.astype(np.float64), f.astype(np.float64))
        print(f"solver {g}x{g}: {out['solver_s_per_sample'][g] * 1e3:.2f} ms/sample")

    seed = cfg["benchmark"]["seeds"][0]
    devices = [device] + (["cpu"] if device != "cpu" else [])
    for name in cfg["benchmark"]["models"]:
        path = art / "models" / f"{name}_seed{seed}.pt"
        if not path.exists():
            print(f"[missing] {path} - run 03 first")
            continue
        model = load_model(path)
        out["models"][name] = {}
        for dev in devices:
            for g, (k, f) in grids.items():
                for b in sc["batch_sizes"]:
                    if b > len(k):
                        continue
                    t = time_model(model, k, f, b, dev)
                    key = f"{dev}_grid{g}_batch{b}"
                    out["models"][name][key] = {"s_per_sample": t,
                                                "speedup_vs_solver": out["solver_s_per_sample"][g] / t}
                    print(f"{name:9s} {key:24s} {t * 1e3:8.3f} ms/sample  "
                          f"speed-up x{out['solver_s_per_sample'][g] / t:8.1f}")
    write_json(res / "speed.json", out)


if __name__ == "__main__":
    main()
