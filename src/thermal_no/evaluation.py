"""Evaluate a trained surrogate on every test set (in-distribution, OOD, super-resolution)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import data as D
from .metrics import evaluate, summarise, predict

TEST_SETS = ["test", "ood_many_sources", "ood_edge_sources", "ood_high_contrast"]


class TestData:
    """Loads all test sets once and keeps them in memory."""

    def __init__(self, data_dir: str | Path):
        data_dir = Path(data_dir)
        self.sets = {name: D.load(data_dir / f"{name}.npz") for name in TEST_SETS}
        sr = D.load(data_dir / "superres.npz")
        self.sets["superres_128"] = {"k": sr["k"], "f": sr["f"], "T": sr["T"]}
        self.sets["superres_64"] = {"k": sr["k_lo"], "f": sr["f_lo"], "T": sr["T_lo"]}


def evaluate_all(model, test_data: TestData, batch_size: int = 32) -> dict:
    """Returns {set_name: {"summary": {...}, "rel_l2": [...], "max_T_signed_rel_err": [...]}}."""
    out = {}
    for name, d in test_data.sets.items():
        per = evaluate(model, d["k"], d["f"], d["T"], batch_size)
        out[name] = {"summary": summarise(per),
                     "rel_l2": per["rel_l2"].round(6).tolist(),
                     "max_T_signed_rel_err": per["max_T_signed_rel_err"].round(6).tolist()}
    return out


def example_predictions(model, test_data: TestData, n: int = 4) -> dict:
    """A few predictions (numpy) for figures."""
    import torch
    ex = {}
    for name in ("test", "ood_many_sources", "superres_128"):
        d = test_data.sets[name]
        k, f = (torch.as_tensor(d[key][:n], dtype=torch.float32) for key in ("k", "f"))
        ex[f"{name}_pred"] = predict(model, k, f).numpy()
        for key in ("k", "f", "T"):
            ex[f"{name}_{key}"] = np.asarray(d[key][:n])
    return ex
