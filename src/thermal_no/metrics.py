"""Evaluation metrics (all computed per sample, then summarised)."""
from __future__ import annotations

import numpy as np
import torch


def relative_l2(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """||pred - true||_2 / ||true||_2 per sample. Shapes (B, N, N) -> (B,)."""
    diff = (pred - true).flatten(1).norm(dim=1)
    return diff / true.flatten(1).norm(dim=1).clamp_min(1e-12)


def max_temperature_errors(pred: torch.Tensor, true: torch.Tensor) -> dict:
    """Error of the peak temperature, the quantity a thermal engineer cares about."""
    pm, tm = pred.flatten(1).max(1).values, true.flatten(1).max(1).values
    return {"max_T_abs_err": (pm - tm).abs(), "max_T_rel_err": (pm - tm).abs() / tm,
            "max_T_signed_rel_err": (pm - tm) / tm}


@torch.no_grad()
def predict(model, k: torch.Tensor, f: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    outs = [model(k[i:i + batch_size].to(device), f[i:i + batch_size].to(device)).cpu()
            for i in range(0, len(k), batch_size)]
    return torch.cat(outs)


def evaluate(model, k, f, T, batch_size: int = 64) -> dict:
    """Return per-sample metric arrays (numpy) for a dataset."""
    k, f, T = (torch.as_tensor(a, dtype=torch.float32) for a in (k, f, T))
    pred = predict(model, k, f, batch_size)
    out = {"rel_l2": relative_l2(pred, T)}
    out.update(max_temperature_errors(pred, T))
    return {key: v.numpy() for key, v in out.items()}


def summarise(per_sample: dict) -> dict:
    """mean / median / 95th percentile of each per-sample metric."""
    s = {}
    for key, v in per_sample.items():
        s[key] = {"mean": float(np.mean(v)), "median": float(np.median(v)),
                  "p95": float(np.percentile(v, 95))}
    return s
