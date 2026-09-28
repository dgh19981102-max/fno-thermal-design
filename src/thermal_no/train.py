"""Training loop shared by all models (data-driven and physics-informed).

Loss
    L = mean_b  ||T_pred - T_true|| / ||T_true||            (relative L2, data term)
      + lambda * mean_b ||A(k) T_pred - b|| ^2 / ||b||^2    (physics term, optional)

The relative L2 loss (instead of plain MSE) stops the few samples with a large
heat load, and therefore large temperatures, from dominating the gradient.

The run length is a fixed number of optimiser *steps*, not epochs, so that runs
with 100 and 1200 training samples get the same compute budget in the ablation.

Checkpointing: the full training state (model, optimiser, scheduler, RNG,
best weights) is saved every `eval_every` steps, so a run interrupted by a
Colab disconnect resumes where it stopped when the script is re-run.
"""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import torch

from .metrics import relative_l2
from .models import Surrogate, build_model, count_params
from .physics import physics_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensors(d: dict, n: int | None = None, device="cpu"):
    sl = slice(None) if n is None else slice(0, n)
    return tuple(torch.as_tensor(d[key][sl], dtype=torch.float32, device=device) for key in ("k", "f", "T"))


def train_model(name: str, model_cfg: dict, train: dict, val: dict, tcfg: dict, seed: int,
                device: str = "cpu", physics_weight: float = 0.0, n_train: int | None = None,
                ckpt_path: str | Path | None = None, log=print) -> tuple[Surrogate, dict]:
    """Train one surrogate and return (best-on-validation model, history)."""
    set_seed(seed)
    k_tr, f_tr, T_tr = to_tensors(train, n_train, device)
    k_va, f_va, T_va = to_tensors(val, None, device)

    model = Surrogate(build_model(name, model_cfg), Surrogate.compute_stats(k_tr, f_tr, T_tr)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    steps, bs = tcfg["steps"], min(tcfg["batch_size"], len(k_tr))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    gen = torch.Generator().manual_seed(seed)          # controls mini-batch order

    history = {"step": [], "train_loss": [], "train_data": [], "train_phys": [], "val_rel_l2": []}
    best = {"val": float("inf"), "state": None, "step": 0}
    step, t_used = 0, 0.0

    ckpt_path = Path(ckpt_path) if ckpt_path else None
    if ckpt_path and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"].cpu()); torch.set_rng_state(ck["torch_rng"].cpu())  # RNG states must live on the CPU
        history, best, step, t_used = ck["history"], ck["best"], ck["step"], ck["time"]
        log(f"  resumed from step {step}")

    perm, pos = torch.randperm(len(k_tr), generator=gen), 0
    acc = {"loss": 0.0, "data": 0.0, "phys": 0.0, "n": 0}
    t0 = time.perf_counter()
    while step < steps:
        # ---- sample a mini-batch (reshuffle after every pass over the data)
        if pos + bs > len(perm):
            perm, pos = torch.randperm(len(k_tr), generator=gen), 0
        idx = perm[pos:pos + bs].to(device)
        pos += bs
        k, f, T = k_tr[idx], f_tr[idx], T_tr[idx]
        if tcfg.get("augment_flip", False):
            # Mirror x -> 1 - x. The domain and the centred heat sink are symmetric,
            # so a mirrored (k, f, T) triple is an exact solution of the same PDE.
            flip = (torch.rand(len(idx), generator=gen) < 0.5).to(device)[:, None, None]
            k, f, T = (torch.where(flip, a.flip(-1), a) for a in (k, f, T))

        # ---- one optimisation step
        model.train()
        T_pred = model(k, f)                                   # physical units, (B, N, N)
        loss_data = relative_l2(T_pred, T).mean()
        loss = loss_data
        loss_phys = torch.zeros((), device=device)
        if physics_weight > 0:
            loss_phys = physics_loss(T_pred, k, f)
            loss = loss + physics_weight * loss_phys
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        step += 1

        acc["loss"] += loss.item(); acc["data"] += loss_data.item(); acc["phys"] += loss_phys.item(); acc["n"] += 1

        # ---- periodic validation + checkpoint
        if step % tcfg["eval_every"] == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                val = torch.cat([relative_l2(model(k_va[i:i + 64], f_va[i:i + 64]), T_va[i:i + 64])
                                 for i in range(0, len(k_va), 64)]).mean().item()
            n = max(acc["n"], 1)
            history["step"].append(step); history["val_rel_l2"].append(val)
            history["train_loss"].append(acc["loss"] / n); history["train_data"].append(acc["data"] / n)
            history["train_phys"].append(acc["phys"] / n)
            acc = {"loss": 0.0, "data": 0.0, "phys": 0.0, "n": 0}
            if val < best["val"]:
                best = {"val": val, "state": copy.deepcopy(model.state_dict()), "step": step}
            log(f"  step {step:6d}/{steps}  train {history['train_data'][-1]:.4f}"
                f"  phys {history['train_phys'][-1]:.2e}  val {val:.4f}  lr {sched.get_last_lr()[0]:.1e}")
            if ckpt_path:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "gen": gen.get_state(), "torch_rng": torch.get_rng_state(), "history": history,
                            "best": best, "step": step,
                            "time": t_used + time.perf_counter() - t0}, ckpt_path)

    model.load_state_dict(best["state"])
    history["best_step"], history["best_val_rel_l2"] = best["step"], best["val"]
    history["train_time_s"] = t_used + time.perf_counter() - t0
    history["n_params"] = count_params(model.net)
    history["n_train"] = len(k_tr)
    return model, history


def save_model(model: Surrogate, name: str, model_cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"name": name, "model_cfg": model_cfg, "state": model.state_dict()}, path)


def load_model(path: Path, device: str = "cpu") -> Surrogate:
    ck = torch.load(path, map_location=device, weights_only=False)
    dummy = {k: 1.0 for k in ("logk_mean", "logk_std", "f_mean", "f_std", "T_std")}
    model = Surrogate(build_model(ck["name"], ck["model_cfg"]), dummy)
    model.load_state_dict(ck["state"])
    return model.to(device).eval()


def finetune(model: Surrogate, k: torch.Tensor, f: torch.Tensor, T: torch.Tensor, steps: int,
             lr: float = 3e-4, batch_size: int = 20, seed: int = 0) -> Surrogate:
    """Continue training a copy of `model` on (k, f, T) with the data loss (normalisation kept fixed)."""
    set_seed(seed)
    model = copy.deepcopy(model)
    device = next(model.parameters()).device
    k, f, T = (a.to(device, torch.float32) for a in (k, f, T))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    bs = min(batch_size, len(k))
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(k), (bs,), generator=gen).to(device)
        loss = relative_l2(model(k[idx], f[idx]), T[idx]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.eval()
