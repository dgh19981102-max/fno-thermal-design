"""Experiment 04 - does a physics-residual loss help? (FNO vs FNO + physics loss)

Grid: training-set size x physics weight x seed. Every run uses the same number
of optimiser steps. The physics term is the discrete FV residual ||A(k) T - b||,
i.e. the same discretisation that produced the labels (see physics.py).
Writes results/ablation/fno_n<N>_pw<w>_seed<s>.json.

Expectation (not a guarantee): physics information matters most when labelled
data is scarce, and adds little once the training set is large.
"""
from pathlib import Path

import _bootstrap  # noqa: F401

from thermal_no import data as D
from thermal_no.config import cli
from thermal_no.evaluation import TestData, evaluate_all
from thermal_no.train import train_model
from thermal_no.utils import hardware_info, pick_device, write_json


def main():
    args, cfg = cli(__doc__.splitlines()[1])
    device = pick_device(args.device)
    data_dir, res, art = Path(cfg["data_dir"]), Path(cfg["results_dir"]), Path(cfg["artifacts_dir"])
    train, val = D.load(data_dir / "train.npz"), D.load(data_dir / "val.npz")
    tests = TestData(data_dir)
    ab = cfg["ablation"]

    for n_train in ab["n_train"]:
        for pw in ab["physics_weight"]:
            for seed in ab["seeds"]:
                tag = f"fno_n{n_train}_pw{pw:g}_seed{seed}"
                out = res / "ablation" / f"{tag}.json"
                if out.exists():
                    print(f"[skip] {out.name}")
                    continue
                print(f"[train] {tag}")
                model, hist = train_model("fno", cfg["models"]["fno"], train, val, cfg["train"], seed, device,
                                          physics_weight=pw, n_train=n_train,
                                          ckpt_path=art / "checkpoints" / f"{tag}.ckpt")
                evals = evaluate_all(model, tests)
                print(f"    test rel L2 {evals['test']['summary']['rel_l2']['mean']:.4f}")
                write_json(out, {"n_train": n_train, "physics_weight": pw, "seed": seed,
                                 "history": hist, "eval": evals, "hardware": hardware_info(device)})


if __name__ == "__main__":
    main()
