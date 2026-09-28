"""Experiment 03 - train FNO, U-Net and DeepONet over several seeds and evaluate them.

For every (model, seed):
    * train on the full training set (data loss only);
    * evaluate on the in-distribution test set, the three OOD sets and the paired
      64/128 super-resolution set;
    * write results/benchmark/<model>_seed<s>.json and models/<model>_seed<s>.pt.
Runs whose JSON already exists are skipped, and unfinished runs resume from their
checkpoint, so the script can simply be re-run after a Colab disconnect.
The random seed changes weight initialisation and mini-batch order; the dataset
itself is fixed (generated once by experiment 02).
"""
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from thermal_no import data as D
from thermal_no.config import cli
from thermal_no.evaluation import TestData, evaluate_all, example_predictions
from thermal_no.train import save_model, train_model
from thermal_no.utils import hardware_info, pick_device, write_json


def main():
    args, cfg = cli(__doc__.splitlines()[1])
    device = pick_device(args.device)
    data_dir, res, art = Path(cfg["data_dir"]), Path(cfg["results_dir"]), Path(cfg["artifacts_dir"])
    train, val = D.load(data_dir / "train.npz"), D.load(data_dir / "val.npz")
    tests = TestData(data_dir)
    print(f"device: {device}, train samples: {len(train['k'])}")

    for name in cfg["benchmark"]["models"]:
        for seed in cfg["benchmark"]["seeds"]:
            out = res / "benchmark" / f"{name}_seed{seed}.json"
            if out.exists():
                print(f"[skip] {out.name} exists")
                continue
            print(f"[train] {name} seed {seed}")
            model, hist = train_model(
                name, cfg["models"][name], train, val, cfg["train"], seed, device,
                ckpt_path=art / "checkpoints" / f"{name}_seed{seed}.ckpt")
            save_model(model, name, cfg["models"][name], art / "models" / f"{name}_seed{seed}.pt")
            evals = evaluate_all(model, tests)
            for set_name, e in evals.items():
                print(f"    {set_name:20s} rel L2 {e['summary']['rel_l2']['mean']:.4f}"
                      f"   max-T rel err {e['summary']['max_T_rel_err']['mean']:.4f}")
            write_json(out, {"model": name, "seed": seed, "model_cfg": cfg["models"][name],
                             "train_cfg": cfg["train"], "history": hist, "eval": evals,
                             "hardware": hardware_info(device)})
            if seed == cfg["benchmark"]["seeds"][0]:
                np.savez_compressed(res / "benchmark" / f"{name}_examples.npz", **example_predictions(model, tests))


if __name__ == "__main__":
    main()
