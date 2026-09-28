"""Experiment 08 - why does the FNO generalise worse than the U-Net here?

Observation from experiments 03/04 (first full run): the benchmark FNO reaches
~0.5 % relative L2 on its training data but ~16 % on the test set, a much larger
train/test gap than the U-Net (~1 % vs ~6 %). The physics loss, which acts as a
regulariser, reduced the FNO test error (experiment 04).

Hypothesis: the gap is over-fitting of the high-capacity spectral weights.
If so, each of the following single changes should reduce the test error:
    fno_small     fewer modes / channels (8 modes, width 24)
    fno_wd        100x stronger weight decay
    fno_flip_aug  exact mirror-symmetry data augmentation (x -> 1 - x)
The variants were declared before running, are run with 3 seeds each, and are
reported whatever they show. They do NOT replace the benchmark numbers.
Writes results/diagnostics/<variant>_seed<s>.json.
"""
import copy
from pathlib import Path

import _bootstrap  # noqa: F401

from thermal_no import data as D
from thermal_no.config import _merge, cli
from thermal_no.evaluation import TestData, evaluate_all
from thermal_no.train import train_model
from thermal_no.utils import hardware_info, pick_device, write_json


def main():
    args, cfg = cli(__doc__.splitlines()[1])
    device = pick_device(args.device)
    data_dir, res, art = Path(cfg["data_dir"]), Path(cfg["results_dir"]), Path(cfg["artifacts_dir"])
    train, val = D.load(data_dir / "train.npz"), D.load(data_dir / "val.npz")
    tests = TestData(data_dir)
    dg = cfg["diagnostics"]

    for variant, change in dg["variants"].items():
        model_cfg = _merge(cfg["models"]["fno"], change.get("model", {}))
        train_cfg = _merge(cfg["train"], change.get("train", {}))
        for seed in dg["seeds"]:
            out = res / "diagnostics" / f"{variant}_seed{seed}.json"
            if out.exists():
                print(f"[skip] {out.name}")
                continue
            print(f"[train] {variant} seed {seed}  model {model_cfg}  train-changes {change.get('train', {})}")
            model, hist = train_model("fno", model_cfg, train, val, train_cfg, seed, device,
                                      ckpt_path=art / "checkpoints" / f"diag_{variant}_seed{seed}.ckpt")
            evals = evaluate_all(model, tests)
            print(f"    train {hist['train_data'][-1]:.4f}   test rel L2 {evals['test']['summary']['rel_l2']['mean']:.4f}"
                  f"   @128 {evals['superres_128']['summary']['rel_l2']['mean']:.4f}")
            write_json(out, {"variant": variant, "seed": seed, "change": copy.deepcopy(change),
                             "model_cfg": model_cfg, "train_cfg": train_cfg, "history": hist,
                             "eval": evals, "hardware": hardware_info(device)})


if __name__ == "__main__":
    main()
