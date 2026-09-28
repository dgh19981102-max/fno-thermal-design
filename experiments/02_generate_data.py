"""Experiment 02 - generate all datasets with the reference FV solver.

Writes to <data_dir>/:
    train.npz, val.npz, test.npz                 in-distribution, 64x64
    ood_many_sources.npz, ood_edge_sources.npz, ood_high_contrast.npz
    superres.npz                                 paired 128x128 / 64x64 problems
and a preview figure docs/figures/dataset_samples.png.
Each split uses its own random seed, so no sample is shared between splits.
"""
import json
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from thermal_no import data as D
from thermal_no.config import REPO_ROOT, cli
from thermal_no.plotting import apply_style, draw_sink, show_field


def main():
    args, cfg = cli(__doc__.splitlines()[1])
    dc = cfg["data"]
    out = Path(cfg["data_dir"])
    base = D.InputConfig(n=dc["n"])
    seed = dc["seed"]
    stats = {}

    splits = [("train", dc["n_train"], base, seed + 1),
              ("val", dc["n_val"], base, seed + 2),
              ("test", dc["n_test"], base, seed + 3)]
    for i, (name, kw) in enumerate([(v, kw) for v, kw in D.VARIANTS.items() if v != "in_dist"]):
        splits.append((name, dc["n_ood"], replace(base, **kw), seed + 10 + i))

    for name, num, icfg, s in splits:
        print(f"[{name}] {num} samples, seed {s}")
        d = D.generate(num, icfg, s)
        D.save(out / f"{name}.npz", d)
        stats[name] = {"num": num, "mean_solve_time_s": float(d["solve_time"].mean()),
                       "T_max_mean": float(d["T"].max(axis=(1, 2)).mean()),
                       "T_max_range": [float(d["T"].max(axis=(1, 2)).min()),
                                       float(d["T"].max(axis=(1, 2)).max())]}

    n_hi = dc["superres_n"]
    print(f"[superres] {dc['n_superres']} paired problems at {n_hi} and {dc['n']}")
    sr = D.generate_paired_superres(dc["n_superres"], replace(base, n=n_hi), n_hi // dc["n"], seed + 20)
    D.save(out / "superres.npz", sr)
    stats["superres"] = {"num": dc["n_superres"], "mean_solve_time_s_hi": float(sr["solve_time"].mean())}

    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["results_dir"]) / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    # Preview figure: 2 training samples + one sample of each OOD set.
    apply_style()
    names = ["train", "train", "ood_many_sources", "ood_edge_sources", "ood_high_contrast"]
    idx = [0, 1, 0, 0, 0]
    fig, axes = plt.subplots(3, len(names), figsize=(3.3 * len(names), 9.3))
    for c, (nm, i) in enumerate(zip(names, idx)):
        d = D.load(out / f"{nm}.npz")
        show_field(axes[0, c], np.log10(d["k"][i]), f"{nm}\nlog10 k", cmap="viridis")
        show_field(axes[1, c], d["f"][i], "heat source f", cmap="magma")
        show_field(axes[2, c], d["T"][i], "temperature T")
        draw_sink(axes[2, c])
    fig_dir = REPO_ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "dataset_samples.png")
    print("saved", fig_dir / "dataset_samples.png")


if __name__ == "__main__":
    main()
