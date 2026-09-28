"""Experiment 07 - aggregate raw results into tables (results/summary/) and figures (docs/figures/).

Statistics: for each run we average a metric over the test samples; tables then
report mean +- sample standard deviation of that number ACROSS SEEDS, together
with the number of seeds. Missing result files are skipped with a warning, so
the script can be run at any stage.
"""
import glob
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from thermal_no.config import REPO_ROOT, cli
from thermal_no.plotting import (ACCENT, GREY, MODEL_COLORS, MODEL_LABELS, PRIMARY, apply_style, draw_sink,
                                 show_field)
from thermal_no.utils import read_json, write_json

SETS = ["test", "ood_many_sources", "ood_edge_sources", "ood_high_contrast", "superres_64", "superres_128"]
SET_LABELS = {"test": "In-distribution", "ood_many_sources": "OOD: 4-6 sources",
              "ood_edge_sources": "OOD: edge sources", "ood_high_contrast": "OOD: k contrast 100",
              "superres_64": "Paired set @64", "superres_128": "Zero-shot @128"}


def pct_log_axis(ax, ticks=(1, 2, 5, 10, 20, 50, 100)):
    """Log y-axis with readable percentage tick labels."""
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    t = [v for v in ticks if lo <= v <= hi] or list(ticks)
    ax.set_yticks(t); ax.set_yticklabels([f"{v:g}" for v in t]); ax.minorticks_off()


def ms(values):
    v = np.asarray(values, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)


def fmt(m, s, pct=True):
    return f"{100 * m:.2f} ± {100 * s:.2f} %" if pct else f"{m:.4g} ± {s:.2g}"


# ------------------------------------------------------------------ benchmark
def benchmark(res, fig_dir, summary, md):
    runs = defaultdict(list)
    for p in sorted(glob.glob(str(res / "benchmark" / "*_seed*.json"))):
        r = read_json(p)
        runs[r["model"]].append(r)
    if not runs:
        print("[warn] no benchmark results"); return
    models = [m for m in ("fno", "unet", "deeponet") if m in runs]
    table = {}
    md.append("## Benchmark (data loss only, full training set)\n")
    md.append("Mean over test samples, then mean ± std across seeds.\n")
    md.append("| Model | params | seeds | " + " | ".join(SET_LABELS[s] for s in SETS) + " |")
    md.append("|---" * (3 + len(SETS)) + "|")
    for m in models:
        table[m] = {}
        row = []
        for s in SETS:
            rel = ms([r["eval"][s]["summary"]["rel_l2"]["mean"] for r in runs[m]])
            mx = ms([r["eval"][s]["summary"]["max_T_rel_err"]["mean"] for r in runs[m]])
            table[m][s] = {"rel_l2": rel, "max_T_rel_err": mx}
            row.append(fmt(*rel[:2]))
        n_par = runs[m][0]["history"]["n_params"]
        md.append(f"| {MODEL_LABELS[m]} | {n_par / 1e6:.2f} M | {len(runs[m])} | " + " | ".join(row) + " |")
    md.append("\n**Relative error of the peak temperature** (same layout):\n")
    md.append("| Model | " + " | ".join(SET_LABELS[s] for s in SETS) + " |")
    md.append("|---" * (1 + len(SETS)) + "|")
    for m in models:
        md.append(f"| {MODEL_LABELS[m]} | " + " | ".join(fmt(*table[m][s]["max_T_rel_err"][:2]) for s in SETS) + " |")
    md.append("")
    summary["benchmark"] = table
    summary["benchmark_train_time_s"] = {m: ms([r["history"]["train_time_s"] for r in runs[m]]) for m in models}

    # grouped bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.3))
    w = 0.8 / len(models)
    x = np.arange(len(SETS))
    for ax, metric, title in ((axes[0], "rel_l2", "Relative L2 error of the temperature field"),
                              (axes[1], "max_T_rel_err", "Relative error of the peak temperature")):
        for i, m in enumerate(models):
            mean = [table[m][s][metric][0] for s in SETS]
            std = [table[m][s][metric][1] for s in SETS]
            ax.bar(x + (i - (len(models) - 1) / 2) * w, np.array(mean) * 100, w, yerr=np.array(std) * 100,
                   color=MODEL_COLORS[m], label=MODEL_LABELS[m], capsize=3)
        ax.set_xticks(x); ax.set_xticklabels([SET_LABELS[s] for s in SETS], rotation=20, ha="right")
        ax.set_ylabel("error [%]  (mean ± std over seeds)"); ax.set_title(title)
        pct_log_axis(ax, (2, 5, 10, 20, 50, 100))
    axes[0].legend()
    fig.savefig(fig_dir / "benchmark_errors.png"); plt.close(fig)

    # training curves (seed 0 of each model)
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in models:
        h = runs[m][0]["history"]
        ax.semilogy(h["step"], h["val_rel_l2"], color=MODEL_COLORS[m], label=MODEL_LABELS[m])
        ax.semilogy(h["step"], h["train_data"], color=MODEL_COLORS[m], ls=":", alpha=0.7)
    ax.set_xlabel("optimiser step"); ax.set_ylabel("relative L2 (solid: val, dotted: train)")
    ax.set_title("Training curves (seed 0)"); ax.legend()
    fig.savefig(fig_dir / "training_curves.png"); plt.close(fig)

    # example predictions
    ex_models = [m for m in models if (res / "benchmark" / f"{m}_examples.npz").exists()]
    if ex_models:
        exs = {m: np.load(res / "benchmark" / f"{m}_examples.npz") for m in ex_models}
        for set_name, fname in (("test", "examples_test.png"), ("superres_128", "examples_superres.png"),
                                ("ood_many_sources", "examples_ood.png")):
            e0 = exs[ex_models[0]]
            T = e0[f"{set_name}_T"][0]
            ncol = 2 + len(ex_models)
            fig, axes = plt.subplots(2, ncol, figsize=(3.2 * ncol, 6.2))
            show_field(axes[0, 0], np.log10(e0[f"{set_name}_k"][0]), "log10 k", cmap="viridis")
            show_field(axes[1, 0], e0[f"{set_name}_f"][0], "heat source f", cmap="magma")
            show_field(axes[0, 1], T, "FV solver (truth)", vmin=0, vmax=T.max()); draw_sink(axes[0, 1])
            axes[1, 1].axis("off")
            err_max = max(np.abs(exs[m][f"{set_name}_pred"][0] - T).max() for m in ex_models)
            for j, m in enumerate(ex_models):
                P = exs[m][f"{set_name}_pred"][0]
                rel = np.linalg.norm(P - T) / np.linalg.norm(T)
                show_field(axes[0, 2 + j], P, f"{MODEL_LABELS[m]} (rel. L2 {100 * rel:.1f}%)", vmin=0, vmax=T.max())
                show_field(axes[1, 2 + j], np.abs(P - T), f"|error| {MODEL_LABELS[m]}", cmap="Reds", vmin=0,
                           vmax=err_max)
            fig.suptitle(f"{SET_LABELS[set_name]}: grid {T.shape[0]}x{T.shape[1]}")
            fig.savefig(fig_dir / fname); plt.close(fig)


# ------------------------------------------------------------------ ablation
def ablation(res, fig_dir, summary, md):
    runs = [read_json(p) for p in sorted(glob.glob(str(res / "ablation" / "*.json")))]
    if not runs:
        print("[warn] no ablation results"); return
    ns = sorted({r["n_train"] for r in runs})
    pws = sorted({r["physics_weight"] for r in runs})
    tab = {}
    md.append("## Physics-loss ablation (FNO)\n")
    md.append("Welch t-test across seeds for the difference in mean test relative L2 (small n: interpret with care).\n")
    md.append("| n_train | " + " | ".join(f"physics weight {pw:g}" for pw in pws) + " | p-value |")
    md.append("|---" * (2 + len(pws)) + "|")
    for n in ns:
        vals = {pw: [r["eval"]["test"]["summary"]["rel_l2"]["mean"] for r in runs
                     if r["n_train"] == n and r["physics_weight"] == pw] for pw in pws}
        tab[n] = {}
        for pw in pws:
            tab[n][pw] = {s: ms([r["eval"][s]["summary"]["rel_l2"]["mean"] for r in runs
                                 if r["n_train"] == n and r["physics_weight"] == pw]) for s in SETS}
        p = (stats.ttest_ind(vals[pws[0]], vals[pws[-1]], equal_var=False).pvalue
             if len(pws) > 1 and min(len(v) for v in vals.values()) > 1 else float("nan"))
        md.append(f"| {n} | " + " | ".join(fmt(*tab[n][pw]["test"][:2]) + f" (n={tab[n][pw]['test'][2]})"
                                          for pw in pws) + f" | {p:.3f} |")
    md.append("")
    summary["ablation"] = {str(n): {f"{pw:g}": v for pw, v in d.items()} for n, d in tab.items()}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, s in zip(axes, ("test", "superres_128")):
        for pw, col in zip(pws, (PRIMARY, "#3f9aa6", ACCENT)):
            m = np.array([tab[n][pw][s][0] for n in ns]) * 100
            sd = np.array([tab[n][pw][s][1] for n in ns]) * 100
            ax.errorbar(ns, m, yerr=sd, marker="o", capsize=3, color=col,
                        label="data loss only" if pw == 0 else f"+ physics loss (weight {pw:g})")
        ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
        pct_log_axis(ax, (10, 15, 20, 30, 40, 50))
        ax.set_xlabel("number of labelled training samples"); ax.set_ylabel("relative L2 error [%]")
        ax.set_title(SET_LABELS[s]); ax.legend()
    fig.savefig(fig_dir / "physics_ablation.png"); plt.close(fig)


# ------------------------------------------------------------------ speed
def speed(res, fig_dir, summary, md):
    p = res / "speed.json"
    if not p.exists():
        print("[warn] no speed results"); return
    sp = read_json(p)
    summary["speed"] = sp
    hw = sp["hardware"]
    md.append("## Inference time\n")
    md.append(f"Hardware: {hw.get('gpu', 'no GPU')} / {hw['cpu']} ({hw['cpu_threads']} threads).\n")
    md.append("| Method | setting | ms per sample | speed-up vs solver |")
    md.append("|---|---|---|---|")
    for g, t in sp["solver_s_per_sample"].items():
        md.append(f"| FV solver | CPU, grid {g} | {t * 1e3:.2f} | 1 |")
    for m, d in sp["models"].items():
        for key, v in d.items():
            md.append(f"| {MODEL_LABELS[m]} | {key} | {v['s_per_sample'] * 1e3:.3f} | {v['speedup_vs_solver']:.1f}x |")
    md.append("")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    labels, vals, cols = ["FV solver\n(CPU, 1 thread)"], [sp["solver_s_per_sample"]["64"] * 1e3], [MODEL_COLORS["solver"]]
    dev = hw["device"]
    for m, d in sp["models"].items():
        for b in (1, max(int(k.split("batch")[1]) for k in d)):
            key = f"{dev}_grid64_batch{b}"
            if key in d:
                labels.append(f"{MODEL_LABELS[m]}\n{'GPU' if dev.startswith('cuda') else dev}\nbatch {b}")
                vals.append(d[key]["s_per_sample"] * 1e3)
                cols.append(MODEL_COLORS[m])
    ax.bar(range(len(vals)), vals, color=cols)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.15, f"{v:.2g}", ha="center", fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("time per sample [ms]")
    ax.set_title(f"Inference time per sample, 64x64 grid ({hw.get('gpu', dev)})")
    fig.savefig(fig_dir / "speed.png"); plt.close(fig)


# ------------------------------------------------------------------ optimisation
def optimisation(res, fig_dir, summary, md):
    p = res / "optimization.json"
    if not p.exists():
        print("[warn] no optimisation results"); return
    op = read_json(p)
    des = np.load(res / "optimization_designs.npz")
    summary["optimization"] = op
    md.append("## Design optimisation (true max T from the FV solver)\n")
    md.append(f"Hardware: {op['hardware'].get('gpu', op['hardware']['device'])}. "
              "Best of all starting designs; 'pred. error' = surrogate prediction minus truth for that design.\n")
    md.append("| Case | uniform | solver+adjoint | surrogate | surrogate pred. error | refined | refined pred. error "
              "| time solver / surrogate / refined [s] |")
    md.append("|---" * 8 + "|")
    for r in op["cases"]:
        bs, bb, bc = (int(np.argmin(r[k]["true_max_T"])) for k in ("solver", "surrogate", "refined"))
        sB, sC = r["surrogate"], r["refined"]
        md.append(f"| {r['case']} | {r['uniform_true_max_T']:.4f} | {r['solver']['true_max_T'][bs]:.4f} | "
                  f"{sB['true_max_T'][bb]:.4f} | {100 * (sB['pred_max_T'][bb] / sB['true_max_T'][bb] - 1):+.1f}% | "
                  f"{sC['true_max_T'][bc]:.4f} | {100 * (sC['pred_max_T'][bc] / sC['true_max_T'][bc] - 1):+.1f}% | "
                  f"{r['solver']['wall_time_s']:.1f} / {sB['wall_time_s']:.1f} / {sC['wall_time_s']:.1f} |")
    if op.get("offline_cost"):
        oc = op["offline_cost"]
        be = oc.get("break_even_problems")
        be_txt = (f"break-even after {be:.0f} design problems" if be
                  else "no break-even: the surrogate was not faster per design problem on this hardware")
        md.append(f"\nOffline cost of the surrogate: data generation {oc['data_generation_s']:.0f} s + training "
                  f"{oc['training_s']:.0f} s; {be_txt}.\n")

    for r in op["cases"]:
        c = r["case"]
        tags = ["solver", "surrogate", "refined"]
        titles = {"solver": "Solver + adjoint", "surrogate": "FNO surrogate", "refined": "FNO, refined"}
        fig, axes = plt.subplots(2, 4, figsize=(14, 6.6))
        vmax = des[f"case{c}_uniform_T"].max()
        show_field(axes[0, 0], des[f"case{c}_f"], "heat sources f", cmap="magma")
        show_field(axes[1, 0], des[f"case{c}_uniform_T"], f"uniform design: max T {vmax:.3f}", vmin=0, vmax=vmax)
        draw_sink(axes[1, 0])
        for j, t in enumerate(tags):
            b = int(np.argmin(r[t]["true_max_T"]))
            show_field(axes[0, j + 1], des[f"case{c}_{t}_rho"], f"{titles[t]}: material rho", cmap="Greys",
                       vmin=0, vmax=1)
            txt = f"true max T {r[t]['true_max_T'][b]:.3f}"
            if "pred_max_T" in r[t]:
                txt += f"\n(predicted {r[t]['pred_max_T'][b]:.3f})"
            show_field(axes[1, j + 1], des[f"case{c}_{t}_T"], txt, vmin=0, vmax=vmax)
            draw_sink(axes[1, j + 1])
        fig.suptitle(f"Material layout optimisation, case {c} (volume fraction "
                     f"{op['config']['volume_fraction']:.0%})")
        fig.savefig(fig_dir / f"optimization_case{c}.png"); plt.close(fig)

    # convergence of case 0
    r = op["cases"][0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.array(r["solver"]["objective"])[:, 0], color=MODEL_COLORS["solver"], label="solver + adjoint (true)")
    ax.plot(np.array(r["surrogate"]["objective_pred"])[:, 0], color=PRIMARY, label="FNO (predicted)")
    ax.axhline(r["surrogate"]["true_max_T"][0], color=PRIMARY, ls="--", lw=1, label="FNO design, true max T")
    ax.axhline(r["solver"]["true_max_T"][0], color=GREY, ls="--", lw=1, label="solver design, true max T")
    ax.set_xlabel("iteration"); ax.set_ylabel("p-norm of T / max T"); ax.set_title("Optimisation history (case 0, start 0)")
    ax.legend()
    fig.savefig(fig_dir / "optimization_history.png"); plt.close(fig)


# ------------------------------------------------------------------ FNO diagnostics (experiment 08)
def diagnostics(res, fig_dir, summary, md):
    diag = [read_json(p) for p in sorted(glob.glob(str(res / "diagnostics" / "*.json")))]
    if not diag:
        print("[warn] no diagnostics results"); return
    seeds = sorted({r["seed"] for r in diag})
    groups = {}
    bench = {Path(p).stem: read_json(p) for p in glob.glob(str(res / "benchmark" / "*_seed*.json"))}
    groups["FNO (benchmark)"] = [bench[f"fno_seed{s}"] for s in seeds if f"fno_seed{s}" in bench]
    abl = {Path(p).stem: read_json(p) for p in glob.glob(str(res / "ablation" / "*.json"))}
    groups["FNO + physics loss 0.01"] = [abl[f"fno_n1200_pw0.01_seed{s}"] for s in seeds
                                         if f"fno_n1200_pw0.01_seed{s}" in abl]
    labels = {"fno_small": "FNO small (8 modes, width 24)", "fno_wd": "FNO, weight decay 1e-3",
              "fno_flip_aug": "FNO + mirror augmentation"}
    for v in labels:
        groups[labels[v]] = [r for r in diag if r["variant"] == v]
    groups["U-Net (benchmark)"] = [bench[f"unet_seed{s}"] for s in seeds if f"unet_seed{s}" in bench]
    groups = {k: v for k, v in groups.items() if v}

    md.append("## FNO generalisation diagnostics (experiment 08)\n")
    md.append(f"Seeds {seeds} for every row. 'train' = relative L2 on the last training mini-batches (data term).\n")
    md.append("| Variant | params | train | test | OOD: 4-6 sources | Zero-shot @128 |")
    md.append("|---|---|---|---|---|---|")
    tab = {}
    for name, runs in groups.items():
        tr = ms([r["history"]["train_data"][-1] for r in runs])
        te = ms([r["eval"]["test"]["summary"]["rel_l2"]["mean"] for r in runs])
        oo = ms([r["eval"]["ood_many_sources"]["summary"]["rel_l2"]["mean"] for r in runs])
        sr = ms([r["eval"]["superres_128"]["summary"]["rel_l2"]["mean"] for r in runs])
        tab[name] = {"train": tr, "test": te, "ood_many_sources": oo, "superres_128": sr}
        md.append(f"| {name} | {runs[0]['history']['n_params'] / 1e6:.2f} M | {fmt(*tr[:2])} | {fmt(*te[:2])} | "
                  f"{fmt(*oo[:2])} | {fmt(*sr[:2])} |")
    md.append("")
    summary["diagnostics"] = tab

    fig, ax = plt.subplots(figsize=(10, 4.2))
    names = list(tab)
    x = np.arange(len(names))
    for off, key, col, lab in ((-0.2, "train", GREY, "train"), (0.2, "test", PRIMARY, "test (in-distribution)")):
        ax.bar(x + off, [tab[n][key][0] * 100 for n in names], 0.4, yerr=[tab[n][key][1] * 100 for n in names],
               color=col, label=lab, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("relative L2 error [%]")
    pct_log_axis(ax, (0.5, 1, 2, 5, 10, 20))
    ax.set_title("Train/test gap of FNO variants (mean ± std over seeds)", pad=28)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.savefig(fig_dir / "fno_diagnostics.png"); plt.close(fig)


def main():
    args, cfg = cli(__doc__.splitlines()[1])
    res = Path(cfg["results_dir"])
    fig_dir = REPO_ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    summary, md = {}, ["# Results summary\n", "Generated by experiments/07_make_figures.py from the raw files in "
                       "this folder.\n"]
    for fn in (benchmark, ablation, diagnostics, speed, optimisation):
        fn(res, fig_dir, summary, md)
    (res / "summary").mkdir(parents=True, exist_ok=True)
    (res / "summary" / "RESULTS.md").write_text("\n".join(md))
    write_json(res / "summary" / "summary.json", summary)
    print("\n".join(md))


if __name__ == "__main__":
    main()
