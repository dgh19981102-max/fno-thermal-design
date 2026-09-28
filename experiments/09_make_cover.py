"""Experiment 09 - portfolio cover image (docs/upwork_cover.png, 1800x1150 px) from the real results."""
import glob
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from thermal_no.config import REPO_ROOT, cli
from thermal_no.plotting import FIELD_CMAP, MODEL_COLORS, PRIMARY
from thermal_no.utils import read_json

BG, PILL, TEXT = "#f5f7f9", "#dde7ee", "#1d2a30"


def main():
    args, cfg = cli(__doc__.splitlines()[0])
    res = Path(cfg["results_dir"])
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(18, 11.5), dpi=100, facecolor=BG)

    # ---- title bar
    bar = fig.add_axes([0, 0.78, 1, 0.22]); bar.axis("off")
    bar.add_patch(Rectangle((0, 0), 1, 1, color=PRIMARY, transform=bar.transAxes))
    bar.text(0.035, 0.70, "Neural-Operator Surrogates for Thermal Design", color="white", fontsize=42,
             fontweight="bold", family="DejaVu Serif", va="center", transform=bar.transAxes)
    bar.text(0.035, 0.42, "FNO vs U-Net vs DeepONet against a verified finite-volume solver, and an honest test of "
             "surrogate-based layout optimisation", color="#d6e7ea", fontsize=18, va="center", transform=bar.transAxes)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    x = 0.035
    for tag in ["PYTORCH", "NEURAL OPERATORS", "FINITE VOLUMES", "ADJOINT OPTIMISATION", "PHYSICS-INFORMED ML"]:
        t = bar.text(x, 0.15, tag, color="#2b4a5a", fontsize=12, fontweight="bold", va="center",
                     transform=bar.transAxes,
                     bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.9", fc=PILL, ec="none"))
        bb = t.get_window_extent(renderer).transformed(bar.transAxes.inverted())
        x = bb.x1 + 0.022

    # ---- left: resolution test (paired problems, mean +- std over seeds)
    runs = {m: [read_json(p) for p in glob.glob(str(res / "benchmark" / f"{m}_seed*.json"))]
            for m in ("fno", "unet", "deeponet")}
    ax = fig.add_axes([0.06, 0.13, 0.33, 0.55], facecolor="white")
    labels = {"fno": "FNO", "unet": "U-Net", "deeponet": "DeepONet"}
    xs = np.arange(3)
    for j, (key, lab, alpha) in enumerate((("superres_64", "trained grid 64x64", 0.55),
                                           ("superres_128", "zero-shot 128x128", 1.0))):
        m = [np.mean([r["eval"][key]["summary"]["rel_l2"]["mean"] for r in runs[k]]) * 100 for k in labels]
        s = [np.std([r["eval"][key]["summary"]["rel_l2"]["mean"] for r in runs[k]], ddof=1) * 100 for k in labels]
        bars = ax.bar(xs + (j - 0.5) * 0.38, m, 0.36, yerr=s, capsize=4, alpha=alpha,
                      color=[MODEL_COLORS[k] for k in labels], label=lab, edgecolor="white")
        for b, v, e in zip(bars, m, s):
            ax.text(b.get_x() + b.get_width() / 2, v + e + 0.5, f"{v:.1f}%", ha="center", fontsize=12, color=TEXT)
    ax.set_xticks(xs); ax.set_xticklabels(labels.values(), fontsize=15)
    ax.set_ylabel("relative L2 error [%]  (5 seeds)", fontsize=13)
    ax.set_title("Same problems, two resolutions", fontsize=19, fontweight="bold", color=PRIMARY, pad=14)
    ax.legend(["light: trained grid 64x64", "dark: zero-shot 128x128"], fontsize=12, frameon=False, loc="upper left",
              handlelength=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_ylim(0, 30)
    ax.text(0.0, -0.13, "FNO is resolution-invariant; U-Net is more accurate on its grid but degrades 4x.",
            transform=ax.transAxes, fontsize=13, color=TEXT)

    # ---- right: optimisation, solver vs surrogate (case 0)
    op = read_json(res / "optimization.json")
    des = np.load(res / "optimization_designs.npz")
    r = op["cases"][0]
    fig.text(0.47, 0.705, "Material layout optimisation (30 % high-k material)", fontsize=19, fontweight="bold",
             color=PRIMARY)
    vmax = float(des["case0_uniform_T"].max()) * 0.35
    cols = [("solver", "Solver + adjoint"), ("surrogate", "Optimised on FNO surrogate")]
    for i, (tag, title) in enumerate(cols):
        b = int(np.argmin(r[tag]["true_max_T"]))
        a1 = fig.add_axes([0.47 + i * 0.26, 0.425, 0.2, 0.25])
        a1.imshow(des[f"case0_{tag}_rho"], origin="lower", cmap="Greys", vmin=0, vmax=1)
        a1.set_xticks([]); a1.set_yticks([]); a1.set_title(title, fontsize=15, color=TEXT)
        a2 = fig.add_axes([0.47 + i * 0.26, 0.155, 0.2, 0.25])
        a2.imshow(des[f"case0_{tag}_T"], origin="lower", cmap=FIELD_CMAP, vmin=0, vmax=vmax)
        a2.plot([0.375 * 64 - 0.5, 0.625 * 64 - 0.5], [-0.3, -0.3], color="#00d4ff", lw=5, clip_on=False)
        a2.set_xticks([]); a2.set_yticks([])
        txt = f"true max T = {r[tag]['true_max_T'][b]:.3f}"
        if "pred_max_T" in r[tag]:
            txt += f"\nsurrogate predicted {r[tag]['pred_max_T'][b]:.3f}"
        a2.set_xlabel(txt, fontsize=14, color=TEXT)
    fig.text(0.47, 0.022, f"Uniform layout: max T = {r['uniform_true_max_T']:.2f}. The optimiser exploits surrogate error:\n"
             "surrogate designs are 1.9-2.6x hotter than adjoint designs (3 cases, solver-verified).",
             fontsize=13, color=TEXT)
    out = REPO_ROOT / "docs" / "upwork_cover.png"
    fig.savefig(out, dpi=100, facecolor=BG)
    print("saved", out)


if __name__ == "__main__":
    main()
