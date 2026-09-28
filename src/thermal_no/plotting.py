"""Shared matplotlib style so every figure in docs/ looks consistent."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PRIMARY = "#154f58"   # deep teal (matches docs/index.html header)
ACCENT = "#d9822b"    # warm orange for contrast
GREY = "#8a9aa0"
MODEL_COLORS = {"fno": PRIMARY, "pino": "#3f9aa6", "unet": ACCENT, "deeponet": "#7a5c99",
                "solver": "#333333"}
MODEL_LABELS = {"fno": "FNO", "pino": "FNO + physics loss", "unet": "U-Net",
                "deeponet": "DeepONet", "solver": "FV solver"}
FIELD_CMAP = "inferno"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    })


def show_field(ax, a, title=None, cmap=FIELD_CMAP, vmin=None, vmax=None, colorbar=True):
    """Imshow with y pointing up (row 0 = bottom edge = sink side)."""
    im = ax.imshow(a, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    if title:
        ax.set_title(title)
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    return im


def draw_sink(ax, sink_width: float = 0.25) -> None:
    """Mark the heat-sink segment on the bottom edge."""
    ax.plot([0.5 - sink_width / 2, 0.5 + sink_width / 2], [0.004, 0.004],
            color="#00d4ff", lw=4, solid_capstyle="butt", clip_on=False)
