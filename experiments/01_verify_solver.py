"""Experiment 01 - verify the finite-volume solver.

Outputs
    results/solver_verification.json
    docs/figures/solver_verification.png
Checks
    (a) MMS: RMS error vs grid size, expected slope 2.
    (b) Real partial-sink problem: grid convergence of max(T) (Richardson-style,
        against a 512x512 reference). The sink edges are singular points (the
        boundary condition switches type), so the observed order may be below 2.
    (c) Energy balance on the same problem.
"""
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from thermal_no import solver
from thermal_no.config import REPO_ROOT, cli
from thermal_no.plotting import PRIMARY, ACCENT, apply_style, draw_sink, show_field
from thermal_no.verification import mms_rms_error, smooth_sink_problem


def main():
    args, cfg = cli(__doc__.splitlines()[1])
    out_dir = Path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = REPO_ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    # (a) MMS convergence
    ns = [16, 32, 64, 128, 256]
    mms = [mms_rms_error(n) for n in ns]
    mms_rates = np.log2(np.array(mms[:-1]) / np.array(mms[1:])).tolist()
    print("MMS RMS errors:", mms, "rates:", mms_rates)

    # (b, c) grid convergence of the real problem
    ns_real = [32, 64, 128, 256, 512]
    tmax, balance, times = [], [], []
    fields = {}
    for n in ns_real:
        k, f = smooth_sink_problem(n)
        t0 = time.perf_counter()
        T = solver.solve(k, f)
        times.append(time.perf_counter() - t0)
        tmax.append(float(T.max()))
        q_in, q_out = solver.total_heat_generated(f), solver.sink_heat_flow(T, k)
        balance.append(abs(q_in - q_out) / q_in)
        if n == 64:
            fields = {"k": k, "f": f, "T": T}
    ref = tmax[-1]
    tmax_err = [abs(t - ref) / ref for t in tmax[:-1]]
    tmax_rates = np.log2(np.array(tmax_err[:-1]) / np.array(tmax_err[1:])).tolist()
    print("max(T):", tmax, "rel. err vs 512:", tmax_err, "rates:", tmax_rates)
    print("energy balance rel. error:", balance)
    print("solve times [s]:", dict(zip(ns_real, times)))

    summary = {
        "mms": {"n": ns, "rms_error": mms, "observed_order": mms_rates},
        "partial_sink": {"n": ns_real, "max_T": tmax, "rel_err_vs_512": tmax_err,
                         "observed_order": tmax_rates, "energy_balance_rel_err": balance,
                         "solve_time_s": times},
    }
    (out_dir / "solver_verification.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    ax = axes[0]
    ax.loglog(ns, mms, "o-", color=PRIMARY, label="MMS RMS error")
    ax.loglog(ns, mms[0] * (np.array(ns) / ns[0]) ** -2.0, "--", color="grey", label="slope -2")
    ax.set_xlabel("cells per side N"); ax.set_ylabel("RMS error")
    ax.set_title("(a) Manufactured solution"); ax.legend()
    ax = axes[1]
    ax.loglog(ns_real[:-1], tmax_err, "s-", color=ACCENT, label="|max T - ref| / ref")
    ax.loglog(ns_real[:-1], tmax_err[0] * (np.array(ns_real[:-1]) / ns_real[0]) ** -2.0, "--",
              color="grey", label="slope -2")
    ax.loglog(ns_real[:-1], tmax_err[0] * (np.array(ns_real[:-1]) / ns_real[0]) ** -1.0, ":",
              color="grey", label="slope -1")
    ax.set_xlabel("cells per side N"); ax.set_title("(b) Grid convergence of max T\n(ref: N = 512)")
    ax.legend()
    for a, ticks in ((axes[0], ns), (axes[1], ns_real[:-1])):
        a.set_xticks(ticks); a.set_xticklabels([str(t) for t in ticks]); a.minorticks_off()
    show_field(axes[2], fields["k"], "(c) Test conductivity k", cmap="viridis")
    show_field(axes[3], fields["T"], "(d) Temperature, N = 64")
    draw_sink(axes[3])
    fig.savefig(fig_dir / "solver_verification.png")
    print("saved", fig_dir / "solver_verification.png")


if __name__ == "__main__":
    main()
