"""Experiment 06 - heat-conducting material layout: solver-based vs surrogate-based optimisation.

For each heat-source layout ("case") and the same set of starting designs:
    A  solver     : Adam + exact adjoint gradients from the FV solver (strong baseline)
    B  surrogate  : Adam + autograd through the trained FNO (all starts batched)
    C  refined    : B, then solve the designs B visited with the real solver,
                    fine-tune the FNO on them, and optimise again
Every final design is re-evaluated with the reference solver. We report
    * the TRUE max temperature of each final design (vs the uniform design),
    * the surrogate's predicted max temperature and its error w.r.t. the truth,
    * wall-clock time of each method (C includes the extra solves and fine-tuning),
    * the offline cost of the surrogate (data generation + training) and the
      number of design problems needed to amortise it.
"""
import time
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from thermal_no import data as D
from thermal_no import solver
from thermal_no.config import cli
from thermal_no.optimize import DesignSpace, initial_designs, optimise, solver_T, true_max_T
from thermal_no.train import finetune, load_model
from thermal_no.utils import hardware_info, pick_device, read_json, write_json


def main():
    args, cfg = cli(__doc__.splitlines()[1])
    device = pick_device(args.device)
    data_dir, res, art = Path(cfg["data_dir"]), Path(cfg["results_dir"]), Path(cfg["artifacts_dir"])
    oc = cfg["optimization"]
    n = cfg["data"]["n"]
    space = DesignSpace(oc["volume_fraction"], oc["k_min"], oc["k_max"], oc["simp_p"], oc["filter_radius"])
    model_path = art / "models" / f"fno_seed{oc['surrogate_seed']}.pt"
    model = load_model(model_path, device)
    train = D.load(data_dir / "train.npz")

    src_cfg = replace(D.InputConfig(n=n), n_sources=(3, 3))
    rng = np.random.default_rng(1234)
    cases = [D.sample_sources(src_cfg, rng) for _ in range(oc["n_cases"])]

    results, designs = [], {}
    for c, f_np in enumerate(cases):
        print(f"=== case {c}")
        z0 = initial_designs(n, oc["n_starts"], seed=c)
        uniform_k = space.conductivity(space.density(torch.zeros(1, n, n, dtype=torch.float64))).numpy()
        T_uniform = float(true_max_T(uniform_k, f_np)[0])
        rec = {"case": c, "uniform_true_max_T": T_uniform}
        print(f"uniform design: max T = {T_uniform:.4f}")

        # --- A: solver + adjoint
        f64 = torch.tensor(f_np, dtype=torch.float64)
        rA = optimise(lambda k, f: solver_T(k, f), f64, space, z0.double(), oc["iters"], oc["lr"], oc["pnorm"])
        trueA = true_max_T(rA.k, f_np)
        rec["solver"] = {"true_max_T": trueA, "wall_time_s": rA.wall_time_s, "objective": rA.objective}
        print(f"A solver    : best true max T {trueA.min():.4f}   time {rA.wall_time_s:.1f}s")

        # --- B: surrogate
        f32 = torch.tensor(f_np, dtype=torch.float32, device=device)

        def run_surrogate(m, snapshot_every=0):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            r = optimise(lambda k, f: m(k, f), f32, space, z0.to(device), oc["iters"], oc["lr"], oc["pnorm"],
                         snapshot_every=snapshot_every)
            with torch.no_grad():
                pred = m(torch.as_tensor(r.k, device=device), f32.expand(len(r.k), n, n)).amax((1, 2)).cpu().numpy()
            return r, pred

        rB, predB = run_surrogate(model, snapshot_every=max(1, oc["iters"] // 10))
        trueB = true_max_T(rB.k, f_np)
        rec["surrogate"] = {"true_max_T": trueB, "pred_max_T": predB, "wall_time_s": rB.wall_time_s,
                            "objective_pred": rB.objective}
        print(f"B surrogate : best true max T {trueB.min():.4f}   predicted {predB[np.argmin(trueB)]:.4f}"
              f"   time {rB.wall_time_s:.1f}s")

        # --- C: refine the surrogate around the designs it visited, then re-optimise
        t0 = time.perf_counter()
        K_new = np.concatenate(rB.snapshots + [rB.k])
        sel = np.random.default_rng(c).choice(len(K_new), min(oc["refine_samples"], len(K_new)), replace=False)
        K_new = K_new[sel].astype(np.float64)
        T_new = np.stack([solver.solve(k, f_np) for k in K_new])
        t_solve = time.perf_counter() - t0
        n_old = len(K_new)                                      # replay as many old samples to limit forgetting
        old = np.random.default_rng(c + 1).choice(len(train["k"]), n_old, replace=False)
        k_ft = torch.tensor(np.concatenate([K_new, train["k"][old]]), dtype=torch.float32)
        f_ft = torch.tensor(np.concatenate([np.repeat(f_np[None], n_old, 0), train["f"][old]]), dtype=torch.float32)
        T_ft = torch.tensor(np.concatenate([T_new, train["T"][old]]), dtype=torch.float32)
        t1 = time.perf_counter()
        model_ft = finetune(model, k_ft, f_ft, T_ft, oc["refine_steps"], seed=c)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_ft = time.perf_counter() - t1
        rC, predC = run_surrogate(model_ft)
        trueC = true_max_T(rC.k, f_np)
        rec["refined"] = {"true_max_T": trueC, "pred_max_T": predC,
                          "wall_time_s": rB.wall_time_s + t_solve + t_ft + rC.wall_time_s,
                          "time_breakdown_s": {"first_opt": rB.wall_time_s, "extra_solves": t_solve,
                                               "finetune": t_ft, "second_opt": rC.wall_time_s},
                          "n_extra_solves": len(K_new)}
        print(f"C refined   : best true max T {trueC.min():.4f}   predicted {predC[np.argmin(trueC)]:.4f}"
              f"   time {rec['refined']['wall_time_s']:.1f}s")
        results.append(rec)

        for tag, r, tr in (("solver", rA, trueA), ("surrogate", rB, trueB), ("refined", rC, trueC)):
            b = int(np.argmin(tr))
            designs[f"case{c}_{tag}_rho"] = r.rho[b]
            designs[f"case{c}_{tag}_T"] = solver.solve(r.k[b].astype(np.float64), f_np)
        designs[f"case{c}_f"] = f_np
        designs[f"case{c}_uniform_T"] = solver.solve(uniform_k[0], f_np)

    # --- offline cost of the surrogate and break-even point
    offline = {}
    stats_path, bench_path = res / "dataset_stats.json", res / "benchmark" / f"fno_seed{oc['surrogate_seed']}.json"
    if stats_path.exists() and bench_path.exists():
        st, bench = read_json(stats_path), read_json(bench_path)
        gen_time = (st["train"]["num"] + st["val"]["num"]) * st["train"]["mean_solve_time_s"]
        offline = {"data_generation_s": gen_time, "training_s": bench["history"]["train_time_s"]}
        t_solver = np.mean([r["solver"]["wall_time_s"] for r in results])
        t_sur = np.mean([r["surrogate"]["wall_time_s"] for r in results])
        saved = t_solver - t_sur
        offline["time_saved_per_problem_s"] = saved
        offline["break_even_problems"] = (gen_time + offline["training_s"]) / saved if saved > 0 else None
        print("offline cost:", offline)

    write_json(res / "optimization.json", {"config": oc, "hardware": hardware_info(device),
                                           "cases": results, "offline_cost": offline})
    np.savez_compressed(res / "optimization_designs.npz", **designs)


if __name__ == "__main__":
    main()
