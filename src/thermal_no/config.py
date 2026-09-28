"""Load experiment configuration (default.yaml + optional overrides)."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "experiments" / "configs"


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(quick: bool = False, override_file: str | None = None) -> dict:
    cfg = yaml.safe_load((CONFIG_DIR / "default.yaml").read_text())
    if quick:
        cfg = _merge(cfg, yaml.safe_load((CONFIG_DIR / "quick.yaml").read_text()))
    if override_file:
        cfg = _merge(cfg, yaml.safe_load(Path(override_file).read_text()))
    for key in ("data_dir", "results_dir"):
        p = Path(cfg[key])
        cfg[key] = str(p if p.is_absolute() else REPO_ROOT / p)
    # large binary outputs (checkpoints, trained weights) - git-ignored
    cfg.setdefault("artifacts_dir", str(REPO_ROOT / "artifacts" / Path(cfg["results_dir"]).name))
    return cfg


def cli(description: str) -> tuple[argparse.Namespace, dict]:
    """Common command-line interface for the experiment scripts."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test configuration")
    ap.add_argument("--config", default=None, help="extra YAML file with overrides")
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    args = ap.parse_args()
    return args, load_config(args.quick, args.config)
