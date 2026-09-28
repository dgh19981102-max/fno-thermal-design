"""Small helpers shared by the experiment scripts."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import torch


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def hardware_info(device: str) -> dict:
    info = {"device": device, "torch": torch.__version__, "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine(), "cpu_threads": torch.get_num_threads()}
    if device.startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, indent=1, default=default))


def read_json(path: Path):
    return json.loads(Path(path).read_text())
