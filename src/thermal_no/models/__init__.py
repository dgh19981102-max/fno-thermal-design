"""Model zoo and the physical-units wrapper shared by training, evaluation and optimisation."""
from __future__ import annotations

import torch
import torch.nn as nn

from .deeponet import DeepONet
from .fno import FNO2d, SpectralConv2d
from .unet import UNet

REGISTRY = {"fno": FNO2d, "unet": UNet, "deeponet": DeepONet}


def build_model(name: str, cfg: dict) -> nn.Module:
    """name is 'fno', 'unet' or 'deeponet'; 'pino' is an FNO trained with the physics loss."""
    arch = "fno" if name == "pino" else name
    return REGISTRY[arch](in_ch=2, out_ch=1, **cfg)


def count_params(model: nn.Module) -> int:
    # complex parameters count as two real numbers
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())


class Surrogate(nn.Module):
    """Maps physical (k, f) -> physical T, hiding all normalisation.

    Network input channels: [(log k - mu_k) / s_k, (f - mu_f) / s_f].
    Network output: T / s_T (T is non-negative with a wide range, so we only rescale).
    The statistics are computed on the training set and stored as buffers so
    they are saved with the checkpoint.
    """

    def __init__(self, net: nn.Module, stats: dict):
        super().__init__()
        self.net = net
        for key in ("logk_mean", "logk_std", "f_mean", "f_std", "T_std"):
            self.register_buffer(key, torch.tensor(float(stats[key])))

    @staticmethod
    def compute_stats(k: torch.Tensor, f: torch.Tensor, T: torch.Tensor) -> dict:
        logk = torch.log(k)
        return {"logk_mean": logk.mean().item(), "logk_std": logk.std().item(),
                "f_mean": f.mean().item(), "f_std": f.std().item(), "T_std": T.std().item()}

    def forward(self, k: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """k, f: (B, N, N) physical fields -> T: (B, N, N)."""
        x = torch.stack([(torch.log(k) - self.logk_mean) / self.logk_std,
                         (f - self.f_mean) / self.f_std], dim=1)
        return self.net(x).squeeze(1) * self.T_std


__all__ = ["FNO2d", "SpectralConv2d", "UNet", "DeepONet", "build_model", "count_params", "Surrogate"]
