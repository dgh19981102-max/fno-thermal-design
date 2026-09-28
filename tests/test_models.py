"""Unit tests for the neural-operator building blocks and training loop."""
import numpy as np
import pytest
import torch

from thermal_no import data as D
from thermal_no import solver
from thermal_no.models import SpectralConv2d, Surrogate, build_model
from thermal_no.physics import physics_loss
from thermal_no.train import train_model

CFG = {"fno": {"modes": 6, "width": 8, "n_layers": 2, "pad_frac": 0.125},
       "unet": {"base": 8, "depth": 3},
       "deeponet": {"p": 16, "width": 32, "branch_res": 32}}


def test_spectral_conv_is_resolution_invariant():
    """A band-limited input sampled on a 32- and a 64-point periodic grid must give
    the same output at the shared grid points (every 2nd fine point)."""
    torch.manual_seed(0)
    layer = SpectralConv2d(2, 3, 5, 5)

    def signal(n):
        s = torch.arange(n) / n
        Y, X = torch.meshgrid(s, s, indexing="ij")
        a = torch.sin(2 * np.pi * 2 * X) * torch.cos(2 * np.pi * 3 * Y) + torch.cos(2 * np.pi * 4 * X)
        b = torch.sin(2 * np.pi * (X + 2 * Y))
        return torch.stack([a, b]).unsqueeze(0)

    with torch.no_grad():
        coarse = layer(signal(32))
        fine = layer(signal(64))
    assert torch.allclose(coarse, fine[..., ::2, ::2], atol=1e-5)


def test_spectral_conv_drops_high_frequencies():
    layer = SpectralConv2d(1, 1, 4, 4)
    n = 32
    s = torch.arange(n) / n
    Y, X = torch.meshgrid(s, s, indexing="ij")
    high = torch.cos(2 * np.pi * 10 * X).reshape(1, 1, n, n)   # frequency 10 > 4 retained modes
    with torch.no_grad():
        assert layer(high).abs().max() < 1e-5


@pytest.mark.parametrize("name", ["fno", "unet", "deeponet"])
@pytest.mark.parametrize("n", [32, 64])
def test_model_shapes(name, n):
    net = build_model(name, CFG[name])
    out = net(torch.randn(3, 2, n, n))
    assert out.shape == (3, 1, n, n)


def test_surrogate_is_differentiable_wrt_k():
    stats = {"logk_mean": 1.0, "logk_std": 1.0, "f_mean": 0.0, "f_std": 1.0, "T_std": 1.0}
    model = Surrogate(build_model("fno", CFG["fno"]), stats)
    k = torch.rand(2, 32, 32) + 1.0
    k.requires_grad_(True)
    model(k, torch.rand(2, 32, 32)).max().backward()
    assert k.grad is not None and torch.isfinite(k.grad).all()


def test_physics_loss_zero_for_exact_solution():
    rng = np.random.default_rng(0)
    cfg = D.InputConfig(n=32)
    k, f = D.sample_conductivity(cfg, rng), D.sample_sources(cfg, rng)
    T = solver.solve(k, f)
    to = lambda a: torch.tensor(a)[None]  # noqa: E731
    assert physics_loss(to(T), to(k), to(f)).item() < 1e-20
    assert physics_loss(to(T * 1.1), to(k), to(f)).item() > 1e-4


@pytest.mark.parametrize("physics_weight", [0.0, 0.001])
def test_training_reduces_validation_error(tmp_path, physics_weight):
    cfg = D.InputConfig(n=32)
    train, val = D.generate(40, cfg, 1, verbose=False), D.generate(10, cfg, 2, verbose=False)
    tcfg = {"steps": 150, "batch_size": 8, "lr": 3e-3, "weight_decay": 0.0, "eval_every": 50}
    model, hist = train_model("fno", CFG["fno"], train, val, tcfg, seed=0,
                              physics_weight=physics_weight, ckpt_path=tmp_path / "ck.pt", log=lambda s: None)
    assert hist["val_rel_l2"][-1] < hist["val_rel_l2"][0]
    assert hist["best_val_rel_l2"] < 0.8
    # resuming a finished run must be a no-op that returns the same best score
    _, hist2 = train_model("fno", CFG["fno"], train, val, tcfg, seed=0, physics_weight=physics_weight,
                           ckpt_path=tmp_path / "ck.pt", log=lambda s: None)
    assert hist2["best_val_rel_l2"] == hist["best_val_rel_l2"]


def test_mirrored_sample_is_exact_solution():
    """Justifies the flip augmentation: x -> 1 - x maps solutions to solutions."""
    rng = np.random.default_rng(3)
    cfg = D.InputConfig(n=32)
    k, f = D.sample_conductivity(cfg, rng), D.sample_sources(cfg, rng)
    T = solver.solve(k, f)
    assert np.allclose(solver.solve(k[:, ::-1].copy(), f[:, ::-1].copy()), T[:, ::-1], atol=1e-12)


def test_training_with_flip_augmentation_runs(tmp_path):
    cfg = D.InputConfig(n=32)
    train, val = D.generate(20, cfg, 1, verbose=False), D.generate(5, cfg, 2, verbose=False)
    tcfg = {"steps": 20, "batch_size": 8, "lr": 3e-3, "weight_decay": 0.0, "eval_every": 10,
            "augment_flip": True}
    _, hist = train_model("fno", CFG["fno"], train, val, tcfg, seed=0, log=lambda s: None)
    assert np.isfinite(hist["best_val_rel_l2"])
