"""Fast shape and loss sanity checks (no dataset download needed)."""

import pytest
import torch

from autoencoders.config import ModelConfig
from autoencoders.evaluation.latent import decode_grid, interpolate, pca_2d
from autoencoders.losses import build_loss, kl_divergence
from autoencoders.models import MODEL_NAMES, build_model

CONFIG = ModelConfig(latent_dim=8, hidden_dim=32)


@pytest.fixture
def images() -> torch.Tensor:
    return torch.rand(4, 1, 28, 28)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forward_shapes(name, images):
    model = build_model(name, CONFIG)
    output = model(images)
    assert output.logits.shape == images.shape
    assert output.z.shape == (4, CONFIG.latent_dim)
    assert model.reconstruct(images).shape == images.shape
    assert model.generate(3, torch.device("cpu")).shape == (3, 1, 28, 28)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_loss_is_differentiable(name, images):
    model = build_model(name, CONFIG)
    losses = build_loss(name, beta=1.0)(model(images), images)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_vae_loss_has_kl_term(images):
    model = build_model("vae", CONFIG)
    losses = build_loss("vae", beta=1.0)(model(images), images)
    assert set(losses) == {"loss", "recon", "kl"}


def test_kl_is_zero_for_standard_normal():
    mu, logvar = torch.zeros(5, 3), torch.zeros(5, 3)
    assert kl_divergence(mu, logvar).item() == pytest.approx(0.0)


def test_vae_eval_is_deterministic(images):
    model = build_model("vae", CONFIG).eval()
    assert torch.equal(model(images).logits, model(images).logits)


def test_latent_helpers(images):
    model = build_model("ae", ModelConfig(latent_dim=2, hidden_dim=32))
    assert interpolate(model, images[0], images[1], steps=5).shape == (5, 1, 28, 28)
    assert decode_grid(model, torch.device("cpu"), grid_size=3).shape == (9, 1, 28, 28)
    assert pca_2d(torch.randn(50, 8)).shape == (50, 2)
