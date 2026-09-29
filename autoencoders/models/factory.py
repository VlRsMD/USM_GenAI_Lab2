"""Builds models by name so the CLI and training code stay model-agnostic."""

from __future__ import annotations

from autoencoders.config import ModelConfig
from autoencoders.models.autoencoder import Autoencoder
from autoencoders.models.base import EncoderDecoder
from autoencoders.models.vae import VariationalAutoencoder

_REGISTRY: dict[str, type[EncoderDecoder]] = {
    "ae": Autoencoder,
    "vae": VariationalAutoencoder,
}

MODEL_NAMES: tuple[str, ...] = tuple(_REGISTRY)

DISPLAY_NAMES = {"ae": "Autoencoder", "vae": "VAE"}


def build_model(name: str, config: ModelConfig) -> EncoderDecoder:
    try:
        model_cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.") from None
    return model_cls(latent_dim=config.latent_dim, hidden_dim=config.hidden_dim)
