"""Loss functions. Every loss returns a dict with a differentiable "loss" entry
plus detached components for logging. Values are summed over pixels / latent
dimensions and averaged over the batch, so AE and VAE reconstruction terms are
directly comparable.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F

from autoencoders.models.base import ModelOutput


class LossFn(Protocol):
    def __call__(self, output: ModelOutput, target: torch.Tensor) -> dict[str, torch.Tensor]: ...


def reconstruction_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Bernoulli negative log-likelihood (binary cross-entropy) per image."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return bce.flatten(start_dim=1).sum(dim=1).mean()


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Closed-form KL(N(mu, sigma^2) || N(0, I)) per image."""
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum(dim=1).mean()


class AutoencoderLoss:
    def __call__(self, output: ModelOutput, target: torch.Tensor) -> dict[str, torch.Tensor]:
        recon = reconstruction_loss(output.logits, target)
        return {"loss": recon, "recon": recon.detach()}


class VAELoss:
    """Negative ELBO: reconstruction + beta * KL."""

    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta

    def __call__(self, output: ModelOutput, target: torch.Tensor) -> dict[str, torch.Tensor]:
        if output.mu is None or output.logvar is None:
            raise ValueError("VAELoss requires mu and logvar in the model output.")
        recon = reconstruction_loss(output.logits, target)
        kl = kl_divergence(output.mu, output.logvar)
        return {"loss": recon + self.beta * kl, "recon": recon.detach(), "kl": kl.detach()}


def build_loss(model_name: str, beta: float) -> LossFn:
    if model_name == "ae":
        return AutoencoderLoss()
    if model_name == "vae":
        return VAELoss(beta=beta)
    raise ValueError(f"No loss defined for model '{model_name}'.")
