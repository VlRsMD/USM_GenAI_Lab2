"""Quantitative comparison of trained models on the test set."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from autoencoders.losses import kl_divergence, reconstruction_loss
from autoencoders.models import EncoderDecoder, VariationalAutoencoder


@torch.no_grad()
def evaluate_reconstruction(
    model: EncoderDecoder, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Reconstruction quality (and KL for the VAE) averaged over images.

    Reconstructions use the deterministic code (z for the AE, mu for the VAE),
    so both models are compared on equal footing.
    """
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    num_samples = 0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        output = model(images)
        recon = torch.sigmoid(output.logits)
        batch_size = images.size(0)

        mse_per_image = F.mse_loss(recon, images, reduction="none").flatten(1).mean(1)
        totals["bce"] += reconstruction_loss(output.logits, images).item() * batch_size
        totals["mse"] += mse_per_image.sum().item()
        totals["psnr_db"] += (10 * torch.log10(1.0 / mse_per_image.clamp_min(1e-10))).sum().item()
        if isinstance(model, VariationalAutoencoder):
            totals["kl"] += kl_divergence(output.mu, output.logvar).item() * batch_size
        num_samples += batch_size

    return {key: value / num_samples for key, value in totals.items()}


@torch.no_grad()
def encode_dataset(
    model: EncoderDecoder, loader: DataLoader, device: torch.device, max_samples: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns latent codes and labels for (a prefix of) the dataset, on CPU."""
    model.eval()
    codes, labels = [], []
    seen = 0
    for images, targets in loader:
        codes.append(model.encode(images.to(device)).cpu())
        labels.append(targets)
        seen += images.size(0)
        if max_samples is not None and seen >= max_samples:
            break
    z, y = torch.cat(codes), torch.cat(labels)
    if max_samples is not None:
        z, y = z[:max_samples], y[:max_samples]
    return z, y


def latent_statistics(z: torch.Tensor) -> dict[str, float]:
    """How close the aggregate latent distribution is to the N(0, I) prior.

    Sampling z ~ N(0, I) only produces meaningful images if the encoded data
    actually occupies that region, which is what the VAE's KL term enforces.
    """
    return {
        "latent_mean_abs": z.mean(dim=0).abs().mean().item(),
        "latent_std_mean": z.std(dim=0).mean().item(),
    }
