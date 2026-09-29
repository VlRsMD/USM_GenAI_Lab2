"""Latent-space operations: projection, interpolation, manifold decoding."""

from __future__ import annotations

import torch

from autoencoders.models import EncoderDecoder


def pca_2d(z: torch.Tensor) -> torch.Tensor:
    """Projects latent codes onto their first two principal components."""
    if z.size(1) <= 2:
        return z
    centered = z - z.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(centered, q=2, center=False)
    return centered @ v[:, :2]


@torch.no_grad()
def interpolate(
    model: EncoderDecoder, x_start: torch.Tensor, x_end: torch.Tensor, steps: int
) -> torch.Tensor:
    """Decodes points on the straight line between the codes of two images."""
    model.eval()
    z_start, z_end = model.encode(x_start.unsqueeze(0)), model.encode(x_end.unsqueeze(0))
    alphas = torch.linspace(0, 1, steps, device=z_start.device).unsqueeze(1)
    z = (1 - alphas) * z_start + alphas * z_end
    return torch.sigmoid(model.decode(z))


@torch.no_grad()
def decode_grid(
    model: EncoderDecoder, device: torch.device, grid_size: int = 15, span: float = 3.0
) -> torch.Tensor:
    """Decodes a regular grid over a 2-D latent space (only for latent_dim == 2)."""
    if model.latent_dim != 2:
        raise ValueError("decode_grid requires a 2-dimensional latent space.")
    model.eval()
    axis = torch.linspace(-span, span, grid_size, device=device)
    # Rows go from top (high z2) to bottom (low z2) to match image orientation.
    z2, z1 = torch.meshgrid(axis.flip(0), axis, indexing="ij")
    z = torch.stack([z1.flatten(), z2.flatten()], dim=1)
    return torch.sigmoid(model.decode(z))
