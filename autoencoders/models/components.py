"""Convolutional encoder / decoder backbones shared by both models.

The encoder maps a 1x28x28 image to a feature vector; each model adds its own
latent head on top (a single projection for the AE, mean and log-variance for
the VAE). The decoder maps a latent vector back to per-pixel logits.
"""

from __future__ import annotations

import torch
from torch import nn

_FEATURE_SHAPE = (64, 7, 7)
_FEATURE_SIZE = 64 * 7 * 7


class ConvEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 32x14x14
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64x7x7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(_FEATURE_SIZE, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, _FEATURE_SIZE),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, _FEATURE_SHAPE),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x14x14
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 1x28x28
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns logits; apply sigmoid to obtain pixel intensities."""
        return self.net(z)
