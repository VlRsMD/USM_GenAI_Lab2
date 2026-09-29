"""Common interface for encoder-decoder models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelOutput:
    logits: torch.Tensor
    z: torch.Tensor
    mu: torch.Tensor | None = None
    logvar: torch.Tensor | None = None


class EncoderDecoder(nn.Module, ABC):
    """Base class: encode images to a latent code and decode it back."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic latent representation used for analysis (z or mu)."""

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Maps latent vectors to pixel logits."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> ModelOutput: ...

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decode(self.encode(x)))

    @torch.no_grad()
    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Decodes latent vectors drawn from the prior N(0, I)."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return torch.sigmoid(self.decode(z))
