"""Classic (deterministic) autoencoder."""

from __future__ import annotations

import torch
from torch import nn

from autoencoders.models.base import EncoderDecoder, ModelOutput
from autoencoders.models.components import ConvDecoder, ConvEncoder


class Autoencoder(EncoderDecoder):
    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__(latent_dim)
        self.encoder = ConvEncoder(hidden_dim)
        self.to_latent = nn.Linear(self.encoder.output_dim, latent_dim)
        self.decoder = ConvDecoder(latent_dim, hidden_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        z = self.encode(x)
        return ModelOutput(logits=self.decode(z), z=z)
