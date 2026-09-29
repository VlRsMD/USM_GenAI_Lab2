"""Variational autoencoder with a diagonal Gaussian posterior q(z|x)."""

from __future__ import annotations

import torch
from torch import nn

from autoencoders.models.base import EncoderDecoder, ModelOutput
from autoencoders.models.components import ConvDecoder, ConvEncoder


class VariationalAutoencoder(EncoderDecoder):
    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__(latent_dim)
        self.encoder = ConvEncoder(hidden_dim)
        self.to_mu = nn.Linear(self.encoder.output_dim, latent_dim)
        self.to_logvar = nn.Linear(self.encoder.output_dim, latent_dim)
        self.decoder = ConvDecoder(latent_dim, hidden_dim)

    def encode_distribution(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        return self.to_mu(features), self.to_logvar(features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode_distribution(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps keeps sampling differentiable w.r.t. mu and sigma."""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        mu, logvar = self.encode_distribution(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return ModelOutput(logits=self.decode(z), z=z, mu=mu, logvar=logvar)
