"""Model-agnostic training loop with per-epoch validation and history."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from autoencoders.losses import LossFn
from autoencoders.models import EncoderDecoder

logger = logging.getLogger(__name__)

History = dict[str, list[float]]


class Trainer:
    def __init__(
        self,
        model: EncoderDecoder,
        loss_fn: LossFn,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> History:
        history: History = defaultdict(list)
        best_val = float("inf")
        best_state: dict[str, torch.Tensor] | None = None

        for epoch in range(1, epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True, desc=f"Epoch {epoch}/{epochs}")
            val_metrics = self.evaluate(val_loader)

            for key, value in train_metrics.items():
                history[f"train_{key}"].append(value)
            for key, value in val_metrics.items():
                history[f"val_{key}"].append(value)

            logger.info(
                "Epoch %d/%d | %s | %s",
                epoch,
                epochs,
                _format("train", train_metrics),
                _format("val", val_metrics),
            )

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        # Keep the weights with the best validation loss.
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return dict(history)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        return self._run_epoch(loader, train=False)

    def _run_epoch(self, loader: DataLoader, train: bool, desc: str | None = None) -> dict[str, float]:
        self.model.train(train)
        totals: dict[str, float] = defaultdict(float)
        num_samples = 0

        batches: Iterable = tqdm(loader, desc=desc, leave=False) if desc else loader
        with torch.set_grad_enabled(train):
            for images, _ in batches:
                images = images.to(self.device, non_blocking=True)
                losses = self.loss_fn(self.model(images), images)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    losses["loss"].backward()
                    self.optimizer.step()

                batch_size = images.size(0)
                num_samples += batch_size
                for key, value in losses.items():
                    totals[key] += value.item() * batch_size

        return {key: value / num_samples for key, value in totals.items()}


def _format(prefix: str, metrics: dict[str, float]) -> str:
    return " ".join(f"{prefix}_{key}={value:.2f}" for key, value in metrics.items())
