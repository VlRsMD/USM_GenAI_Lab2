"""End-to-end training of one model: data, optimisation, checkpoint, history."""

from __future__ import annotations

import logging

import torch

from autoencoders.config import ExperimentConfig
from autoencoders.data import MnistLoaders
from autoencoders.losses import build_loss
from autoencoders.models import build_model
from autoencoders.training.trainer import History, Trainer
from autoencoders.utils import checkpoint_path, get_device, save_checkpoint, save_json, set_seed

logger = logging.getLogger(__name__)


def train_model(model_name: str, config: ExperimentConfig, loaders: MnistLoaders) -> History:
    set_seed(config.train.seed)
    device = get_device()
    logger.info("Training '%s' on %s", model_name, device)

    model = build_model(model_name, config.model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    trainer = Trainer(model, build_loss(model_name, config.model.beta), optimizer, device)
    history = trainer.fit(loaders.train, loaders.val, config.train.epochs)

    config.paths.make_dirs()
    save_checkpoint(checkpoint_path(config.paths.checkpoints, model_name), model_name, model, config.model)
    save_json(history, config.paths.metrics / f"{model_name}_history.json")
    return history
