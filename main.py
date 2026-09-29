"""Command-line entry point.

Examples:
    python main.py run                     # train both models, then compare
    python main.py train --model vae --epochs 30
    python main.py compare
    python main.py run --latent-dim 2      # 2-D latent space for direct plots
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

# Only torch-free modules are imported at the top: the device has to be chosen
# before torch initialises CUDA (see autoencoders/device.py).
from autoencoders.config import DataConfig, ExperimentConfig, ModelConfig, PathsConfig, TrainConfig
from autoencoders.device import DEVICE_CHOICES, configure_device

MODEL_CHOICES = ("ae", "vae", "all")


def parse_args() -> argparse.Namespace:
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(description="Autoencoder vs VAE on MNIST")
    parser.add_argument("command", choices=["train", "compare", "run"])
    parser.add_argument("--model", choices=MODEL_CHOICES, default="all", help="model(s) to train")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--epochs", type=int, default=defaults.train.epochs)
    parser.add_argument("--lr", type=float, default=defaults.train.learning_rate)
    parser.add_argument("--batch-size", type=int, default=defaults.data.batch_size)
    parser.add_argument("--latent-dim", type=int, default=defaults.model.latent_dim)
    parser.add_argument("--hidden-dim", type=int, default=defaults.model.hidden_dim)
    parser.add_argument("--beta", type=float, default=defaults.model.beta, help="VAE KL weight")
    parser.add_argument("--seed", type=int, default=defaults.train.seed)
    parser.add_argument("--num-workers", type=int, default=defaults.data.num_workers)
    parser.add_argument("--output-dir", type=Path, default=defaults.paths.output_dir)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    data_defaults = DataConfig()
    return ExperimentConfig(
        data=replace(data_defaults, batch_size=args.batch_size, num_workers=args.num_workers),
        model=ModelConfig(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, beta=args.beta),
        train=TrainConfig(epochs=args.epochs, learning_rate=args.lr, seed=args.seed),
        paths=PathsConfig(output_dir=args.output_dir),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    configure_device(args.device)

    from autoencoders.data import build_loaders
    from autoencoders.evaluation import compare_models
    from autoencoders.evaluation.compare import format_table
    from autoencoders.models import MODEL_NAMES
    from autoencoders.training import train_model
    from autoencoders.utils import get_device, save_json

    config = build_config(args)
    config.paths.make_dirs()
    save_json(config.to_dict(), config.paths.metrics / "config.json")

    loaders = build_loaders(
        config.data, seed=config.train.seed, pin_memory=get_device().type == "cuda"
    )

    if args.command in ("train", "run"):
        names = MODEL_NAMES if args.model == "all" else (args.model,)
        for name in names:
            train_model(name, config, loaders)

    if args.command in ("compare", "run"):
        metrics = compare_models(config, loaders)
        print("\nTest-set comparison\n" + format_table(metrics))


if __name__ == "__main__":
    main()
