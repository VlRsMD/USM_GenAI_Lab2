"""Loads the trained models and produces every comparison metric and figure."""

from __future__ import annotations

import logging

import torch

from autoencoders.config import ExperimentConfig
from autoencoders.data import MnistLoaders
from autoencoders.evaluation.latent import decode_grid, interpolate, pca_2d
from autoencoders.evaluation.metrics import encode_dataset, evaluate_reconstruction, latent_statistics
from autoencoders.models import MODEL_NAMES
from autoencoders.models.factory import DISPLAY_NAMES
from autoencoders.utils import checkpoint_path, get_device, load_checkpoint, load_json, save_json, set_seed
from autoencoders.visualization import (
    plot_interpolations,
    plot_latent_space,
    plot_loss_curves,
    plot_manifold,
    plot_reconstructions,
    plot_samples,
)

logger = logging.getLogger(__name__)

NUM_RECONSTRUCTIONS = 12
NUM_SAMPLES = 64
NUM_LATENT_POINTS = 5_000
INTERPOLATION_STEPS = 12
GRID_SIZE = 15


def compare_models(config: ExperimentConfig, loaders: MnistLoaders) -> dict[str, dict[str, float]]:
    set_seed(config.train.seed)
    device = get_device()
    config.paths.make_dirs()
    figures = config.paths.figures

    models = {
        DISPLAY_NAMES[name]: load_checkpoint(checkpoint_path(config.paths.checkpoints, name), device)[0]
        for name in MODEL_NAMES
    }

    metrics: dict[str, dict[str, float]] = {}
    latents: dict[str, torch.Tensor] = {}
    labels = torch.empty(0)
    for title, model in models.items():
        metrics[title] = evaluate_reconstruction(model, loaders.test, device)
        z, labels = encode_dataset(model, loaders.test, device, max_samples=NUM_LATENT_POINTS)
        latents[title] = z
        metrics[title].update(latent_statistics(z))
    save_json(metrics, config.paths.metrics / "comparison.json")

    histories = {}
    for name in MODEL_NAMES:
        history_file = config.paths.metrics / f"{name}_history.json"
        if history_file.exists():
            histories[DISPLAY_NAMES[name]] = load_json(history_file)
    if histories:
        plot_loss_curves(histories, figures / "loss_curves.png")

    # Reconstructions of the same test images.
    test_images, test_labels = next(iter(loaders.test))
    originals = test_images[:NUM_RECONSTRUCTIONS].to(device)
    plot_reconstructions(
        originals,
        {title: model.reconstruct(originals) for title, model in models.items()},
        figures / "reconstructions.png",
    )

    plot_samples(
        {title: model.generate(NUM_SAMPLES, device) for title, model in models.items()},
        figures / "generated_samples.png",
    )

    latent_dim = next(iter(models.values())).latent_dim
    plot_latent_space(
        {title: pca_2d(z) for title, z in latents.items()},
        labels,
        figures / "latent_space.png",
        projected=latent_dim > 2,
    )

    start, end = _pick_pair(test_images, test_labels)
    plot_interpolations(
        {
            title: interpolate(model, start.to(device), end.to(device), INTERPOLATION_STEPS)
            for title, model in models.items()
        },
        figures / "interpolations.png",
    )

    if latent_dim == 2:
        plot_manifold(
            {title: decode_grid(model, device, GRID_SIZE) for title, model in models.items()},
            GRID_SIZE,
            figures / "latent_manifold.png",
        )

    logger.info("Saved figures to %s", figures)
    return metrics


def _pick_pair(images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """First image of the batch and the first one showing a different digit."""
    other = int((labels != labels[0]).nonzero()[0])
    return images[0], images[other]


def format_table(metrics: dict[str, dict[str, float]]) -> str:
    columns = sorted({key for values in metrics.values() for key in values})
    header = f"{'Model':<14}" + "".join(f"{col:>18}" for col in columns)
    lines = [header, "-" * len(header)]
    for title, values in metrics.items():
        cells = "".join(f"{values[col]:>18.4f}" if col in values else f"{'-':>18}" for col in columns)
        lines.append(f"{title:<14}{cells}")
    return "\n".join(lines)
