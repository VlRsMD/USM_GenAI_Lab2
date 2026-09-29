"""Figures used in the comparison. Every function saves one PNG and closes it."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are only written to disk

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

History = dict[str, list[float]]


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _image_grid(images: torch.Tensor, nrow: int) -> torch.Tensor:
    grid = make_grid(images.detach().cpu(), nrow=nrow, padding=2, pad_value=1.0)
    return grid.permute(1, 2, 0)


def plot_loss_curves(histories: dict[str, History], path: Path) -> None:
    keys = ["loss", "recon", "kl"]
    present = [k for k in keys if any(f"train_{k}" in h for h in histories.values())]
    fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4), squeeze=False)

    for ax, key in zip(axes[0], present):
        # One colour per model; solid = train, dashed = validation.
        for color_index, (name, history) in enumerate(histories.items()):
            if f"train_{key}" not in history:
                continue
            color = f"C{color_index}"
            epochs = range(1, len(history[f"train_{key}"]) + 1)
            ax.plot(epochs, history[f"train_{key}"], color=color, label=f"{name} train")
            ax.plot(epochs, history[f"val_{key}"], color=color, linestyle="--", label=f"{name} val")
        ax.set_title({"loss": "Total loss", "recon": "Reconstruction (BCE)", "kl": "KL divergence"}[key])
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    _save(fig, path)


def plot_reconstructions(
    originals: torch.Tensor, reconstructions: dict[str, torch.Tensor], path: Path
) -> None:
    rows = {"Original": originals, **reconstructions}
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 1.4 * len(rows)))
    for ax, (title, images) in zip(axes, rows.items()):
        ax.imshow(_image_grid(images, nrow=images.size(0)), cmap="gray")
        ax.set_ylabel(title, rotation=0, ha="right", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Test-set reconstructions")
    _save(fig, path)


def plot_samples(samples: dict[str, torch.Tensor], path: Path, nrow: int = 8) -> None:
    fig, axes = plt.subplots(1, len(samples), figsize=(5 * len(samples), 5), squeeze=False)
    for ax, (title, images) in zip(axes[0], samples.items()):
        ax.imshow(_image_grid(images, nrow=nrow), cmap="gray")
        ax.set_title(f"{title}: decode(z), z ~ N(0, I)")
        ax.axis("off")
    _save(fig, path)


def plot_latent_space(
    projections: dict[str, torch.Tensor], labels: torch.Tensor, path: Path, projected: bool
) -> None:
    fig, axes = plt.subplots(1, len(projections), figsize=(6 * len(projections), 5), squeeze=False)
    scatter = None
    for ax, (title, points) in zip(axes[0], projections.items()):
        # vmin/vmax centre each of the 10 colours on its digit's colorbar tick.
        scatter = ax.scatter(
            points[:, 0], points[:, 1], c=labels, cmap="tab10", vmin=-0.5, vmax=9.5, s=3, alpha=0.7
        )
        ax.set_title(f"{title} latent space" + (" (PCA)" if projected else ""))
        ax.set_xlabel("PC 1" if projected else "z1")
        ax.set_ylabel("PC 2" if projected else "z2")
        ax.grid(alpha=0.3)
    if scatter is not None:
        fig.colorbar(scatter, ax=axes[0].tolist(), ticks=range(10), label="Digit")
    _save(fig, path)


def plot_interpolations(rows: dict[str, torch.Tensor], path: Path) -> None:
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 1.5 * len(rows)))
    for ax, (title, images) in zip(axes, rows.items()):
        ax.imshow(_image_grid(images, nrow=images.size(0)), cmap="gray")
        ax.set_ylabel(title, rotation=0, ha="right", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Linear interpolation in latent space")
    _save(fig, path)


def plot_manifold(grids: dict[str, torch.Tensor], grid_size: int, path: Path) -> None:
    fig, axes = plt.subplots(1, len(grids), figsize=(7 * len(grids), 7), squeeze=False)
    for ax, (title, images) in zip(axes[0], grids.items()):
        ax.imshow(_image_grid(images, nrow=grid_size), cmap="gray")
        ax.set_title(f"{title}: decoded 2-D latent grid [-3, 3]^2")
        ax.axis("off")
    _save(fig, path)
