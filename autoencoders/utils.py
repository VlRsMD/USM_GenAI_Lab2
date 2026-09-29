"""Small helpers: reproducibility, device selection, checkpoints and JSON I/O."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from autoencoders.config import ModelConfig
from autoencoders.models import EncoderDecoder, build_model

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    # A broken CUDA setup is hidden earlier by autoencoders.device.configure_device.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_path(directory: Path, model_name: str) -> Path:
    return directory / f"{model_name}.pt"


def save_checkpoint(
    path: Path, model_name: str, model: EncoderDecoder, model_config: ModelConfig
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "model_config": vars(model_config),
            "state_dict": model.state_dict(),
        },
        path,
    )
    logger.info("Saved checkpoint to %s", path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[EncoderDecoder, ModelConfig]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Train the model first.")
    payload = torch.load(path, map_location=device, weights_only=True)
    model_config = ModelConfig(**payload["model_config"])
    model = build_model(payload["model_name"], model_config)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), model_config


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
