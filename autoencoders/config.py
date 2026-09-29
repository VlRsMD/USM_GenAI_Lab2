"""Central configuration for data, models, training and output paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DataConfig:
    root: Path = PROJECT_ROOT / "data"
    batch_size: int = 128
    num_workers: int = 2
    val_size: int = 5_000


@dataclass(frozen=True)
class ModelConfig:
    latent_dim: int = 16
    hidden_dim: int = 256
    # Weight of the KL term in the VAE loss (beta = 1 is the standard VAE).
    beta: float = 1.0


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42


@dataclass(frozen=True)
class PathsConfig:
    output_dir: Path = PROJECT_ROOT / "outputs"

    @property
    def checkpoints(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def figures(self) -> Path:
        return self.output_dir / "figures"

    @property
    def metrics(self) -> Path:
        return self.output_dir / "metrics"

    def make_dirs(self) -> None:
        for directory in (self.checkpoints, self.figures, self.metrics):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def to_dict(self) -> dict:
        return _stringify_paths(asdict(self))


def _stringify_paths(value):
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value
