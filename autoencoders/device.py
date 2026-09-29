"""Device selection that must run *before* torch initialises CUDA.

`torch.cuda.is_available()` can report True while every kernel launch fails
(e.g. an NVIDIA driver older than the CUDA version torch was built for). Once
torch has seen such a GPU, even CPU training breaks, because optimizers query
the accelerator on every step. So CUDA is probed in a separate process and,
if unusable, hidden from this process via CUDA_VISIBLE_DEVICES.

This module deliberately does not import torch.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

DEVICE_CHOICES = ("auto", "cuda", "cpu")

_PROBE = "import torch, sys; torch.zeros(1, device='cuda'); sys.exit(0)"


def cuda_is_usable() -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-c", _PROBE], capture_output=True, timeout=120, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def configure_device(choice: str) -> None:
    """Hides the GPU from torch unless CUDA was requested or works."""
    if choice not in DEVICE_CHOICES:
        raise ValueError(f"Unknown device '{choice}'. Choose from {DEVICE_CHOICES}.")
    if choice == "cuda":
        return
    if choice == "auto" and cuda_is_usable():
        return
    if choice == "auto":
        logger.warning("CUDA is not usable on this machine; training on CPU.")
    # "-1" rather than "": on Windows assigning an empty string unsets the variable.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
