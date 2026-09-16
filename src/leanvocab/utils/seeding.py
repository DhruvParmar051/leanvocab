"""Deterministic seeding across the libraries used in the study."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> list[str]:
    """Seed Python, NumPy and (when installed) MLX and torch.

    Returns the names of the libraries that were seeded, so callers can log them.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    seeded = ["python", "numpy"]

    if importlib.util.find_spec("mlx") is not None:
        mx_random = importlib.import_module("mlx.core.random")
        mx_random.seed(seed)
        seeded.append("mlx")

    if importlib.util.find_spec("torch") is not None:
        torch = importlib.import_module("torch")
        torch.manual_seed(seed)
        seeded.append("torch")

    logger.info("Seeded %s with seed=%d", ",".join(seeded), seed)
    return seeded
