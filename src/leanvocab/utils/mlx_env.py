"""MLX runtime switches that must be set before ``mlx`` is imported.

MLX 0.32 enables TF32 (TensorFloat-32) matmul on Apple GPUs by default: ~1.9x faster
matmuls at ~1e-3 relative error (float32 without TF32: ~1e-6). Every entry point sets it
explicitly and records the value.
"""

from __future__ import annotations

import os
import sys

TF32_ENV = "MLX_ENABLE_TF32"


def configure_mlx(tf32: bool) -> dict[str, str]:
    value = "1" if tf32 else "0"
    if "mlx.core" in sys.modules and os.environ.get(TF32_ENV) != value:
        raise RuntimeError(
            f"mlx is already imported with {TF32_ENV}={os.environ.get(TF32_ENV)!r}; "
            "call configure_mlx before importing mlx"
        )
    os.environ[TF32_ENV] = value
    return {TF32_ENV: value}
