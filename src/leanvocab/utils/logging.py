"""Logging setup shared by every entry point."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"


def setup_logging(run_dir: Path | None = None, level: int = logging.INFO) -> None:
    """Configure the root logger to log to stderr and, if given, to ``run_dir/run.log``."""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
