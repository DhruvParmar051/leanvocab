"""YAML config loading with strict schema validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ConfigDict

T = TypeVar("T", bound="StrictConfig")


class StrictConfig(BaseModel):
    """Base for all experiment configs: unknown keys are errors, instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def load_config(path: Path, schema: type[T]) -> T:
    """Load ``path`` as YAML and validate it against ``schema``."""
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must be a YAML mapping, got {type(raw).__name__}")
    return schema.model_validate(raw)


def config_hash(config: BaseModel) -> str:
    """SHA256 of the canonical JSON dump of ``config``."""
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
