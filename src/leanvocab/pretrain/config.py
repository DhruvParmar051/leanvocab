"""Phase 4 configuration schema (MLX-free)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from leanvocab.model.config import EncoderConfig
from leanvocab.utils.config import StrictConfig


class ModelShape(StrictConfig):
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    max_position_embeddings: int
    type_vocab_size: int
    hidden_dropout: float = Field(ge=0.0, lt=1.0)
    layer_norm_eps: float = Field(gt=0.0)
    initializer_range: float = Field(gt=0.0)


class OptimConfig(StrictConfig):
    learning_rate: float = Field(gt=0.0)
    warmup_fraction: float = Field(ge=0.0, lt=1.0)
    beta1: float
    beta2: float
    eps: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    grad_clip: float = Field(gt=0.0)


class MLMConfig(StrictConfig):
    mask_prob: float = Field(gt=0.0, lt=1.0)
    replace_mask: float = Field(ge=0.0, le=1.0)
    replace_random: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _fractions(self) -> MLMConfig:
        if self.replace_mask + self.replace_random > 1.0:
            raise ValueError("replace_mask + replace_random must be <= 1")
        return self


class EvalConfig(StrictConfig):
    heldout_bytes: int = Field(gt=0)
    batch_seqs: int = Field(gt=0)
    every_steps: int = Field(gt=0)
    seed: int = Field(ge=0)


class ArmConfig(StrictConfig):
    id: str
    tokenizer: str
    embedding_dim: int | None = None


class DryRunConfig(StrictConfig):
    minutes: float = Field(gt=0.0)
    timing_skip_steps: int = Field(ge=0)


class Phase4Config(StrictConfig):
    seed: int = Field(ge=0)
    tf32: bool
    data_dir: Path
    tokenizer_dir: Path
    pretok_dir: Path
    checkpoint_dir: Path
    special_tokens: list[str]
    budget_bytes: int = Field(gt=0)
    seq_len: int = Field(gt=2)
    batch_bytes: int = Field(gt=0, description="Effective batch size in raw text bytes")
    max_micro_batch_seqs: int = Field(gt=0)
    pretok_batch_docs: int = Field(gt=0)
    log_every_steps: int = Field(gt=0)
    checkpoint_every_steps: int = Field(gt=0)
    keep_checkpoints: int = Field(ge=1)
    model: ModelShape
    optim: OptimConfig
    mlm: MLMConfig
    eval: EvalConfig
    dry_run: DryRunConfig
    arms: list[ArmConfig]

    def arm(self, arm_id: str) -> ArmConfig:
        for arm in self.arms:
            if arm.id == arm_id:
                return arm
        raise KeyError(f"Unknown arm {arm_id!r}; known: {[a.id for a in self.arms]}")

    def encoder_config(self, arm: ArmConfig, vocab_size: int) -> EncoderConfig:
        return EncoderConfig(
            vocab_size=vocab_size,
            embedding_dim=arm.embedding_dim,
            **self.model.model_dump(),
        )

    def training_key(self, arm: ArmConfig) -> str:
        """Identity of a training run: everything that changes the trajectory."""
        shared = self.model_dump(
            mode="json", exclude={"arms", "dry_run", "log_every_steps", "keep_checkpoints"}
        )
        blob: dict[str, Any] = {"shared": shared, "arm": arm.model_dump(mode="json")}
        return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:12]
