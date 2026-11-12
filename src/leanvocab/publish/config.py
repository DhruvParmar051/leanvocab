"""Schema for the Hugging Face publishing config."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from leanvocab.utils.config import StrictConfig


class ArmRepo(StrictConfig):
    id: str
    repo: str
    tokenizer: str
    role: str
    bert_compatible: bool


class PublishConfig(StrictConfig):
    owner: str
    private: bool
    license: str
    language: str
    source_dataset: str
    repo_url: str

    checkpoint_dir: Path
    tokenizer_dir: Path
    phase2_results: Path
    phase3_results: Path
    phase4_results: Path
    pretrain_config: Path
    tokenizer_config: Path
    intrinsic_report: Path

    include_optimizer: bool
    model_max_length: int = Field(gt=0)
    hidden_act: str
    attention_dropout: float = Field(ge=0.0, lt=1.0)

    tokenizer_repo: str
    tokenizer_ids: list[str]
    arms: list[ArmRepo]

    def arm(self, arm_id: str) -> ArmRepo:
        for arm in self.arms:
            if arm.id == arm_id:
                return arm
        raise KeyError(f"Unknown arm {arm_id!r}; known: {[a.id for a in self.arms]}")
