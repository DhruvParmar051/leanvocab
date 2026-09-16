"""Config schema and record type for Phase 1 data preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import Field, model_validator

from leanvocab.utils.config import StrictConfig


class SourceConfig(StrictConfig):
    path: str
    name: str
    split: str
    revision: str = Field(min_length=40, max_length=40, description="Pinned dataset commit SHA")
    file_prefix: str = Field(description="Repo directory whose parquet files form the config")
    read_batch_rows: int = Field(gt=0)
    disable_xet: bool = Field(
        description="Use plain resumable HTTP instead of the Xet transfer backend"
    )
    text_field: str
    id_field: str
    url_field: str
    token_count_field: str


class DataPrepConfig(StrictConfig):
    source: SourceConfig
    seed: int = Field(ge=0)
    target_text_bytes: int = Field(gt=0, description="Stop after this many NFC UTF-8 bytes read")
    max_docs: int | None = Field(default=None, gt=0, description="Dry-run cap on docs read")
    heldout_bytes: int = Field(gt=0)
    heldout_fraction: float = Field(gt=0.0, lt=1.0)
    shard_bytes: int = Field(gt=0)
    row_group_rows: int = Field(gt=0)
    parquet_compression: str
    parquet_compression_level: int
    log_every_docs: int = Field(gt=0)
    out_dir: Path
    raw_dir: Path = Field(description="Local cache of downloaded source parquet files")

    @model_validator(mode="after")
    def _heldout_fits(self) -> DataPrepConfig:
        expected = self.target_text_bytes * self.heldout_fraction
        if expected < self.heldout_bytes:
            raise ValueError(
                f"heldout_fraction={self.heldout_fraction} yields ~{expected:.0f} bytes, "
                f"below heldout_bytes={self.heldout_bytes}; raise the fraction"
            )
        return self


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    text: str
    url: str
    token_count_gpt2: int
