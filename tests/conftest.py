from __future__ import annotations

import os

# Exact float32 matmul in tests (MLX enables TF32 on Apple GPUs by default); must be set
# before mlx is first imported.
os.environ["MLX_ENABLE_TF32"] = "0"

from pathlib import Path

import pytest

from leanvocab.data.schema import DataPrepConfig, Document, SourceConfig


def make_config(out_dir: Path, **overrides: object) -> DataPrepConfig:
    base: dict[str, object] = {
        "source": SourceConfig(
            path="fake/corpus",
            name="fake",
            split="train",
            revision="0" * 40,
            file_prefix="fake/",
            read_batch_rows=64,
            disable_xet=True,
            text_field="text",
            id_field="id",
            url_field="url",
            token_count_field="token_count",
        ),
        "seed": 0,
        "target_text_bytes": 10**9,
        "max_docs": None,
        "heldout_bytes": 2_000,
        "heldout_fraction": 0.2,
        "shard_bytes": 5_000,
        "row_group_rows": 16,
        "parquet_compression": "zstd",
        "parquet_compression_level": 3,
        "log_every_docs": 1_000,
        "out_dir": out_dir,
        "raw_dir": out_dir / "raw",
    }
    base.update(overrides)
    return DataPrepConfig.model_validate(base)


def fake_docs(n: int = 400) -> list[Document]:
    docs = []
    for i in range(n):
        # Every 10th doc duplicates doc 0's text; decomposed accents exercise NFC.
        text = "Café number 0 " * 5 if i % 10 == 0 else f"Document {i} body. " * (1 + i % 7)
        docs.append(Document(id=f"doc-{i}", text=text, url=f"https://x/{i}", token_count_gpt2=i))
    return docs


@pytest.fixture
def config_factory():
    return make_config
