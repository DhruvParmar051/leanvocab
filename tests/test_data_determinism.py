from __future__ import annotations

import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from leanvocab.data.dedup import ExactDeduper
from leanvocab.data.prepare import prepare
from leanvocab.data.split import heldout_key, is_heldout_candidate
from tests.conftest import fake_docs, make_config


def _read_ids(root: Path, rel: str) -> list[str]:
    return pq.read_table(root / rel, columns=["id"]).column("id").to_pylist()


def test_pipeline_is_byte_identical_across_runs(tmp_path: Path) -> None:
    out = tmp_path / "out"
    m1, _ = prepare(make_config(out), fake_docs())
    first = (out / "manifest.json").read_bytes()
    shutil.rmtree(out)
    m2, _ = prepare(make_config(out), fake_docs())
    assert m1 == m2
    assert m1["train"]["num_shards"] > 1
    assert (out / "manifest.json").read_bytes() == first


def test_split_disjoint_dedup_and_cap(tmp_path: Path) -> None:
    out = tmp_path / "out"
    m, _ = prepare(make_config(out), fake_docs())
    train_ids = [i for s in m["train"]["shards"] for i in _read_ids(out, s["path"])]
    held_ids = _read_ids(out, m["heldout"]["path"])

    assert not set(train_ids) & set(held_ids)
    assert len(train_ids) == len(set(train_ids))
    # 40 docs share doc-0's text; only the first survives.
    assert m["dedup_stats"]["duplicates_removed"] == 39
    assert m["read"]["docs_changed_by_nfc"] == 40
    assert 0 < m["heldout"]["text_bytes"] <= 2_000
    assert m["heldout"]["discarded_candidates"] > 0
    assert not os.access(out / m["heldout"]["path"], os.W_OK)

    texts = pq.read_table(out / m["train"]["shards"][0]["path"]).column("text").to_pylist()
    assert "Café" not in "".join(texts)


def test_discarded_candidates_never_reach_train(tmp_path: Path) -> None:
    out = tmp_path / "out"
    m, _ = prepare(make_config(out), fake_docs())
    train_ids = {i for s in m["train"]["shards"] for i in _read_ids(out, s["path"])}
    for i in range(400):
        if is_heldout_candidate(heldout_key(f"doc-{i}", 0), 0.2):
            assert f"doc-{i}" not in train_ids


def test_budget_and_max_docs_stop(tmp_path: Path) -> None:
    m, _ = prepare(make_config(tmp_path / "a", max_docs=50), fake_docs())
    assert m["read"]["docs"] == 50 and m["stop_reason"] == "max_docs"
    m, _ = prepare(make_config(tmp_path / "b", target_text_bytes=10_000), fake_docs())
    assert m["stop_reason"] == "target_text_bytes"
    assert 10_000 <= m["read"]["text_bytes"] < 10_000 + 200


def test_refuses_to_overwrite(tmp_path: Path) -> None:
    prepare(make_config(tmp_path), fake_docs(50))
    with pytest.raises(FileExistsError):
        prepare(make_config(tmp_path), fake_docs(50))


def test_heldout_assignment_is_order_independent() -> None:
    ids = [f"doc-{i}" for i in range(1000)]
    forward = {i for i in ids if is_heldout_candidate(heldout_key(i, 0), 0.1)}
    backward = {i for i in reversed(ids) if is_heldout_candidate(heldout_key(i, 0), 0.1)}
    assert forward == backward
    assert 60 < len(forward) < 140
    assert forward != {i for i in ids if is_heldout_candidate(heldout_key(i, 1), 0.1)}


def test_deduper_counts() -> None:
    d = ExactDeduper()
    assert [d.seen(b"a"), d.seen(b"b"), d.seen(b"a")] == [False, False, True]
    assert d.stats()["duplicates_removed"] == 1
    assert d.stats()["duplicate_bytes_removed"] == 1
