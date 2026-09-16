from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from leanvocab.data.schema import DataPrepConfig
from leanvocab.utils.config import config_hash, load_config
from leanvocab.utils.registry import RunContext, write_metrics
from tests.conftest import make_config

REPO = Path(__file__).resolve().parents[1]


def test_write_metrics_and_index(tmp_path: Path) -> None:
    cfg = make_config(tmp_path / "data")
    ctx = RunContext(phase="phaseX", config=cfg, seed=7, results_root=tmp_path / "results")
    out = write_metrics(ctx, {"score": 1.5})

    payload = json.loads(out.read_text())
    assert out == tmp_path / "results/phaseX" / ctx.run_id / "metrics.json"
    assert payload["metrics"] == {"score": 1.5}
    prov = payload["provenance"]
    assert prov["seed"] == 7
    assert prov["config_sha256"] == config_hash(cfg)
    assert {"git_sha", "git_dirty", "libraries", "python"} <= prov.keys()
    assert "numpy" in prov["libraries"]

    lines = (tmp_path / "results/index.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["run_id"] == ctx.run_id

    with pytest.raises(FileExistsError):
        write_metrics(ctx, {"score": 2.0})


def test_config_hash_is_stable_and_sensitive(tmp_path: Path) -> None:
    a = make_config(tmp_path)
    assert config_hash(a) == config_hash(make_config(tmp_path))
    assert config_hash(a) != config_hash(make_config(tmp_path, seed=1))


def test_repo_config_loads_and_rejects_unknown_keys(tmp_path: Path) -> None:
    cfg = load_config(REPO / "configs/data/fineweb_edu_2b.yaml", DataPrepConfig)
    assert cfg.heldout_bytes == 50_000_000
    bad = tmp_path / "bad.yaml"
    bad.write_text((REPO / "configs/data/fineweb_edu_2b.yaml").read_text() + "typo_key: 1\n")
    with pytest.raises(ValidationError):
        load_config(bad, DataPrepConfig)
