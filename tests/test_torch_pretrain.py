"""Phase 4 on the PyTorch backend: masking rules, training, and exact resume on CPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from leanvocab.pretrain.config import Phase4Config
from leanvocab.pretrain.data import SpecialIds
from leanvocab.pretrain.masking import num_predictions
from tests.test_pretrain import _cfg, _tiny_tokenizer, _write_corpus

torch = pytest.importorskip("torch")

from leanvocab.pretrain.torch_masking import mask_tokens  # noqa: E402
from leanvocab.pretrain.torch_mlm import (  # noqa: E402
    FINAL_DIR,
    MODEL_FILE,
    resolve_device,
    train,
)


def _ids(vocab_size: int = 50) -> SpecialIds:
    return SpecialIds(pad=0, unk=1, cls=2, sep=3, mask=4, random_low=5, vocab_size=vocab_size)


def _torch_cfg(tmp: Path, **overrides: object) -> Phase4Config:
    cfg = _cfg(tmp, **overrides)
    return cfg.model_copy(
        update={
            "backend": "torch",
            "torch": cfg.torch.model_copy(update={"device": "cpu", "amp_dtype": "none"}),
        }
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    _write_corpus(tmp_path / "data")
    _tiny_tokenizer(tmp_path / "toks")
    return tmp_path


def test_mask_tokens_rules() -> None:
    ids = _ids()
    rng = np.random.default_rng(0)
    batch = rng.integers(5, 50, size=(400, 32)).astype(np.int64)
    batch[:, 0] = ids.cls
    batch[:, -1] = ids.sep
    batch[:, 10] = ids.sep  # document separator inside the sequence
    k = num_predictions(32, 0.15)
    assert k == 4
    tensor = torch.from_numpy(batch)
    generator = torch.Generator().manual_seed(0)
    out = mask_tokens(generator, tensor, ids, k, 0.8, 0.1)
    corrupted, positions, labels = (np.asarray(t.numpy()) for t in out)

    assert positions.shape == (400, k)
    assert not np.isin(positions, [0, 10, 31]).any()
    assert all(len(set(row)) == k for row in positions)
    np.testing.assert_array_equal(labels, np.take_along_axis(batch, positions, axis=1))
    untouched = np.ones_like(batch, dtype=bool)
    np.put_along_axis(untouched, positions, False, axis=1)
    np.testing.assert_array_equal(corrupted[untouched], batch[untouched])

    new: np.ndarray = np.take_along_axis(corrupted, positions, axis=1)
    masked = (new == ids.mask).mean()
    kept = (new == labels).mean()
    assert 0.76 < masked < 0.84
    assert 0.07 < kept < 0.15  # 10% kept, plus random draws that hit the label
    assert (new >= ids.random_low).sum() + (new == ids.mask).sum() == new.size

    again = mask_tokens(torch.Generator().manual_seed(0), tensor, ids, k, 0.8, 0.1)[1]
    np.testing.assert_array_equal(again.numpy(), positions)


def test_resolve_device_refuses_silent_cpu_fallback(corpus: Path) -> None:
    cfg = _cfg(corpus).model_copy(update={"backend": "torch"})
    if torch.cuda.is_available():
        assert resolve_device(cfg).type == "cuda"
    else:
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            resolve_device(cfg)


def _final_params(root: Path) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file

    return load_file(str(root / FINAL_DIR / MODEL_FILE))


@pytest.mark.parametrize("arm_id", ["tiny", "tiny-fact"])
def test_training_runs_and_resume_is_exact(corpus: Path, arm_id: str) -> None:
    cfg = _torch_cfg(corpus)
    arm = cfg.arm(arm_id)
    (corpus / "runA").mkdir()
    (corpus / "runB").mkdir()

    full = train(cfg, arm, corpus / "runA", corpus / "ckptA", dry_run_minutes=None)
    steps = full["plan"]["total_steps"]
    assert steps >= 6 and full["plan"]["n_micro"] > 1
    assert full["steps_done"] == steps
    assert full["backend"] == "torch" and full["device"] == "cpu"
    assert full["final_eval"]["bits_per_byte"] > 0
    assert full["bytes_seen"] == pytest.approx(full["plan"]["batch_bytes"] * steps, rel=0.1)
    first_loss = json.loads((corpus / "runA/curve.jsonl").read_text().splitlines()[0])["loss"]
    assert np.isfinite(first_loss)

    partial = train(cfg, arm, corpus / "runB", corpus / "ckptB", None, stop_after_steps=4)
    assert partial["steps_done"] == 4
    resumed = train(cfg, arm, corpus / "runB", corpus / "ckptB", dry_run_minutes=None)
    assert resumed["steps_done"] == steps

    a, b = _final_params(corpus / "ckptA"), _final_params(corpus / "ckptB")
    assert a.keys() == b.keys()
    for name in a:
        np.testing.assert_allclose(a[name], b[name], rtol=1e-5, atol=1e-7, err_msg=name)
    assert resumed["final_eval"]["bits_per_byte"] == pytest.approx(
        full["final_eval"]["bits_per_byte"], rel=1e-5
    )


def test_checkpoint_is_loadable_by_the_mlx_backend(corpus: Path) -> None:
    """Both backends write the same parameter names, so artifacts stay interchangeable."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from leanvocab.model.encoder import BertForMaskedLM as MlxBert

    cfg = _torch_cfg(corpus)
    arm = cfg.arm("tiny")
    (corpus / "runC").mkdir()
    train(cfg, arm, corpus / "runC", corpus / "ckptC", dry_run_minutes=None)

    tok_meta = json.loads((corpus / "toks/tiny/meta.json").read_text())
    mlx_model = MlxBert(cfg.encoder_config(arm, tok_meta["vocab_size"]))
    path = corpus / "ckptC" / FINAL_DIR / MODEL_FILE
    mlx_model.load_weights(str(path))
    loaded = mx.load(str(path))
    assert isinstance(loaded, dict)
    saved = {k: np.array(v) for k, v in loaded.items()}
    in_model = {k: np.array(v) for k, v in tree_flatten(mlx_model.parameters())}
    assert saved.keys() == in_model.keys()
    for name in saved:
        np.testing.assert_array_equal(saved[name], in_model[name], err_msg=name)


def test_dry_run_reports_estimate(corpus: Path) -> None:
    cfg = _torch_cfg(corpus)
    (corpus / "runD").mkdir()
    out = train(cfg, cfg.arm("tiny"), corpus / "runD", corpus / "ckptD", dry_run_minutes=0.02)
    timing = out["dry_run"]
    assert timing["sec_per_step"] > 0
    assert timing["estimated_full_run_sec"] > 0
    assert out["final_eval"]["bits_per_byte"] > 0
    assert not (corpus / "ckptD" / FINAL_DIR).exists()


def test_torch_backend_imports_without_mlx() -> None:
    """The cluster's Linux nodes have no MLX wheels, so the torch path must not need it."""
    script = """
import sys

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "mlx" or name.startswith("mlx."):
            raise ImportError(name)
        return None

sys.meta_path.insert(0, Blocker())
import leanvocab.pretrain.run
from leanvocab.pretrain.torch_mlm import train
assert "mlx" not in sys.modules
print("ok")
"""
    import subprocess

    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
