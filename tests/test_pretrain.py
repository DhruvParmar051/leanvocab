"""Phase 4 pretraining pieces: masking, byte accounting, batching, and resume determinism."""

from __future__ import annotations

import json
import random
from pathlib import Path

import mlx.core as mx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers

from leanvocab.pretrain.config import Phase4Config
from leanvocab.pretrain.data import (
    SEQ_BYTES_FILE,
    TOKENS_FILE,
    PackedSequences,
    SpecialIds,
    pretokenize_train,
    select_documents,
    special_ids,
    token_bytes,
)
from leanvocab.pretrain.masking import mask_tokens, num_predictions
from leanvocab.pretrain.mlm import FINAL_DIR, MODEL_FILE, plan_batches, train
from leanvocab.tokenizers.common import save_tokenizer

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
LETTERS = "abcdefghij"


def _ids(vocab_size: int = 50) -> SpecialIds:
    return SpecialIds(pad=0, unk=1, cls=2, sep=3, mask=4, random_low=5, vocab_size=vocab_size)


def test_mask_tokens_rules() -> None:
    ids = _ids()
    rng = np.random.default_rng(0)
    batch = rng.integers(5, 50, size=(400, 32)).astype(np.int32)
    batch[:, 0] = ids.cls
    batch[:, -1] = ids.sep
    batch[:, 10] = ids.sep  # document separator inside the sequence
    k = num_predictions(32, 0.15)
    assert k == 4
    corrupted, positions, labels = mask_tokens(mx.random.key(0), mx.array(batch), ids, k, 0.8, 0.1)
    corrupted, positions, labels = np.array(corrupted), np.array(positions), np.array(labels)

    assert positions.shape == (400, k)
    assert not np.isin(positions, [0, 10, 31]).any()
    assert all(len(set(row)) == k for row in positions)
    np.testing.assert_array_equal(labels, np.take_along_axis(batch, positions, axis=1))
    untouched = np.ones_like(batch, dtype=bool)
    np.put_along_axis(untouched, positions, False, axis=1)
    np.testing.assert_array_equal(corrupted[untouched], batch[untouched])

    new = np.take_along_axis(corrupted, positions, axis=1)
    masked = (new == ids.mask).mean()
    kept = (new == labels).mean()
    assert 0.76 < masked < 0.84
    assert 0.07 < kept < 0.15  # 10% kept, plus random draws that hit the label
    assert (new >= ids.random_low).sum() + (new == ids.mask).sum() == new.size

    again = np.array(mask_tokens(mx.random.key(0), mx.array(batch), ids, k, 0.8, 0.1)[1])
    np.testing.assert_array_equal(again, positions)


def test_token_bytes_sum_and_shared_spans() -> None:
    text = "ab é"
    # "ab" | " é" as two byte-fallback pieces sharing the same span
    offsets = np.array([[0, 2], [2, 4], [2, 4]])
    got = token_bytes(text, offsets, len(text.encode()))
    assert got.sum() == pytest.approx(5.0)
    np.testing.assert_allclose(got, [2.0, 1.5, 1.5])
    # WordPiece-style: whitespace not covered by any token is spread proportionally
    got = token_bytes("ab cd", np.array([[0, 2], [3, 5]]), 5)
    np.testing.assert_allclose(got, [2.5, 2.5])


def _write_corpus(root: Path, n_docs: int = 300) -> None:
    rnd = random.Random(0)
    texts = [
        " ".join(
            "".join(rnd.choices(LETTERS, k=rnd.randint(1, 6))) for _ in range(rnd.randint(5, 60))
        )
        for _ in range(n_docs)
    ]
    shards = []
    (root / "shards").mkdir(parents=True)
    for s in range(2):
        part = texts[s::2]
        path = f"shards/train-{s:05d}.parquet"
        pq.write_table(
            pa.table(
                {
                    "id": [f"d{s}-{i}" for i in range(len(part))],
                    "text": part,
                    "n_bytes": [len(t.encode()) for t in part],
                }
            ),
            root / path,
        )
        shards.append({"path": path, "sha256": "x"})
    (root / "heldout").mkdir()
    held = texts[:40]
    pq.write_table(
        pa.table({"text": held, "n_bytes": [len(t.encode()) for t in held]}),
        root / "heldout/heldout.parquet",
    )
    manifest = {
        "train": {"shards": shards},
        "heldout": {"path": "heldout/heldout.parquet", "sha256": "h"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def _tiny_tokenizer(tok_dir: Path) -> Tokenizer:
    vocab = SPECIALS + list(LETTERS) + ["##" + c for c in LETTERS] + ["ab", "##cd", "ef"]
    tok = Tokenizer(models.WordPiece({t: i for i, t in enumerate(vocab)}, unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.decoder = decoders.WordPiece()
    tok.add_special_tokens(SPECIALS)
    save_tokenizer(tok, tok_dir / "tiny", {"special_tokens": SPECIALS})
    return tok


def _cfg(tmp: Path, **overrides: object) -> Phase4Config:
    base: dict[str, object] = {
        "seed": 0,
        "tf32": False,
        "data_dir": tmp / "data",
        "tokenizer_dir": tmp / "toks",
        "pretok_dir": tmp / "pretok",
        "checkpoint_dir": tmp / "ckpt",
        "special_tokens": SPECIALS,
        "budget_bytes": 12_000,
        "seq_len": 32,
        "batch_bytes": 1_000,
        "max_micro_batch_seqs": 4,
        "pretok_batch_docs": 7,
        "log_every_steps": 2,
        "checkpoint_every_steps": 3,
        "keep_checkpoints": 2,
        "model": {
            "hidden_size": 32,
            "num_layers": 2,
            "num_heads": 4,
            "intermediate_size": 64,
            "max_position_embeddings": 32,
            "type_vocab_size": 2,
            "hidden_dropout": 0.1,
            "layer_norm_eps": 1e-12,
            "initializer_range": 0.02,
        },
        "optim": {
            "learning_rate": 1e-3,
            "warmup_fraction": 0.2,
            "beta1": 0.9,
            "beta2": 0.98,
            "eps": 1e-6,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
        },
        "mlm": {"mask_prob": 0.15, "replace_mask": 0.8, "replace_random": 0.1},
        "eval": {"heldout_bytes": 2_000, "batch_seqs": 8, "every_steps": 4, "seed": 7},
        "dry_run": {"minutes": 0.05, "timing_skip_steps": 1},
        "arms": [
            {"id": "tiny", "tokenizer": "tiny"},
            {"id": "tiny-fact", "tokenizer": "tiny", "embedding_dim": 8},
        ],
    }
    base.update(overrides)
    return Phase4Config.model_validate(base)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    _write_corpus(tmp_path / "data")
    _tiny_tokenizer(tmp_path / "toks")
    return tmp_path


def test_pretokenize_packs_documents_and_bytes(corpus: Path) -> None:
    cfg = _cfg(corpus)
    tok = Tokenizer.from_file(str(corpus / "toks/tiny/tokenizer.json"))
    ids = special_ids(tok, SPECIALS)
    plan = select_documents(cfg.data_dir, cfg.budget_bytes)
    assert plan[-1][1] > 0
    out = corpus / "packed"
    meta = pretokenize_train(out, tok, ids, cfg.data_dir, cfg.budget_bytes, 32, 7)
    assert meta["text_bytes"] >= cfg.budget_bytes
    stream = np.fromfile(out / TOKENS_FILE, dtype=np.uint16)
    assert stream.size == meta["stream_tokens"]
    assert (stream == ids.sep).sum() == meta["docs"]
    seq_bytes = np.load(out / SEQ_BYTES_FILE)
    assert seq_bytes.size == meta["sequences"] == stream.size // 30
    assert seq_bytes.sum() <= meta["text_bytes"]
    assert seq_bytes.sum() > meta["text_bytes"] * (1 - 30 / stream.size) - 1e-6

    packed = PackedSequences(out / TOKENS_FILE, 32, ids)
    rows = packed.batch(np.array([0, 3]))
    assert rows.shape == (2, 32)
    assert (rows[:, 0] == ids.cls).all() and (rows[:, -1] == ids.sep).all()
    np.testing.assert_array_equal(rows[1, 1:-1], stream[90:120])


def test_batch_plan_equalizes_bytes(corpus: Path) -> None:
    cfg = _cfg(corpus, batch_bytes=10_000, max_micro_batch_seqs=16)
    dense = plan_batches(cfg, n_seqs=10_000, seq_bytes_total=10_000 * 150.0)
    sparse = plan_batches(cfg, n_seqs=10_000, seq_bytes_total=10_000 * 100.0)
    # targets 66.7 and 100 seqs, split into <=16-seq micro-batches of equal size
    assert (dense.micro_seqs, dense.n_micro) == (13, 5)
    assert (sparse.micro_seqs, sparse.n_micro) == (14, 7)
    assert abs(dense.batch_bytes - 10_000) / 10_000 < 0.08
    assert abs(sparse.batch_bytes - 10_000) / 10_000 < 0.08


def _final_params(root: Path) -> dict[str, np.ndarray]:
    return {k: np.array(v) for k, v in mx.load(str(root / FINAL_DIR / MODEL_FILE)).items()}


@pytest.mark.parametrize("arm_id", ["tiny", "tiny-fact"])
def test_training_runs_and_resume_is_exact(corpus: Path, arm_id: str) -> None:
    cfg = _cfg(corpus)
    arm = cfg.arm(arm_id)
    (corpus / "runA").mkdir()
    (corpus / "runB").mkdir()

    full = train(cfg, arm, corpus / "runA", corpus / "ckptA", dry_run_minutes=None)
    steps = full["plan"]["total_steps"]
    assert steps >= 6 and full["plan"]["n_micro"] > 1
    assert full["steps_done"] == steps
    assert full["final_eval"]["bits_per_byte"] > 0
    assert full["bytes_seen"] == pytest.approx(full["plan"]["batch_bytes"] * steps, rel=0.1)
    first_loss = json.loads((corpus / "runA/curve.jsonl").read_text().splitlines()[0])["loss"]
    assert np.isfinite(first_loss)

    partial = train(cfg, arm, corpus / "runB", corpus / "ckptB", None, stop_after_steps=4)
    assert partial["steps_done"] == 4 and partial["final_eval"] is not None
    resumed = train(cfg, arm, corpus / "runB", corpus / "ckptB", dry_run_minutes=None)
    assert resumed["steps_done"] == steps

    a, b = _final_params(corpus / "ckptA"), _final_params(corpus / "ckptB")
    assert a.keys() == b.keys()
    # Same batches, masks and dropout after resume; GPU kernels in a re-traced graph can
    # differ in the last float bits, so compare at float32 noise level.
    for name in a:
        np.testing.assert_allclose(a[name], b[name], rtol=1e-5, atol=1e-7, err_msg=name)
    assert resumed["final_eval"]["bits_per_byte"] == pytest.approx(
        full["final_eval"]["bits_per_byte"], rel=1e-5
    )


def test_dry_run_reports_estimate(corpus: Path) -> None:
    cfg = _cfg(corpus)
    (corpus / "runC").mkdir()
    out = train(cfg, cfg.arm("tiny"), corpus / "runC", corpus / "ckptC", dry_run_minutes=0.02)
    timing = out["dry_run"]
    assert timing["sec_per_step"] > 0
    assert timing["estimated_full_run_sec"] > 0
    assert out["final_eval"]["bits_per_byte"] > 0
    assert not (corpus / "ckptC" / FINAL_DIR).exists()
