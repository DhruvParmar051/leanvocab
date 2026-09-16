"""MLX encoder checked against the PyTorch reference (HF BertForMaskedLM) on CPU."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from leanvocab.model.encoder import (
    BertForMaskedLM,
    EncoderConfig,
    count_parameters,
    init_weights,
)

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _cfg(**overrides: object) -> EncoderConfig:
    base: dict[str, object] = {
        "vocab_size": 97,
        "hidden_size": 32,
        "num_layers": 2,
        "num_heads": 4,
        "intermediate_size": 64,
        "max_position_embeddings": 16,
        "type_vocab_size": 2,
        "hidden_dropout": 0.1,
        "layer_norm_eps": 1e-12,
        "initializer_range": 0.02,
    }
    base.update(overrides)
    return EncoderConfig.model_validate(base)


def _mlx_model(cfg: EncoderConfig, seed: int = 0) -> BertForMaskedLM:
    mx.random.seed(seed)
    model = BertForMaskedLM(cfg)
    # Larger than BERT's 0.02 so differences are not hidden by near-zero activations.
    init_weights(model, std=0.2)
    model.cls.predictions.bias = mx.random.normal((cfg.vocab_size,))
    model.eval()
    return model


def _hf_model(cfg: EncoderConfig, mlx_model: BertForMaskedLM):  # type: ignore[no-untyped-def]
    hf_cfg = transformers.BertConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_layers,
        num_attention_heads=cfg.num_heads,
        intermediate_size=cfg.intermediate_size,
        max_position_embeddings=cfg.max_position_embeddings,
        type_vocab_size=cfg.type_vocab_size,
        hidden_act="gelu",
        layer_norm_eps=cfg.layer_norm_eps,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        attn_implementation="eager",
    )
    hf = transformers.BertForMaskedLM(hf_cfg).eval()
    weights = {
        k: torch.from_numpy(np.array(v, dtype=np.float32))
        for k, v in tree_flatten(mlx_model.parameters())
    }
    weights["cls.predictions.decoder.weight"] = weights["bert.embeddings.word_embeddings.weight"]
    missing, unexpected = hf.load_state_dict(weights, strict=False)
    assert not unexpected, unexpected
    assert all("position_ids" in k or k == "cls.predictions.decoder.bias" for k in missing), missing
    with torch.no_grad():
        hf.cls.predictions.decoder.bias.copy_(weights["cls.predictions.bias"])
    return hf


def _inputs(cfg: EncoderConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    ids = rng.integers(0, cfg.vocab_size, size=(3, 12))
    mask = np.ones_like(ids)
    mask[1, 8:] = 0
    mask[2, 3:] = 0
    types = rng.integers(0, 2, size=ids.shape)
    return ids, mask, types


def test_matches_hf_bert_for_masked_lm() -> None:
    cfg = _cfg()
    model = _mlx_model(cfg)
    hf = _hf_model(cfg, model)
    ids, mask, types = _inputs(cfg)

    ours_hidden = np.array(
        model.bert(mx.array(ids), mx.array(mask), mx.array(types)).last_hidden_state
    )
    ours_logits = np.array(model(mx.array(ids), mx.array(mask), mx.array(types)))
    with torch.no_grad():
        out = hf(
            input_ids=torch.from_numpy(ids),
            attention_mask=torch.from_numpy(mask),
            token_type_ids=torch.from_numpy(types),
            output_hidden_states=True,
        )
    keep = mask.astype(bool)
    np.testing.assert_allclose(
        ours_hidden[keep], out.hidden_states[-1].numpy()[keep], rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(ours_logits[keep], out.logits.numpy()[keep], rtol=1e-4, atol=1e-4)


def test_position_subset_equals_full_logits() -> None:
    cfg = _cfg()
    model = _mlx_model(cfg)
    ids, _, _ = _inputs(cfg)
    full = np.array(model(mx.array(ids)))
    positions = np.array([[0, 5, 11], [2, 3, 4], [11, 10, 9]])
    subset = np.array(model(mx.array(ids), positions=mx.array(positions)))
    expected = np.take_along_axis(full, positions[..., None], axis=1)
    np.testing.assert_allclose(subset, expected, rtol=1e-5, atol=1e-5)


def test_factorized_embeddings_and_tied_head() -> None:
    cfg = _cfg(embedding_dim=8)
    model = _mlx_model(cfg)
    emb = model.bert.embeddings
    ids = np.array([[1, 2, 3]])
    got = np.array(emb(mx.array(ids), None))

    word = np.array(emb.word_embeddings.weight)[ids]
    pos = np.array(emb.position_embeddings.weight)[None, :3]
    typ = np.array(emb.token_type_embeddings.weight)[0][None, None]
    x = word + pos + typ
    x = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + cfg.layer_norm_eps)
    x = x * np.array(emb.LayerNorm.weight) + np.array(emb.LayerNorm.bias)
    proj = emb.embedding_projection
    expected = x @ np.array(proj.weight).T + np.array(proj.bias)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)

    assert model(mx.array(ids)).shape == (1, 3, cfg.vocab_size)
    assert model.cls.predictions.transform.dense.weight.shape == (8, cfg.hidden_size)
    counts = count_parameters(model)
    assert counts["word_embeddings"] == cfg.vocab_size * 8


def test_full_size_parameter_count_matches_phase3() -> None:
    cfg = _cfg(
        vocab_size=30_522,
        hidden_size=384,
        num_layers=6,
        num_heads=12,
        intermediate_size=1536,
        max_position_embeddings=256,
    )
    counts = count_parameters(BertForMaskedLM(cfg))
    assert counts["encoder"] == 22_467_072
    # MLM head: transform dense + LayerNorm + per-token output bias (decoder is tied)
    assert counts["mlm_head"] == 384 * 384 + 384 + 2 * 384 + 30_522


def test_dropout_only_in_train_mode() -> None:
    cfg = _cfg()
    model = _mlx_model(cfg)
    ids = mx.array(_inputs(cfg)[0])
    a, b = np.array(model(ids)), np.array(model(ids))
    np.testing.assert_array_equal(a, b)
    model.train()
    c, d = np.array(model(ids)), np.array(model(ids))
    assert not np.allclose(c, d)
