"""The PyTorch encoder is held to the MLX one: same parameters in, same outputs out.

This is what makes a CUDA run comparable with the MLX pilot runs, so the comparison is on
float32 CPU with TF32 off (``tests/conftest.py``) and dropout disabled.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from leanvocab.model.config import EncoderConfig
from leanvocab.model.encoder import BertForMaskedLM as MlxBert
from leanvocab.model.encoder import count_parameters as mlx_count
from leanvocab.model.encoder import init_weights as mlx_init

torch = pytest.importorskip("torch")

from leanvocab.model.torch_encoder import (  # noqa: E402
    BertForMaskedLM as TorchBert,
)
from leanvocab.model.torch_encoder import (  # noqa: E402
    count_parameters as torch_count,
)
from leanvocab.model.torch_encoder import (  # noqa: E402
    decay_parameter_groups,
    init_weights,
)

SHAPES: list[dict[str, object]] = [{}, {"embedding_dim": 8}]


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


def _paired_models(cfg: EncoderConfig) -> tuple[MlxBert, torch.nn.Module]:
    """An MLX model and a PyTorch model carrying the same weights."""
    mx.random.seed(0)
    mlx_model = MlxBert(cfg)
    # Larger than BERT's 0.02 so differences are not hidden by near-zero activations.
    mlx_init(mlx_model, std=0.2)
    mlx_model.cls.predictions.bias = mx.random.normal((cfg.vocab_size,))
    mlx_model.eval()

    torch_model = TorchBert(cfg).eval()
    weights = {
        k: torch.from_numpy(np.array(v, dtype=np.float32))
        for k, v in tree_flatten(mlx_model.parameters())
    }
    missing, unexpected = torch_model.load_state_dict(weights, strict=True)
    assert not missing and not unexpected
    return mlx_model, torch_model


def _inputs(cfg: EncoderConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    ids = rng.integers(0, cfg.vocab_size, size=(3, 12))
    mask = np.ones_like(ids)
    mask[1, 8:] = 0
    mask[2, 3:] = 0
    types = rng.integers(0, 2, size=ids.shape)
    return ids, mask, types


@pytest.mark.parametrize("shape", SHAPES)
def test_parameter_trees_are_identical(shape: dict[str, object]) -> None:
    cfg = _cfg(**shape)
    mlx_names = {k: tuple(v.shape) for k, v in tree_flatten(MlxBert(cfg).parameters())}
    torch_names = {k: tuple(v.shape) for k, v in TorchBert(cfg).named_parameters()}
    assert mlx_names == torch_names
    assert mlx_count(MlxBert(cfg)) == torch_count(TorchBert(cfg))


@pytest.mark.parametrize("shape", SHAPES)
def test_forward_matches_mlx(shape: dict[str, object]) -> None:
    cfg = _cfg(**shape)
    mlx_model, torch_model = _paired_models(cfg)
    ids, mask, types = _inputs(cfg)
    keep = mask.astype(bool)

    mlx_hidden = np.array(
        mlx_model.bert(mx.array(ids), mx.array(mask), mx.array(types)).last_hidden_state
    )
    mlx_logits = np.array(mlx_model(mx.array(ids), mx.array(mask), mx.array(types)))
    with torch.no_grad():
        t_ids, t_mask, t_types = (torch.from_numpy(a) for a in (ids, mask, types))
        torch_hidden = torch_model.bert(t_ids, t_mask, t_types).last_hidden_state.numpy()
        torch_logits = torch_model(t_ids, t_mask, t_types).numpy()

    np.testing.assert_allclose(torch_hidden[keep], mlx_hidden[keep], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(torch_logits[keep], mlx_logits[keep], rtol=1e-5, atol=1e-5)


def test_position_subset_matches_mlx() -> None:
    cfg = _cfg()
    mlx_model, torch_model = _paired_models(cfg)
    ids, _, _ = _inputs(cfg)
    positions = np.array([[0, 5, 11], [2, 3, 4], [11, 10, 9]])
    mlx_subset = np.array(mlx_model(mx.array(ids), positions=mx.array(positions)))
    with torch.no_grad():
        torch_subset = torch_model(
            torch.from_numpy(ids), positions=torch.from_numpy(positions)
        ).numpy()
        full = torch_model(torch.from_numpy(ids)).numpy()
    expected = np.take_along_axis(full, positions[..., None], axis=1)
    np.testing.assert_allclose(torch_subset, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(torch_subset, mlx_subset, rtol=1e-5, atol=1e-5)


def test_dropout_only_in_train_mode() -> None:
    cfg = _cfg()
    _, model = _paired_models(cfg)
    ids = torch.from_numpy(_inputs(cfg)[0])
    with torch.no_grad():
        a, b = model(ids), model(ids)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        model.train()
        c, d = model(ids), model(ids)
    assert not torch.allclose(c, d)


def test_init_weights_zeroes_bias_and_sets_layernorm() -> None:
    cfg = _cfg()
    model = TorchBert(cfg)
    init_weights(model, std=0.2)
    for name, p in model.named_parameters():
        if name.endswith("bias") and ".LayerNorm." not in name and not name.startswith("cls."):
            torch.testing.assert_close(p, torch.zeros_like(p))
        elif name.endswith(".LayerNorm.weight"):
            torch.testing.assert_close(p, torch.ones_like(p))
    std = model.bert.encoder.layer[0].intermediate.dense.weight.std().item()
    assert 0.15 < std < 0.25


def test_decay_groups_exclude_bias_and_layernorm() -> None:
    cfg = _cfg()
    model = TorchBert(cfg)
    groups = decay_parameter_groups(model, weight_decay=0.01)
    sizes = [sum(p.numel() for p in g["params"]) for g in groups]
    assert sum(sizes) == sum(p.numel() for p in model.parameters())
    expected_no_decay = sum(
        p.numel()
        for name, p in model.named_parameters()
        if name.endswith("bias") or ".LayerNorm." in name
    )
    assert sizes[1] == expected_no_decay
    assert groups[0]["weight_decay"] == 0.01 and groups[1]["weight_decay"] == 0.0
