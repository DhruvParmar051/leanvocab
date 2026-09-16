"""BERT-style (post-LN) encoder and MLM head in MLX.

Parameter names mirror Hugging Face ``BertForMaskedLM`` (``bert.embeddings...``,
``bert.encoder.layer.N...``, ``cls.predictions...``) so weights can be checked against, and
exported to, the PyTorch implementation. With ``embedding_dim`` set, embeddings are
factorized ALBERT-style: word/position/type embeddings live in ``embedding_dim``, are
normalized there and projected to ``hidden_size``; the MLM transform maps back to
``embedding_dim`` so the output layer stays tied to the word embeddings.

Attention-probability dropout is not implemented (fused SDPA has none); hidden dropout is.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from leanvocab.model.config import EncoderConfig

__all__ = ["BertForMaskedLM", "BertModel", "EncoderConfig", "count_parameters", "init_weights"]


@dataclass
class EncoderOutput:
    last_hidden_state: mx.array


class Embeddings(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        dim = cfg.token_dim
        self.word_embeddings = nn.Embedding(cfg.vocab_size, dim)
        self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, dim)
        self.token_type_embeddings = nn.Embedding(cfg.type_vocab_size, dim)
        self.LayerNorm = nn.LayerNorm(dim, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout)
        if cfg.factorized:
            self.embedding_projection = nn.Linear(dim, cfg.hidden_size)

    def __call__(self, input_ids: mx.array, token_type_ids: mx.array | None) -> mx.array:
        seq_len = input_ids.shape[1]
        positions = mx.arange(seq_len)[None, :]
        types = token_type_ids if token_type_ids is not None else mx.zeros_like(input_ids)
        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(positions)
            + self.token_type_embeddings(types)
        )
        x = self.dropout(self.LayerNorm(x))
        if "embedding_projection" in self:
            x = self.embedding_projection(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        if cfg.hidden_size % cfg.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.query = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.key = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.value = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        b, n, _ = x.shape

        def split(t: mx.array) -> mx.array:
            return t.reshape(b, n, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=mask)
        return out.transpose(0, 2, 1, 3).reshape(b, n, -1)


class DenseLayerNorm(nn.Module):
    """``dense`` -> dropout -> residual add -> ``LayerNorm`` (BERT's *Output blocks)."""

    def __init__(self, in_dim: int, cfg: EncoderConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(in_dim, cfg.hidden_size)
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout)

    def __call__(self, x: mx.array, residual: mx.array) -> mx.array:
        return self.LayerNorm(self.dropout(self.dense(x)) + residual)


class Attention(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.self = SelfAttention(cfg)
        self.output = DenseLayerNorm(cfg.hidden_size, cfg)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        return self.output(self.self(x, mask), x)


class Intermediate(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.intermediate_size)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.dense(x))


class Layer(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.attention = Attention(cfg)
        self.intermediate = Intermediate(cfg)
        self.output = DenseLayerNorm(cfg.intermediate_size, cfg)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        h = self.attention(x, mask)
        return self.output(self.intermediate(h), h)


class LayerStack(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.layer = [Layer(cfg) for _ in range(cfg.num_layers)]

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        for layer in self.layer:
            x = layer(x, mask)
        return x


def additive_mask(attention_mask: mx.array, dtype: mx.Dtype) -> mx.array:
    """(B, L) 1/0 padding mask -> (B, 1, 1, L) additive mask for SDPA."""
    neg = mx.array(mx.finfo(dtype).min, dtype=dtype)
    return mx.where(attention_mask[:, None, None, :].astype(mx.bool_), mx.array(0, dtype), neg)


class BertModel(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.embeddings = Embeddings(cfg)
        self.encoder = LayerStack(cfg)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        token_type_ids: mx.array | None = None,
    ) -> EncoderOutput:
        x = self.embeddings(input_ids, token_type_ids)
        mask = None if attention_mask is None else additive_mask(attention_mask, x.dtype)
        return EncoderOutput(last_hidden_state=self.encoder(x, mask))


class PredictionTransform(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.token_dim)
        self.LayerNorm = nn.LayerNorm(cfg.token_dim, eps=cfg.layer_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        return self.LayerNorm(nn.gelu(self.dense(x)))


class Predictions(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.transform = PredictionTransform(cfg)
        self.bias = mx.zeros((cfg.vocab_size,))


class MLMHead(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.predictions = Predictions(cfg)


class BertForMaskedLM(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.config = cfg
        self.bert = BertModel(cfg)
        self.cls = MLMHead(cfg)

    def mlm_logits(self, hidden: mx.array) -> mx.array:
        """Logits for (..., hidden) states, decoder tied to the word embeddings."""
        h = self.cls.predictions.transform(hidden)
        weight = self.bert.embeddings.word_embeddings.weight
        return h @ weight.T + self.cls.predictions.bias

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        token_type_ids: mx.array | None = None,
        positions: mx.array | None = None,
    ) -> mx.array:
        """MLM logits. With ``positions`` (B, k), only those positions are scored."""
        hidden = self.bert(input_ids, attention_mask, token_type_ids).last_hidden_state
        if positions is not None:
            hidden = mx.take_along_axis(hidden, positions[..., None], axis=1)
        return self.mlm_logits(hidden)


def init_weights(model: nn.Module, std: float) -> None:
    """BERT initialization: N(0, std) weights, zero biases, unit LayerNorm."""
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight = mx.random.normal(module.weight.shape, scale=std)
            if "bias" in module:
                module.bias = mx.zeros_like(module.bias)
        elif isinstance(module, nn.Embedding):
            module.weight = mx.random.normal(module.weight.shape, scale=std)
        elif isinstance(module, nn.LayerNorm):
            module.weight = mx.ones_like(module.weight)
            module.bias = mx.zeros_like(module.bias)


def count_parameters(model: nn.Module) -> dict[str, int]:
    from mlx.utils import tree_flatten

    params = dict(tree_flatten(model.parameters()))
    total = sum(int(v.size) for v in params.values())
    head = sum(int(v.size) for k, v in params.items() if k.startswith("cls."))
    word = int(params["bert.embeddings.word_embeddings.weight"].size)
    return {"total": total, "encoder": total - head, "mlm_head": head, "word_embeddings": word}
