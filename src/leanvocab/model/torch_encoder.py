"""BERT-style (post-LN) encoder and MLM head in PyTorch, for CUDA runs.

A parameter-for-parameter mirror of :mod:`leanvocab.model.encoder` (MLX): same module tree,
same Hugging Face ``BertForMaskedLM`` names, same factorization behaviour, same tied output
layer. ``tests/test_torch_encoder_parity.py`` holds the two implementations to the same
outputs, so a CUDA run is comparable with the MLX pilot runs.

As in the MLX version there is no attention-probability dropout (fused SDPA has none).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from leanvocab.model.config import EncoderConfig

__all__ = ["BertForMaskedLM", "BertModel", "EncoderConfig", "count_parameters", "init_weights"]


@dataclass
class EncoderOutput:
    last_hidden_state: Tensor


class Embeddings(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        dim = cfg.token_dim
        self.word_embeddings = nn.Embedding(cfg.vocab_size, dim)
        self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, dim)
        self.token_type_embeddings = nn.Embedding(cfg.type_vocab_size, dim)
        self.LayerNorm = nn.LayerNorm(dim, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout)
        self.embedding_projection = nn.Linear(dim, cfg.hidden_size) if cfg.factorized else None

    def forward(self, input_ids: Tensor, token_type_ids: Tensor | None) -> Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
        types = token_type_ids if token_type_ids is not None else torch.zeros_like(input_ids)
        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(positions)
            + self.token_type_embeddings(types)
        )
        x = self.dropout(self.LayerNorm(x))
        if self.embedding_projection is not None:
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

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        b, n, _ = x.shape

        def split(t: Tensor) -> Tensor:
            return t.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.head_dim**-0.5)
        return out.transpose(1, 2).reshape(b, n, -1)


class DenseLayerNorm(nn.Module):
    """``dense`` -> dropout -> residual add -> ``LayerNorm`` (BERT's *Output blocks)."""

    def __init__(self, in_dim: int, cfg: EncoderConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(in_dim, cfg.hidden_size)
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.hidden_dropout)

    def forward(self, x: Tensor, residual: Tensor) -> Tensor:
        return self.LayerNorm(self.dropout(self.dense(x)) + residual)


class Attention(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.self = SelfAttention(cfg)
        self.output = DenseLayerNorm(cfg.hidden_size, cfg)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        return self.output(self.self(x, mask), x)


class Intermediate(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.dense(x))


class Layer(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.attention = Attention(cfg)
        self.intermediate = Intermediate(cfg)
        self.output = DenseLayerNorm(cfg.intermediate_size, cfg)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        h = self.attention(x, mask)
        return self.output(self.intermediate(h), h)


class LayerStack(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.layer = nn.ModuleList(Layer(cfg) for _ in range(cfg.num_layers))

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        for layer in self.layer:
            x = layer(x, mask)
        return x


def additive_mask(attention_mask: Tensor, dtype: torch.dtype) -> Tensor:
    """(B, L) 1/0 padding mask -> (B, 1, 1, L) additive mask for SDPA."""
    neg = torch.finfo(dtype).min
    keep = attention_mask[:, None, None, :].bool()
    return torch.where(keep, torch.zeros((), dtype=dtype, device=keep.device), neg)


class BertModel(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.embeddings = Embeddings(cfg)
        self.encoder = LayerStack(cfg)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
    ) -> EncoderOutput:
        x = self.embeddings(input_ids, token_type_ids)
        mask = None if attention_mask is None else additive_mask(attention_mask, x.dtype)
        return EncoderOutput(last_hidden_state=self.encoder(x, mask))


class PredictionTransform(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.token_dim)
        self.LayerNorm = nn.LayerNorm(cfg.token_dim, eps=cfg.layer_norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.LayerNorm(F.gelu(self.dense(x)))


class Predictions(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.transform = PredictionTransform(cfg)
        self.bias = nn.Parameter(torch.zeros(cfg.vocab_size))


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

    def mlm_logits(self, hidden: Tensor) -> Tensor:
        """Logits for (..., hidden) states, decoder tied to the word embeddings."""
        h = self.cls.predictions.transform(hidden)
        weight = self.bert.embeddings.word_embeddings.weight
        return h @ weight.T + self.cls.predictions.bias

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        positions: Tensor | None = None,
    ) -> Tensor:
        """MLM logits. With ``positions`` (B, k), only those positions are scored."""
        hidden = self.bert(input_ids, attention_mask, token_type_ids).last_hidden_state
        if positions is not None:
            index = positions[..., None].expand(-1, -1, hidden.shape[-1])
            hidden = torch.gather(hidden, 1, index)
        return self.mlm_logits(hidden)


def init_weights(model: nn.Module, std: float) -> None:
    """BERT initialization: N(0, std) weights, zero biases, unit LayerNorm."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def count_parameters(model: nn.Module) -> dict[str, int]:
    params = dict(model.named_parameters())
    total = sum(p.numel() for p in params.values())
    head = sum(p.numel() for name, p in params.items() if name.startswith("cls."))
    word = params["bert.embeddings.word_embeddings.weight"].numel()
    return {"total": total, "encoder": total - head, "mlm_head": head, "word_embeddings": word}


def decay_parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    """BERT recipe: no weight decay on biases and LayerNorm parameters."""
    decay = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not (name.endswith("bias") or ".LayerNorm." in name)
    ]
    no_decay = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and (name.endswith("bias") or ".LayerNorm." in name)
    ]
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
