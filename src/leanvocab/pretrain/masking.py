"""BERT MLM corruption with a fixed number of predictions per sequence (compile-friendly)."""

from __future__ import annotations

import mlx.core as mx

from leanvocab.pretrain.data import SpecialIds


def num_predictions(seq_len: int, mask_prob: float) -> int:
    """Predictions per packed sequence: ``round(mask_prob * (seq_len - 2))``."""
    k = round(mask_prob * (seq_len - 2))
    if k < 1:
        raise ValueError("mask_prob too small for seq_len")
    return k


def mask_tokens(
    key: mx.array,
    ids: mx.array,
    special: SpecialIds,
    k: int,
    replace_mask: float,
    replace_random: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """Pick ``k`` maskable positions per row and corrupt them 80/10/10.

    Returns (corrupted ids, positions (B, k), labels (B, k)). [PAD]/[CLS]/[SEP]/[MASK] are
    never selected.
    """
    k_pos, k_kind, k_rand = mx.random.split(key, 3)
    scores = mx.random.uniform(shape=ids.shape, key=k_pos)
    unmaskable = mx.zeros(ids.shape, dtype=mx.bool_)
    for token in special.unmaskable:
        unmaskable = unmaskable | (ids == token)
    scores = mx.where(unmaskable, mx.array(-1.0), scores)
    positions = mx.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    labels = mx.take_along_axis(ids, positions, axis=1)

    kind = mx.random.uniform(shape=positions.shape, key=k_kind)
    random_ids = mx.random.randint(
        special.random_low, special.vocab_size, shape=positions.shape, key=k_rand
    ).astype(ids.dtype)
    replacement = mx.where(
        kind < replace_mask,
        mx.array(special.mask, dtype=ids.dtype),
        mx.where(kind < replace_mask + replace_random, random_ids, labels),
    )
    corrupted = mx.put_along_axis(ids, positions, replacement, axis=1)
    return corrupted, positions, labels
