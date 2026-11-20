"""BERT MLM corruption in PyTorch, mirroring :mod:`leanvocab.pretrain.masking`.

Same rule as the MLX version: a fixed ``k`` predictions per packed sequence (so a compiled
graph keeps a static shape), chosen uniformly among maskable positions, then corrupted
80/10/10 into [MASK] / a random token / the original.
"""

from __future__ import annotations

import torch
from torch import Tensor

from leanvocab.pretrain.data import SpecialIds

__all__ = ["mask_tokens"]


def mask_tokens(
    generator: torch.Generator,
    ids: Tensor,
    special: SpecialIds,
    k: int,
    replace_mask: float,
    replace_random: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pick ``k`` maskable positions per row and corrupt them.

    Returns (corrupted ids, positions (B, k), labels (B, k)). [PAD]/[CLS]/[SEP]/[MASK] are
    never selected.
    """
    device = ids.device
    scores = torch.rand(ids.shape, generator=generator, device=device)
    unmaskable = torch.zeros(ids.shape, dtype=torch.bool, device=device)
    for token in special.unmaskable:
        unmaskable |= ids == token
    scores = torch.where(unmaskable, torch.full_like(scores, -1.0), scores)
    positions = scores.topk(k, dim=1).indices
    labels = torch.gather(ids, 1, positions)

    kind = torch.rand(positions.shape, generator=generator, device=device)
    random_ids = torch.randint(
        special.random_low,
        special.vocab_size,
        positions.shape,
        generator=generator,
        device=device,
        dtype=ids.dtype,
    )
    mask_token = torch.full_like(labels, special.mask)
    replacement = torch.where(
        kind < replace_mask,
        mask_token,
        torch.where(kind < replace_mask + replace_random, random_ids, labels),
    )
    corrupted = ids.scatter(1, positions, replacement)
    return corrupted, positions, labels
