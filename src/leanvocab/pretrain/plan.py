"""Batch planning shared by every backend (no framework import).

Fairness across arms comes from the batch being measured in *bytes of raw text*: the number
of sequences per step is ``batch_bytes`` divided by the arm's mean bytes per packed sequence,
so every arm takes about the same number of steps under the same step-based schedule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from leanvocab.pretrain.config import Phase4Config

BITS_PER_NAT = 1.0 / math.log(2.0)


def num_predictions(seq_len: int, mask_prob: float) -> int:
    """Predictions per packed sequence: ``round(mask_prob * (seq_len - 2))``."""
    k = round(mask_prob * (seq_len - 2))
    if k < 1:
        raise ValueError("mask_prob too small for seq_len")
    return k


@dataclass(frozen=True)
class BatchPlan:
    micro_seqs: int
    n_micro: int
    total_steps: int
    warmup_steps: int
    mean_seq_bytes: float

    @property
    def batch_seqs(self) -> int:
        return self.micro_seqs * self.n_micro

    @property
    def batch_bytes(self) -> float:
        return self.batch_seqs * self.mean_seq_bytes


def plan_batches(cfg: Phase4Config, n_seqs: int, seq_bytes_total: float) -> BatchPlan:
    mean_seq_bytes = seq_bytes_total / n_seqs
    target = cfg.batch_bytes / mean_seq_bytes
    n_micro = max(1, math.ceil(target / cfg.max_micro_batch_seqs))
    micro = max(1, round(target / n_micro))
    total = n_seqs // (micro * n_micro)
    if total < 1:
        raise ValueError("Budget smaller than one batch")
    return BatchPlan(
        micro, n_micro, total, round(cfg.optim.warmup_fraction * total), mean_seq_bytes
    )
