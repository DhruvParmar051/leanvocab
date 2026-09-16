"""MLM pretraining loop (MLX). Import through ``leanvocab.pretrain.run`` so MLX is configured.

Fairness across arms:
- same documents (``budget_bytes`` of raw text) and one pass over them;
- the same effective batch size in *bytes*: sequences per step = ``batch_bytes`` divided by
  the arm's mean bytes per packed sequence, so every arm takes ~the same number of steps and
  uses the same step-based learning-rate schedule;
- the same optimizer, schedule, masking rate and seed.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from leanvocab.model.encoder import BertForMaskedLM, count_parameters, init_weights
from leanvocab.pretrain.config import ArmConfig, Phase4Config
from leanvocab.pretrain.data import (
    EVAL_BYTES_FILE,
    EVAL_TOKENS_FILE,
    SEQ_BYTES_FILE,
    TOKENS_FILE,
    PackedSequences,
    SpecialIds,
    prepare_arm_data,
)
from leanvocab.pretrain.masking import mask_tokens, num_predictions

logger = logging.getLogger(__name__)

STATE_FILE = "state.json"
MODEL_FILE = "model.safetensors"
OPTIMIZER_FILE = "optimizer.safetensors"
FINAL_DIR = "final"
BITS_PER_NAT = 1.0 / math.log(2.0)


def _scalar(a: mx.array) -> float:
    return float(np.asarray(a))


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


def build_optimizer(cfg: Phase4Config, plan: BatchPlan) -> optim.MultiOptimizer:
    o = cfg.optim
    warmup = max(plan.warmup_steps, 1)
    schedule = optim.join_schedules(
        [
            optim.linear_schedule(0.0, o.learning_rate, warmup),
            optim.linear_schedule(o.learning_rate, 0.0, max(plan.total_steps - warmup, 1)),
        ],
        [warmup],
    )

    def make(weight_decay: float) -> optim.AdamW:
        return optim.AdamW(
            learning_rate=schedule,
            betas=[o.beta1, o.beta2],
            eps=o.eps,
            weight_decay=weight_decay,
        )

    def decayed(path: str, _: mx.array) -> bool:
        # BERT recipe: no weight decay on biases and LayerNorm parameters.
        return not (path.endswith("bias") or ".LayerNorm." in path)

    return optim.MultiOptimizer([make(o.weight_decay), make(0.0)], [decayed])


class Checkpoints:
    def __init__(self, root: Path, keep: int) -> None:
        self.root = root
        self.keep = keep

    def _steps(self) -> list[Path]:
        if not self.root.exists():
            return []
        done = [p for p in self.root.glob("step_*") if (p / STATE_FILE).exists()]
        return sorted(done, key=lambda p: int(p.name.split("_")[1]))

    def latest(self) -> Path | None:
        steps = self._steps()
        return steps[-1] if steps else None

    def save(
        self, name: str, model: nn.Module, optimizer: optim.Optimizer, state: dict[str, Any]
    ) -> Path:
        final = self.root / name
        tmp = self.root / f"{name}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        mx.save_safetensors(str(tmp / MODEL_FILE), dict(tree_flatten(model.parameters())))
        mx.save_safetensors(str(tmp / OPTIMIZER_FILE), dict(tree_flatten(optimizer.state)))
        (tmp / STATE_FILE).write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)
        for old in self._steps()[: -self.keep]:
            shutil.rmtree(old)
        return final

    @staticmethod
    def load(path: Path, model: nn.Module, optimizer: optim.Optimizer) -> dict[str, Any]:
        model.load_weights(str(path / MODEL_FILE))
        loaded = mx.load(str(path / OPTIMIZER_FILE))
        if not isinstance(loaded, dict):
            raise TypeError(f"Unexpected optimizer checkpoint format in {path}")
        optimizer.state = tree_unflatten(list(loaded.items()))
        state: dict[str, Any] = json.loads((path / STATE_FILE).read_text())
        return state


def evaluate(
    model: BertForMaskedLM, data_dir: Path, ids: SpecialIds, cfg: Phase4Config, k: int
) -> dict[str, float]:
    """Held-out MLM loss and bits per byte with fixed masks (dropout off)."""
    token_bytes = np.fromfile(data_dir / EVAL_BYTES_FILE, dtype=np.float32)
    packed = PackedSequences(data_dir / EVAL_TOKENS_FILE, cfg.seq_len, ids)
    body = cfg.seq_len - 2
    was_training = model.training
    model.eval()
    nll_sum = bytes_sum = correct = count = 0.0
    try:
        for b, start in enumerate(range(0, len(packed), cfg.eval.batch_seqs)):
            index = np.arange(start, min(start + cfg.eval.batch_seqs, len(packed)))
            batch = mx.array(packed.batch(index))
            key = mx.random.key(cfg.eval.seed + b)
            corrupted, positions, labels = mask_tokens(
                key, batch, ids, k, cfg.mlm.replace_mask, cfg.mlm.replace_random
            )
            logits = model(corrupted, positions=positions)
            nll = nn.losses.cross_entropy(logits, labels, reduction="none")
            hits = (logits.argmax(axis=-1) == labels).astype(mx.float32)
            # Bytes of the masked tokens: position p in a row maps to stream offset p - 1.
            pos = np.array(positions).astype(np.int64)
            stream_idx = index[:, None] * body + (pos - 1)
            masked_bytes = token_bytes[stream_idx].astype(np.float64)
            mx.eval(nll, hits)
            nll_sum += _scalar(nll.sum())
            correct += _scalar(hits.sum())
            count += float(nll.size)
            bytes_sum += float(masked_bytes.sum())
    finally:
        if was_training:
            model.train()
    return {
        "mlm_loss": nll_sum / count,
        "mlm_accuracy": correct / count,
        "bits_per_byte": nll_sum * BITS_PER_NAT / bytes_sum,
        "masked_tokens": count,
        "masked_bytes": bytes_sum,
    }


def train(
    cfg: Phase4Config,
    arm: ArmConfig,
    run_dir: Path,
    checkpoint_root: Path,
    dry_run_minutes: float | None,
    stop_after_steps: int | None = None,
) -> dict[str, Any]:
    """Train one arm. ``stop_after_steps`` checkpoints and returns early (for resume tests)."""
    data_dir, meta = prepare_arm_data(
        cfg.pretok_dir,
        cfg.tokenizer_dir,
        arm.tokenizer,
        cfg.special_tokens,
        cfg.data_dir,
        cfg.budget_bytes,
        cfg.seq_len,
        cfg.pretok_batch_docs,
        cfg.eval.heldout_bytes,
    )
    ids = SpecialIds(**meta["special_ids"])
    train_seqs = PackedSequences(data_dir / TOKENS_FILE, cfg.seq_len, ids)
    seq_bytes = np.load(data_dir / SEQ_BYTES_FILE)
    if seq_bytes.size != len(train_seqs):
        raise ValueError("Sequence byte table does not match the token stream")
    plan = plan_batches(cfg, len(train_seqs), float(seq_bytes.sum()))
    k = num_predictions(cfg.seq_len, cfg.mlm.mask_prob)
    tokens_per_step = plan.batch_seqs * cfg.seq_len
    logger.info(
        "%s: %d seqs, batch %d x %d = %d seqs (%.0f bytes), %d steps, warmup %d, k=%d",
        arm.id,
        len(train_seqs),
        plan.micro_seqs,
        plan.n_micro,
        plan.batch_seqs,
        plan.batch_bytes,
        plan.total_steps,
        plan.warmup_steps,
        k,
    )

    mx.random.seed(cfg.seed)
    model = BertForMaskedLM(cfg.encoder_config(arm, ids.vocab_size))
    init_weights(model, cfg.model.initializer_range)
    model.train()
    optimizer = build_optimizer(cfg, plan)
    optimizer.init(model.trainable_parameters())
    params = count_parameters(model)
    order = np.random.default_rng(cfg.seed).permutation(len(train_seqs))

    ckpts = Checkpoints(checkpoint_root, cfg.keep_checkpoints)
    step = 0
    bytes_seen = 0.0
    train_sec = 0.0
    if dry_run_minutes is None and (latest := ckpts.latest()) is not None:
        state = Checkpoints.load(latest, model, optimizer)
        step, bytes_seen, train_sec = state["step"], state["bytes_seen"], state["train_sec"]
        logger.info("Resumed %s at step %d from %s", arm.id, step, latest)
    mx.eval(model.parameters(), optimizer.state)

    def loss_fn(m: BertForMaskedLM, batch: mx.array, key: mx.array) -> mx.array:
        corrupted, positions, labels = mask_tokens(
            key, batch, ids, k, cfg.mlm.replace_mask, cfg.mlm.replace_random
        )
        logits = m(corrupted, positions=positions)
        return nn.losses.cross_entropy(logits, labels, reduction="mean")

    grad_fn = nn.value_and_grad(model, loss_fn)
    micro_state = [model.state, mx.random.state]

    def _micro(batch: mx.array, key: mx.array) -> tuple[mx.array, Any]:
        return grad_fn(model, batch, key)

    micro_step = mx.compile(_micro, inputs=micro_state, outputs=micro_state)
    update_state = [model.state, optimizer.state]

    def _update(grads: Any) -> mx.array:
        grads, norm = optim.clip_grad_norm(grads, cfg.optim.grad_clip)
        optimizer.update(model, grads)
        return norm

    update_step = mx.compile(_update, inputs=update_state, outputs=update_state)

    curve_path = run_dir / "curve.jsonl"
    curve = curve_path.open("a", encoding="utf-8")
    evals: list[dict[str, Any]] = []
    timing: dict[str, Any] = {}
    window_loss = 0.0
    window_steps = 0
    window_start = time.perf_counter()
    run_start = time.perf_counter()
    timed_steps = 0
    timed_sec = 0.0
    deadline = None if dry_run_minutes is None else run_start + 60 * dry_run_minutes
    mx.reset_peak_memory()

    def log_eval(at_step: int) -> dict[str, Any]:
        start = time.perf_counter()
        result = evaluate(model, data_dir, ids, cfg, k)
        result |= {
            "step": at_step,
            "bytes_seen": bytes_seen,
            "eval_sec": time.perf_counter() - start,
        }
        evals.append(result)
        curve.write(json.dumps({"type": "eval", **result}) + "\n")
        curve.flush()
        logger.info(
            "eval step %d: bits/byte %.4f, mlm loss %.4f, acc %.4f (%.0fs)",
            at_step,
            result["bits_per_byte"],
            result["mlm_loss"],
            result["mlm_accuracy"],
            result["eval_sec"],
        )
        return result

    try:
        while step < plan.total_steps:
            step_start = time.perf_counter()
            index = order[step * plan.batch_seqs : (step + 1) * plan.batch_seqs]
            accum: Any = None
            loss_total = mx.array(0.0)
            # Dropout draws from the global RNG; re-seeding from (seed, step) makes every step
            # reproducible on resume without saving RNG state (opaque in MLX 0.32).
            mx.random.seed(cfg.seed * 1_000_003 + 2 * step + 1)
            base_key = mx.random.key(cfg.seed * 1_000_003 + 2 * step)
            for m, micro_keys in enumerate(mx.random.split(base_key, plan.n_micro)):
                chunk = index[m * plan.micro_seqs : (m + 1) * plan.micro_seqs]
                loss, grads = micro_step(mx.array(train_seqs.batch(chunk)), micro_keys)
                loss_total = loss_total + loss
                accum = grads if accum is None else tree_map(mx.add, accum, grads)
            if plan.n_micro > 1:
                accum = tree_map(lambda g: g / plan.n_micro, accum)
            norm = update_step(accum)
            mx.eval(loss_total, norm, model.parameters(), optimizer.state)
            elapsed = time.perf_counter() - step_start
            step += 1
            bytes_seen += float(seq_bytes[index].sum())
            train_sec += elapsed
            if step > cfg.dry_run.timing_skip_steps:
                timed_steps += 1
                timed_sec += elapsed
            window_loss += _scalar(loss_total) / plan.n_micro
            window_steps += 1

            if step % cfg.log_every_steps == 0 or step == plan.total_steps:
                window = time.perf_counter() - window_start
                record = {
                    "type": "train",
                    "step": step,
                    "loss": window_loss / window_steps,
                    "lr": _scalar(optimizer.optimizers[0].learning_rate),
                    "grad_norm": _scalar(norm),
                    "tokens_per_sec": window_steps * tokens_per_step / window,
                    "bytes_per_sec": window_steps * plan.batch_bytes / window,
                    "bytes_seen": bytes_seen,
                    "peak_memory_gb": mx.get_peak_memory() / 1e9,
                }
                curve.write(json.dumps(record) + "\n")
                curve.flush()
                logger.info(
                    "step %d/%d loss %.4f lr %.2e gnorm %.2f %.0f tok/s %.2f GB",
                    step,
                    plan.total_steps,
                    record["loss"],
                    record["lr"],
                    record["grad_norm"],
                    record["tokens_per_sec"],
                    record["peak_memory_gb"],
                )
                window_loss, window_steps, window_start = 0.0, 0, time.perf_counter()

            if dry_run_minutes is None and step % cfg.eval.every_steps == 0:
                log_eval(step)
            if dry_run_minutes is None and step % cfg.checkpoint_every_steps == 0:
                ckpts.save(
                    f"step_{step:07d}",
                    model,
                    optimizer,
                    {"step": step, "bytes_seen": bytes_seen, "train_sec": train_sec},
                )
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if stop_after_steps is not None and step >= stop_after_steps:
                ckpts.save(
                    f"step_{step:07d}",
                    model,
                    optimizer,
                    {"step": step, "bytes_seen": bytes_seen, "train_sec": train_sec},
                )
                break

        if dry_run_minutes is not None:
            eval_result = log_eval(step)
            start = time.perf_counter()
            ckpts.save(
                f"step_{step:07d}",
                model,
                optimizer,
                {"step": step, "bytes_seen": bytes_seen, "train_sec": train_sec},
            )
            ckpt_sec = time.perf_counter() - start
            sec_per_step = timed_sec / max(timed_steps, 1)
            n_evals = plan.total_steps // cfg.eval.every_steps + 1
            n_ckpts = plan.total_steps // cfg.checkpoint_every_steps + 1
            estimate = (
                plan.total_steps * sec_per_step
                + n_evals * eval_result["eval_sec"]
                + n_ckpts * ckpt_sec
            )
            timing = {
                "steps_run": step,
                "sec_per_step": sec_per_step,
                "timed_steps": timed_steps,
                "eval_sec": eval_result["eval_sec"],
                "checkpoint_sec": ckpt_sec,
                "estimated_full_run_sec": estimate,
                "estimated_full_run_hours": estimate / 3600,
            }
            logger.info("Dry run %s: %s", arm.id, json.dumps(timing))
        elif step >= plan.total_steps:
            if not evals or evals[-1]["step"] != step:
                log_eval(step)
            ckpts.save(
                FINAL_DIR,
                model,
                optimizer,
                {"step": step, "bytes_seen": bytes_seen, "train_sec": train_sec},
            )
    finally:
        curve.close()

    return {
        "arm": arm.model_dump(mode="json"),
        "data": meta,
        "params": params,
        "plan": {
            "micro_seqs": plan.micro_seqs,
            "n_micro": plan.n_micro,
            "batch_seqs": plan.batch_seqs,
            "batch_bytes": plan.batch_bytes,
            "total_steps": plan.total_steps,
            "warmup_steps": plan.warmup_steps,
            "predictions_per_seq": k,
        },
        "steps_done": step,
        "bytes_seen": bytes_seen,
        "train_sec": train_sec,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "final_eval": evals[-1] if evals else None,
        "evals": evals,
        "dry_run": timing or None,
        "checkpoint_dir": str(checkpoint_root),
    }
