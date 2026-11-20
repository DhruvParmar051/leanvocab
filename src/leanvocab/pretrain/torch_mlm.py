"""MLM pretraining loop (PyTorch/CUDA), mirroring :mod:`leanvocab.pretrain.mlm`.

Same fairness rules as the MLX loop - identical documents, identical bytes of text per
optimizer step, identical optimizer, schedule, masking rate and seed - so the two backends
produce comparable runs. The encoder is held to the MLX one by
``tests/test_torch_encoder_parity.py``, and checkpoints keep the same file name and
parameter names, so published artifacts are backend-independent.

Reproducibility without storing RNG state: every step re-seeds from ``(seed, step)``, as in
the MLX loop, so a resume continues on the same masks and dropout draws.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from leanvocab.model.torch_encoder import (
    BertForMaskedLM,
    count_parameters,
    decay_parameter_groups,
    init_weights,
)
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
from leanvocab.pretrain.plan import BITS_PER_NAT, BatchPlan, num_predictions, plan_batches
from leanvocab.pretrain.torch_masking import mask_tokens

logger = logging.getLogger(__name__)

STATE_FILE = "state.json"
MODEL_FILE = "model.safetensors"
OPTIMIZER_FILE = "optimizer.pt"
FINAL_DIR = "final"
AMP_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}

__all__ = ["FINAL_DIR", "MODEL_FILE", "OPTIMIZER_FILE", "STATE_FILE", "train"]


def resolve_device(cfg: Phase4Config) -> torch.device:
    """The configured device, refusing to silently fall back to CPU."""
    if cfg.torch.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "backend=torch with device=cuda but CUDA is not available; "
            "set torch.device: cpu in the config to run on CPU"
        )
    return torch.device(cfg.torch.device)


def build_optimizer(cfg: Phase4Config, plan: BatchPlan, model: nn.Module) -> tuple[AdamW, LambdaLR]:
    o = cfg.optim
    warmup = max(plan.warmup_steps, 1)
    decay_steps = max(plan.total_steps - warmup, 1)
    optimizer = AdamW(
        decay_parameter_groups(model, o.weight_decay),
        lr=o.learning_rate,
        betas=(o.beta1, o.beta2),
        eps=o.eps,
    )

    def factor(step: int) -> float:
        """Linear warmup then linear decay to zero, as a multiple of the peak rate."""
        if step < warmup:
            return step / warmup
        return max(0.0, 1.0 - (step - warmup) / decay_steps)

    return optimizer, LambdaLR(optimizer, factor)


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
        self,
        name: str,
        model: nn.Module,
        optimizer: AdamW,
        scheduler: LambdaLR,
        state: dict[str, Any],
    ) -> Path:
        final = self.root / name
        tmp = self.root / f"{name}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        weights = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        save_file(weights, str(tmp / MODEL_FILE))
        torch.save(
            {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
            tmp / OPTIMIZER_FILE,
        )
        (tmp / STATE_FILE).write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)
        for old in self._steps()[: -self.keep]:
            shutil.rmtree(old)
        return final

    @staticmethod
    def load(
        path: Path,
        model: nn.Module,
        optimizer: AdamW,
        scheduler: LambdaLR,
        device: torch.device,
    ) -> dict[str, Any]:
        model.load_state_dict(load_file(str(path / MODEL_FILE), device=str(device)))
        blob = torch.load(path / OPTIMIZER_FILE, map_location=device, weights_only=True)
        optimizer.load_state_dict(blob["optimizer"])
        scheduler.load_state_dict(blob["scheduler"])
        state: dict[str, Any] = json.loads((path / STATE_FILE).read_text())
        return state


def _batch_tensor(rows: np.ndarray, device: torch.device) -> Tensor:
    return torch.from_numpy(rows).to(device=device, dtype=torch.long, non_blocking=True)


def evaluate(
    model: BertForMaskedLM,
    data_dir: Path,
    ids: SpecialIds,
    cfg: Phase4Config,
    k: int,
    device: torch.device,
) -> dict[str, float]:
    """Held-out MLM loss and bits per byte with fixed masks (dropout off)."""
    token_bytes = np.fromfile(data_dir / EVAL_BYTES_FILE, dtype=np.float32)
    packed = PackedSequences(data_dir / EVAL_TOKENS_FILE, cfg.seq_len, ids)
    body = cfg.seq_len - 2
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device)
    nll_sum = bytes_sum = correct = count = 0.0
    try:
        with torch.no_grad():
            for b, start in enumerate(range(0, len(packed), cfg.eval.batch_seqs)):
                index = np.arange(start, min(start + cfg.eval.batch_seqs, len(packed)))
                batch = _batch_tensor(packed.batch(index), device)
                generator.manual_seed(cfg.eval.seed + b)
                corrupted, positions, labels = mask_tokens(
                    generator, batch, ids, k, cfg.mlm.replace_mask, cfg.mlm.replace_random
                )
                logits = model(corrupted, positions=positions).float()
                nll = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
                )
                hits = (logits.argmax(dim=-1) == labels).float()
                # Bytes of the masked tokens: position p in a row maps to stream offset p - 1.
                pos = positions.cpu().numpy().astype(np.int64)
                stream_idx = index[:, None] * body + (pos - 1)
                nll_sum += float(nll.sum().item())
                correct += float(hits.sum().item())
                count += float(nll.numel())
                bytes_sum += float(token_bytes[stream_idx].astype(np.float64).sum())
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
    """Train one arm on PyTorch. ``stop_after_steps`` checkpoints and returns early."""
    device = resolve_device(cfg)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = cfg.torch.allow_tf32
        torch.backends.cudnn.allow_tf32 = cfg.torch.allow_tf32
        torch.cuda.reset_peak_memory_stats()

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
        "%s on %s: %d seqs, batch %d x %d = %d seqs (%.0f bytes), %d steps, warmup %d, k=%d",
        arm.id,
        device,
        len(train_seqs),
        plan.micro_seqs,
        plan.n_micro,
        plan.batch_seqs,
        plan.batch_bytes,
        plan.total_steps,
        plan.warmup_steps,
        k,
    )

    torch.manual_seed(cfg.seed)
    model = BertForMaskedLM(cfg.encoder_config(arm, ids.vocab_size))
    init_weights(model, cfg.model.initializer_range)
    model.to(device)
    model.train()
    optimizer, scheduler = build_optimizer(cfg, plan, model)
    params = count_parameters(model)
    order = np.random.default_rng(cfg.seed).permutation(len(train_seqs))

    amp_dtype = AMP_DTYPES.get(cfg.torch.amp_dtype)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype is torch.float16)
    step_model = torch.compile(model) if cfg.torch.compile else model
    generator = torch.Generator(device=device)

    ckpts = Checkpoints(checkpoint_root, cfg.keep_checkpoints)
    step = 0
    bytes_seen = 0.0
    train_sec = 0.0
    if dry_run_minutes is None and (latest := ckpts.latest()) is not None:
        state = Checkpoints.load(latest, model, optimizer, scheduler, device)
        step, bytes_seen, train_sec = state["step"], state["bytes_seen"], state["train_sec"]
        logger.info("Resumed %s at step %d from %s", arm.id, step, latest)

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

    def peak_gb() -> float:
        return torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

    def log_eval(at_step: int) -> dict[str, Any]:
        start = time.perf_counter()
        result = evaluate(model, data_dir, ids, cfg, k, device)
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

    def checkpoint(name: str) -> None:
        ckpts.save(
            name,
            model,
            optimizer,
            scheduler,
            {"step": step, "bytes_seen": bytes_seen, "train_sec": train_sec},
        )

    try:
        while step < plan.total_steps:
            step_start = time.perf_counter()
            index = order[step * plan.batch_seqs : (step + 1) * plan.batch_seqs]
            # Dropout draws from the global RNG; re-seeding from (seed, step) makes every step
            # reproducible on resume without storing RNG state.
            torch.manual_seed(cfg.seed * 1_000_003 + 2 * step + 1)
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            for m in range(plan.n_micro):
                chunk = index[m * plan.micro_seqs : (m + 1) * plan.micro_seqs]
                batch = _batch_tensor(train_seqs.batch(chunk), device)
                generator.manual_seed(cfg.seed * 1_000_003 + 2 * step + 1_000_000 * (m + 1))
                corrupted, positions, labels = mask_tokens(
                    generator, batch, ids, k, cfg.mlm.replace_mask, cfg.mlm.replace_random
                )
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    logits = step_model(corrupted, positions=positions)
                    loss = nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1)
                    )
                # torch ships no annotation for Tensor.backward
                scaler.scale(loss / plan.n_micro).backward()  # type: ignore[no-untyped-call]
                loss_total += float(loss.detach().item())
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            lr = float(scheduler.get_last_lr()[0])
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - step_start
            step += 1
            bytes_seen += float(seq_bytes[index].sum())
            train_sec += elapsed
            if step > cfg.dry_run.timing_skip_steps:
                timed_steps += 1
                timed_sec += elapsed
            window_loss += loss_total / plan.n_micro
            window_steps += 1

            if step % cfg.log_every_steps == 0 or step == plan.total_steps:
                window = time.perf_counter() - window_start
                record = {
                    "type": "train",
                    "step": step,
                    "loss": window_loss / window_steps,
                    "lr": lr,
                    "grad_norm": float(norm),
                    "tokens_per_sec": window_steps * tokens_per_step / window,
                    "bytes_per_sec": window_steps * plan.batch_bytes / window,
                    "bytes_seen": bytes_seen,
                    "peak_memory_gb": peak_gb(),
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
                checkpoint(f"step_{step:07d}")
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if stop_after_steps is not None and step >= stop_after_steps:
                checkpoint(f"step_{step:07d}")
                break

        if dry_run_minutes is not None:
            eval_result = log_eval(step)
            start = time.perf_counter()
            checkpoint(f"step_{step:07d}")
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
            checkpoint(FINAL_DIR)
    finally:
        curve.close()

    return {
        "arm": arm.model_dump(mode="json"),
        "backend": "torch",
        "device": str(device),
        "amp_dtype": cfg.torch.amp_dtype,
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
        "peak_memory_gb": peak_gb(),
        "final_eval": evals[-1] if evals else None,
        "evals": evals,
        "dry_run": timing or None,
        "checkpoint_dir": str(checkpoint_root),
    }
