"""Phase 4 entry point: MLM pretraining of one arm (full run, resumable, or timed dry run).

The backend comes from the config and can be overridden with ``--backend``: ``mlx`` for
Apple silicon, ``torch`` for CUDA. The two share the data pipeline, the batch plan and the
evaluation, and their encoders are held to each other by the parity test.

Usage:
    uv run python -m leanvocab.pretrain.run --config configs/pretrain/phase4.yaml --arm bpe-8k
    uv run python -m leanvocab.pretrain.run --config ... --arm bpe-8k --dry-run
    uv run python -m leanvocab.pretrain.run --config ... --arm bpe-8k --backend torch
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from leanvocab.pretrain.config import Phase4Config
from leanvocab.utils.config import load_config
from leanvocab.utils.mlx_env import configure_mlx

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Timed run from scratch")
    parser.add_argument(
        "--backend",
        choices=["mlx", "torch"],
        default=None,
        help="Override the config's backend (mlx on Apple silicon, torch on CUDA)",
    )
    parser.add_argument(
        "--dry-run-root",
        type=Path,
        default=None,
        help="Checkpoint root for dry runs (default: <checkpoint_dir>/dry-run)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, Phase4Config)
    if args.backend is not None and args.backend != cfg.backend:
        cfg = cfg.model_copy(update={"backend": args.backend})
    env = configure_mlx(cfg.tf32) if cfg.backend == "mlx" else {}
    arm = cfg.arm(args.arm)

    # Backend-dependent imports only after the environment is configured.
    if cfg.backend == "mlx":
        from leanvocab.pretrain.mlm import train
    else:
        from leanvocab.pretrain.torch_mlm import train
    from leanvocab.utils.logging import setup_logging
    from leanvocab.utils.registry import RunContext, write_metrics
    from leanvocab.utils.seeding import seed_everything

    key = cfg.training_key(arm)
    finished = cfg.checkpoint_dir / f"{arm.id}-{key}" / "final" / "state.json"
    if not args.dry_run and finished.exists():
        setup_logging(None)
        logger.info("Arm %s (key %s) already finished; skipping", arm.id, key)
        return

    phase = "phase4-dryrun" if args.dry_run else "phase4"
    ctx = RunContext(phase=phase, config=cfg, seed=cfg.seed)
    ctx.run_id = f"{ctx.run_id}-{arm.id}"
    setup_logging(ctx.run_dir)
    seed_everything(cfg.seed)
    if args.dry_run:
        root = args.dry_run_root or cfg.checkpoint_dir / "dry-run"
        ckpt_root = root / f"{arm.id}-{key}-{ctx.run_id}"
    else:
        ckpt_root = cfg.checkpoint_dir / f"{arm.id}-{key}"
    logger.info(
        "Run %s arm=%s backend=%s key=%s env=%s ckpt=%s",
        ctx.run_id,
        arm.id,
        cfg.backend,
        key,
        env,
        ckpt_root,
    )

    metrics = train(
        cfg,
        arm,
        ctx.run_dir,
        ckpt_root,
        dry_run_minutes=cfg.dry_run.minutes if args.dry_run else None,
    )
    metrics["training_key"] = key
    metrics["backend"] = cfg.backend
    write_metrics(ctx, metrics)


if __name__ == "__main__":
    main()
