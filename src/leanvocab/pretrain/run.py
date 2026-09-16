"""Phase 4 entry point: MLM pretraining of one arm (full run, resumable, or timed dry run).

Usage:
    uv run python -m leanvocab.pretrain.run --config configs/pretrain/phase4.yaml --arm bpe-8k
    uv run python -m leanvocab.pretrain.run --config ... --arm bpe-8k --dry-run
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
        "--dry-run-root",
        type=Path,
        default=None,
        help="Checkpoint root for dry runs (default: <checkpoint_dir>/dry-run)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, Phase4Config)
    env = configure_mlx(cfg.tf32)
    arm = cfg.arm(args.arm)

    # MLX-dependent imports only after the environment is configured.
    from leanvocab.pretrain.mlm import train
    from leanvocab.utils.logging import setup_logging
    from leanvocab.utils.registry import RunContext, write_metrics
    from leanvocab.utils.seeding import seed_everything

    phase = "phase4-dryrun" if args.dry_run else "phase4"
    ctx = RunContext(phase=phase, config=cfg, seed=cfg.seed)
    ctx.run_id = f"{ctx.run_id}-{arm.id}"
    setup_logging(ctx.run_dir)
    seed_everything(cfg.seed)
    key = cfg.training_key(arm)
    if args.dry_run:
        root = args.dry_run_root or cfg.checkpoint_dir / "dry-run"
        ckpt_root = root / f"{arm.id}-{key}-{ctx.run_id}"
    else:
        ckpt_root = cfg.checkpoint_dir / f"{arm.id}-{key}"
    logger.info("Run %s arm=%s key=%s env=%s ckpt=%s", ctx.run_id, arm.id, key, env, ckpt_root)

    metrics = train(
        cfg,
        arm,
        ctx.run_dir,
        ckpt_root,
        dry_run_minutes=cfg.dry_run.minutes if args.dry_run else None,
    )
    metrics["training_key"] = key
    write_metrics(ctx, metrics)


if __name__ == "__main__":
    main()
