"""Run registry: provenance capture and results bookkeeping."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from tokstudy.utils.config import config_hash

logger = logging.getLogger(__name__)

TRACKED_PACKAGES: tuple[str, ...] = (
    "tokstudy",
    "numpy",
    "pyarrow",
    "datasets",
    "huggingface-hub",
    "pydantic",
    "tokenizers",
    "mlx",
    "torch",
    "mteb",
    "sentence-transformers",
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def git_state() -> dict[str, Any]:
    """Current commit SHA (``None`` before the first commit) and dirty flag."""
    try:
        sha: str | None = _git("rev-parse", "HEAD")
    except RuntimeError:
        logger.warning("No git commit found; recording git_sha=None")
        sha = None
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return {"git_sha": sha, "git_dirty": dirty}


def library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


@dataclass
class RunContext:
    """Identity and provenance of one experiment run."""

    phase: str
    config: BaseModel
    seed: int
    results_root: Path = REPO_ROOT / "results"
    run_id: str = field(init=False)
    config_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        self.config_sha256 = config_hash(self.config)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{stamp}-{self.config_sha256[:8]}"

    @property
    def run_dir(self) -> Path:
        return self.results_root / self.phase / self.run_id

    def provenance(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "run_id": self.run_id,
            "seed": self.seed,
            "config_sha256": self.config_sha256,
            "config": self.config.model_dump(mode="json"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "libraries": library_versions(),
            **git_state(),
        }


def write_metrics(ctx: RunContext, metrics: dict[str, Any]) -> Path:
    """Write ``metrics.json`` for the run and append a summary line to ``results/index.jsonl``.

    Raises ``FileExistsError`` rather than overwriting an existing run's metrics.
    """
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    out = ctx.run_dir / "metrics.json"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing metrics: {out}")
    payload = {"provenance": ctx.provenance(), "metrics": metrics}
    with out.open("x", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    index_line = {
        "phase": ctx.phase,
        "run_id": ctx.run_id,
        "seed": ctx.seed,
        "config_sha256": ctx.config_sha256,
        "git_sha": payload["provenance"]["git_sha"],
        "metrics_path": str(out.relative_to(ctx.results_root.parent)),
        "written_at": datetime.now(UTC).isoformat(),
    }
    with (ctx.results_root / "index.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(index_line, sort_keys=True) + "\n")
    logger.info("Wrote metrics to %s", out)
    return out
