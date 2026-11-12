"""Publish the Phase 2 tokenizers and the finished Phase 4 checkpoints to the Hugging Face Hub.

Staging is separate from uploading: ``--dry-run`` writes exactly what would be pushed into
``--stage-dir`` and stops, so the cards and configs can be reviewed first.

Usage:
    uv run python -m leanvocab.publish.hf --config configs/publish/hf.yaml --dry-run
    uv run python -m leanvocab.publish.hf --config configs/publish/hf.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from safetensors import safe_open

from leanvocab.publish import cards
from leanvocab.publish.cards import ArmFacts, StudyFacts
from leanvocab.publish.config import ArmRepo, PublishConfig
from leanvocab.utils.config import load_config
from leanvocab.utils.logging import setup_logging

logger = logging.getLogger(__name__)

MODEL_FILE = "model.safetensors"
METRICS_FILE = "metrics.json"
TOKENIZER_FILE = "tokenizer.json"
META_FILE = "meta.json"
ENCODER_CONFIG_FILE = "leanvocab_config.json"
CARD_FILE = "README.md"
HEAD_PREFIX = "cls.predictions"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected {path}")
    with path.open(encoding="utf-8") as fh:
        loaded: dict[str, Any] = json.load(fh)
    return loaded


def _latest_run(results_dir: Path, arm_id: str) -> dict[str, Any]:
    """The newest recorded Phase 4 run for ``arm_id``.

    Run ids are ``<timestamp>-<config hash>-<arm id>``; the arm id is matched on that third
    field so that ``bert-orig`` cannot pick up ``factorized-bert-orig``.
    """
    candidates = sorted(
        p for p in results_dir.iterdir() if p.is_dir() and p.name.split("-", 2)[2:] == [arm_id]
    )
    if not candidates:
        raise FileNotFoundError(f"No Phase 4 run for arm {arm_id!r} under {results_dir}")
    return _read_json(candidates[-1] / METRICS_FILE)


def _param_counts(model_path: Path) -> tuple[int, int]:
    """(encoder parameters, all parameters in the file); the MLM head is excluded from the first."""
    total = 0
    head = 0
    with safe_open(str(model_path), framework="np") as handle:
        for key in handle.keys():
            shape = handle.get_slice(key).get_shape()
            count = 1
            for dim in shape:
                count *= dim
            total += count
            if key.startswith(HEAD_PREFIX):
                head += count
    return total - head, total


def _encoder_config(pretrain: dict[str, Any], arm: ArmRepo, vocab_size: int) -> dict[str, Any]:
    arm_entry = next(a for a in pretrain["arms"] if a["id"] == arm.id)
    return {
        "vocab_size": vocab_size,
        "embedding_dim": arm_entry["embedding_dim"],
        **pretrain["model"],
    }


def _transformers_config(
    cfg: PublishConfig, encoder: dict[str, Any], pretrain: dict[str, Any]
) -> dict[str, Any]:
    return {
        "architectures": ["BertForMaskedLM"],
        "model_type": "bert",
        "vocab_size": encoder["vocab_size"],
        "hidden_size": encoder["hidden_size"],
        "num_hidden_layers": encoder["num_layers"],
        "num_attention_heads": encoder["num_heads"],
        "intermediate_size": encoder["intermediate_size"],
        "hidden_act": cfg.hidden_act,
        "max_position_embeddings": encoder["max_position_embeddings"],
        "type_vocab_size": encoder["type_vocab_size"],
        "hidden_dropout_prob": encoder["hidden_dropout"],
        "attention_probs_dropout_prob": cfg.attention_dropout,
        "layer_norm_eps": encoder["layer_norm_eps"],
        "initializer_range": encoder["initializer_range"],
        "pad_token_id": pretrain["special_tokens"].index("[PAD]"),
        "position_embedding_type": "absolute",
        "tie_word_embeddings": True,
    }


def _tokenizer_config(cfg: PublishConfig, specials: list[str]) -> dict[str, Any]:
    pad, unk, cls_token, sep, mask = specials
    return {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": cfg.model_max_length,
        "pad_token": pad,
        "unk_token": unk,
        "cls_token": cls_token,
        "sep_token": sep,
        "mask_token": mask,
        "clean_up_tokenization_spaces": False,
    }


def _special_tokens_map(specials: list[str]) -> dict[str, str]:
    pad, unk, cls_token, sep, mask = specials
    return {
        "pad_token": pad,
        "unk_token": unk,
        "cls_token": cls_token,
        "sep_token": sep,
        "mask_token": mask,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_facts(cfg: PublishConfig) -> StudyFacts:
    """Read every recorded number the cards need."""
    phase3 = _read_json(cfg.phase3_results)["metrics"]
    arms: dict[str, ArmFacts] = {}
    pretrain: dict[str, Any] | None = None
    for arm in cfg.arms:
        record = _latest_run(cfg.phase4_results, arm.id)
        run, provenance = record["metrics"], record["provenance"]
        pretrain = provenance["config"]
        meta = _read_json(cfg.tokenizer_dir / arm.tokenizer / META_FILE)
        model_path = Path(run["checkpoint_dir"]) / "final" / MODEL_FILE
        encoder_params, total_params = _param_counts(model_path)
        arms[arm.id] = ArmFacts(
            arm=arm,
            run=run,
            provenance=provenance,
            tokenizer_meta=meta,
            intrinsic=phase3["tokenizers"][arm.tokenizer],
            encoder_config=_encoder_config(pretrain, arm, meta["vocab_size"]),
            encoder_params=encoder_params,
            total_params=total_params,
        )
        logger.info(
            "%s: %.4f bits/byte, %.2fM encoder params, run %s",
            arm.id,
            run["final_eval"]["bits_per_byte"],
            encoder_params / 1e6,
            provenance["run_id"],
        )
    if pretrain is None:
        raise ValueError("No arms configured")
    return StudyFacts(
        control_id=cfg.arms[0].id,
        arms=arms,
        heldout=phase3["heldout"],
        pretrain=pretrain,
    )


def stage_model_repo(cfg: PublishConfig, facts: StudyFacts, arm_id: str, stage: Path) -> Path:
    """Write one model repo into ``stage`` and return its directory."""
    f = facts.arms[arm_id]
    out = stage / f.arm.repo
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    final_dir = Path(f.run["checkpoint_dir"]) / "final"
    shutil.copy2(final_dir / MODEL_FILE, out / MODEL_FILE)
    if cfg.include_optimizer:
        shutil.copy2(final_dir / "optimizer.safetensors", out / "optimizer.safetensors")
    shutil.copy2(cfg.tokenizer_dir / f.arm.tokenizer / TOKENIZER_FILE, out / TOKENIZER_FILE)

    specials = facts.pretrain["special_tokens"]
    _write_json(out / ENCODER_CONFIG_FILE, f.encoder_config)
    _write_json(out / METRICS_FILE, {"metrics": f.run, "provenance": f.provenance})
    _write_json(out / "tokenizer_config.json", _tokenizer_config(cfg, specials))
    _write_json(out / "special_tokens_map.json", _special_tokens_map(specials))
    if f.arm.bert_compatible:
        _write_json(
            out / "config.json", _transformers_config(cfg, f.encoder_config, facts.pretrain)
        )
    (out / CARD_FILE).write_text(cards.model_card(cfg, facts, arm_id), encoding="utf-8")
    logger.info("Staged %s (%d files) at %s", f.arm.repo, len(list(out.iterdir())), out)
    return out


def stage_tokenizer_repo(cfg: PublishConfig, facts: StudyFacts, stage: Path) -> Path:
    """Write the all-tokenizers repo into ``stage`` and return its directory."""
    out = stage / cfg.tokenizer_repo
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    phase3 = _read_json(cfg.phase3_results)["metrics"]
    phase2_record = _read_json(cfg.phase2_results)

    metas: dict[str, dict[str, Any]] = {}
    for tid in cfg.tokenizer_ids:
        src = cfg.tokenizer_dir / tid
        dst = out / tid
        dst.mkdir()
        shutil.copy2(src / TOKENIZER_FILE, dst / TOKENIZER_FILE)
        shutil.copy2(src / META_FILE, dst / META_FILE)
        metas[tid] = _read_json(src / META_FILE)
        _write_json(
            dst / "tokenizer_config.json", _tokenizer_config(cfg, metas[tid]["special_tokens"])
        )
        _write_json(
            dst / "special_tokens_map.json", _special_tokens_map(metas[tid]["special_tokens"])
        )

    card = cards.tokenizer_card(
        cfg,
        metas=metas,
        intrinsic=phase3["tokenizers"],
        heldout=phase3["heldout"],
        provenance=phase2_record["provenance"],
        phase2=phase2_record["provenance"]["config"],
    )
    (out / CARD_FILE).write_text(card, encoding="utf-8")
    logger.info("Staged %s (%d tokenizers) at %s", cfg.tokenizer_repo, len(metas), out)
    return out


def upload(cfg: PublishConfig, repo: str, folder: Path, repo_type: str, message: str) -> str:
    """Create ``repo`` if needed and upload ``folder`` into it; returns the repo URL."""
    api = HfApi()
    repo_id = f"{cfg.owner}/{repo}"
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=cfg.private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(folder),
        commit_message=message,
    )
    url = f"https://huggingface.co/{repo_id}"
    logger.info("Uploaded %s -> %s", folder.name, url)
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, default=Path("data/hf-staging"))
    parser.add_argument("--dry-run", action="store_true", help="Stage the repos but do not upload")
    parser.add_argument("--only", nargs="*", default=None, help="Repo names to limit the run to")
    args = parser.parse_args()

    setup_logging(None)
    cfg = load_config(args.config, PublishConfig)
    facts = collect_facts(cfg)

    staged: list[tuple[str, Path, str]] = []
    for arm in cfg.arms:
        if args.only and arm.repo not in args.only:
            continue
        staged.append(("model", stage_model_repo(cfg, facts, arm.id, args.stage_dir), arm.repo))
    if not args.only or cfg.tokenizer_repo in args.only:
        staged.append(
            ("model", stage_tokenizer_repo(cfg, facts, args.stage_dir), cfg.tokenizer_repo)
        )

    if args.dry_run:
        logger.info(
            "Dry run: %d repos staged under %s, nothing uploaded", len(staged), args.stage_dir
        )
        return

    message = "Add leanvocab Phase 4 weights, tokenizer and model card"
    for repo_type, folder, repo in staged:
        upload(cfg, repo, folder, repo_type, message)


if __name__ == "__main__":
    main()
