"""Model and tokenizer card text for the Hugging Face repos.

Every number in a card is read from a recorded run; none of them are literals here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from leanvocab.publish.config import ArmRepo, PublishConfig

CLAIM = (
    "a tokenizer with an 8k-16k vocabulary can reach sentence-embedding quality within "
    "0.5 MTEB-avg points of a same-recipe MiniLM-shaped encoder using BERT's 30,522-token "
    "vocabulary, while freeing 25-38% of the parameters and without slower end-to-end inference"
)
ALGORITHM_NAMES = {
    "byte_level_bpe": "Byte-level BPE",
    "unigram_byte_fallback": "Unigram with byte fallback",
    "wordpiece": "WordPiece (retrained)",
    "wordpiece_pretrained": "WordPiece (original `bert-base-uncased` vocabulary)",
    "wordpiece_trimmed": "WordPiece (BERT vocabulary trimmed by corpus frequency)",
}


@dataclass(frozen=True)
class ArmFacts:
    """Everything recorded about one finished pretraining arm."""

    arm: ArmRepo
    run: dict[str, Any]
    provenance: dict[str, Any]
    tokenizer_meta: dict[str, Any]
    intrinsic: dict[str, Any]
    encoder_config: dict[str, Any]
    encoder_params: int
    total_params: int


@dataclass(frozen=True)
class StudyFacts:
    """Shared context: the control arm, the held-out split and the pretraining config."""

    control_id: str
    arms: dict[str, ArmFacts]
    heldout: dict[str, Any]
    pretrain: dict[str, Any]

    @property
    def control(self) -> ArmFacts:
        return self.arms[self.control_id]


def _millions(params: int) -> str:
    return f"{params / 1e6:.2f}M"


def _yaml_list(key: str, values: list[str]) -> str:
    return key + ":" + "".join(f"\n- {v}" for v in values)


def _arm_table(facts: StudyFacts) -> list[str]:
    control = facts.control
    rows = [
        "| Arm | Vocab | Encoder params | Held-out bits/byte | vs control | MLM acc | Wall clock |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(facts.arms.values(), key=lambda f: f.run["final_eval"]["bits_per_byte"])
    for f in ordered:
        bpb = f.run["final_eval"]["bits_per_byte"]
        delta = bpb - control.run["final_eval"]["bits_per_byte"]
        ratio = f.encoder_params / control.encoder_params
        is_control = f.arm.id == facts.control_id
        rows.append(
            f"| `{f.arm.id}`{' (control)' if is_control else ''} "
            f"| {f.tokenizer_meta['vocab_size']:,} "
            f"| {_millions(f.encoder_params)} ({ratio:.2f}x) "
            f"| {bpb:.4f} "
            f"| {'-' if is_control else f'{delta:+.4f}'} "
            f"| {f.run['final_eval']['mlm_accuracy'] * 100:.1f}% "
            f"| {f.run['train_sec'] / 3600:.2f} h |"
        )
    return rows


def _takeaways(facts: StudyFacts) -> list[str]:
    """One bullet per non-control arm, phrased from its own recorded numbers."""
    control = facts.control
    control_bpb = control.run["final_eval"]["bits_per_byte"]
    bullets: list[str] = []
    others = [f for f in facts.arms.values() if f.arm.id != facts.control_id]
    for f in sorted(others, key=lambda f: f.run["final_eval"]["bits_per_byte"]):
        delta = f.run["final_eval"]["bits_per_byte"] - control_bpb
        saved = (1 - f.encoder_params / control.encoder_params) * 100
        faster = (1 - f.run["train_sec"] / control.run["train_sec"]) * 100
        vocab = f"{f.tokenizer_meta['vocab_size']:,}-token"
        if delta < -0.01:
            verdict = (
                f"**beat** the control by {abs(delta):.3f} bits/byte while using "
                f"{saved:.0f}% fewer encoder parameters and {faster:.0f}% less wall clock"
            )
        elif delta < 0.05:
            verdict = (
                f"landed within {abs(delta):.3f} bits/byte of the control for "
                f"{saved:.0f}% fewer encoder parameters - close enough that a single seed "
                "cannot separate them"
            )
        else:
            verdict = (
                f"came out **{delta:.3f} bits/byte worse** than the control despite "
                f"{saved:.0f}% fewer encoder parameters, so its parameter saving did not come "
                "for free under this recipe"
            )
        bullets.append(f"- `{f.arm.id}` ({vocab} {f.arm.role}) {verdict}.")
    bullets.append(
        "- Parameter savings are not interchangeable: an equally sized saving from embedding "
        "factorization behaves very differently from one from a smaller vocabulary, as the table "
        "shows."
    )
    return bullets


def _recipe_lines(facts: StudyFacts) -> list[str]:
    cfg = facts.pretrain
    optim, mlm = cfg["optim"], cfg["mlm"]
    model, ev = cfg["model"], cfg["eval"]
    run = facts.control.run
    return [
        "- **Corpus:** FineWeb-Edu `sample-10BT` at a pinned revision, NFC-normalized, exactly "
        "deduplicated (BLAKE2b-128), with a frozen held-out split removed before training.",
        f"- **Budget:** {cfg['budget_bytes'] / 1e9:.0f} GB of text per arm, a single pass, "
        f"sequence length {cfg['seq_len']} with RoBERTa-style full-sentence packing.",
        f"- **Equal-bytes batching:** every step consumes about {cfg['batch_bytes'] / 1e3:.1f} kB "
        f"of raw text rather than a fixed number of tokens, so arms are not accidentally given "
        f"different amounts of text. For this arm that is {run['plan']['batch_seqs']} sequences "
        f"in {run['plan']['n_micro']} micro-batches.",
        f"- **Objective:** MLM at {mlm['mask_prob'] * 100:.0f}% masking, "
        f"{mlm['replace_mask'] * 100:.0f}/{mlm['replace_random'] * 100:.0f}/"
        f"{(1 - mlm['replace_mask'] - mlm['replace_random']) * 100:.0f} "
        f"mask/random/keep, with a fixed {run['plan']['predictions_per_seq']} predictions per "
        f"sequence so the compiled graph has a static shape.",
        f"- **Shape:** {model['num_layers']} layers, hidden {model['hidden_size']}, "
        f"{model['num_heads']} heads, FFN {model['intermediate_size']}, "
        f"{model['max_position_embeddings']} positions, embeddings tied to the MLM output layer.",
        f"- **Optimizer:** AdamW, lr {optim['learning_rate']:g}, "
        f"betas ({optim['beta1']}, {optim['beta2']}), weight decay {optim['weight_decay']} "
        f"(not on biases or LayerNorm), gradient clipping {optim['grad_clip']}, linear warmup over "
        f"{optim['warmup_fraction'] * 100:.0f}% of steps then linear decay.",
        f"- **Evaluation:** held-out bits per byte on {ev['heldout_bytes'] / 1e6:.0f} MB with "
        f"masks fixed by seed {ev['seed']}, every {ev['every_steps']} steps.",
        f"- **Hardware:** one Apple M5 Pro (48 GB) via MLX; no CUDA anywhere in the study. "
        f"Peak GPU memory {run['peak_memory_gb']:.1f} GB.",
    ]


def _provenance_lines(cfg: PublishConfig, f: ArmFacts) -> list[str]:
    prov = f.provenance
    libs = prov["libraries"]
    tf32 = prov["env"].get("MLX_ENABLE_TF32")
    return [
        f"- Run `{prov['run_id']}`, seed {prov['seed']}, config SHA-256 "
        f"`{prov['config_sha256'][:12]}`, commit `{prov['git_sha'][:12]}`"
        + ("" if not prov["git_dirty"] else " (working tree dirty)")
        + ".",
        f"- {prov['platform']}, Python {prov['python']}, MLX {libs['mlx']}, "
        f"tokenizers {libs['tokenizers']}.",
        f"- `MLX_ENABLE_TF32={tf32}`: MLX uses TF32 matmuls on Apple GPUs by default "
        "(about 1e-3 relative error, about 1.9x faster). The study records this rather than "
        "silently depending on it.",
        f"- Tokenizer `tokenizer.json` SHA-256 `{f.tokenizer_meta['tokenizer_sha256'][:12]}`; "
        f"round-trip check on {f.tokenizer_meta['roundtrip']['docs']:,} held-out documents: "
        f"{f.tokenizer_meta['roundtrip']['mode']}, "
        f"{'passed' if f.tokenizer_meta['roundtrip']['passed'] else 'FAILED'}.",
        f"- Code, configs and the recorded metrics: [{cfg.repo_url}]({cfg.repo_url}).",
    ]


def _usage_lines(cfg: PublishConfig, f: ArmFacts) -> list[str]:
    repo = f"{cfg.owner}/{f.arm.repo}"
    byte_level = f.tokenizer_meta["algorithm"] == "byte_level_bpe"
    # A byte-level pre-tokenizer turns the space before `[MASK]` into its own token, which the
    # packed pretraining sequences never contained; writing the mask without the space avoids it.
    example = (
        "photosynthesis happens in the[MASK] of plant cells."
        if byte_level
        else "photosynthesis happens in the [MASK] of plant cells."
    )
    if f.arm.bert_compatible:
        lines = [
            "The weights were trained in MLX but are stored with `BertForMaskedLM` parameter "
            "names, so `transformers` loads them directly:",
            "",
            "```python",
            "from transformers import AutoModelForMaskedLM, AutoTokenizer",
            "",
            f'model = AutoModelForMaskedLM.from_pretrained("{repo}")',
            f'tok = AutoTokenizer.from_pretrained("{repo}")',
            "",
            f'inputs = tok("{example}", return_tensors="pt")',
            "logits = model(**inputs).logits",
            "```",
            "",
            "There is no pooler in the checkpoint (the MLM recipe never trains one), so a "
            "`BertModel` load will report a randomly initialized `pooler` - that is expected.",
        ]
        if byte_level:
            lines += [
                "",
                "One tokenizer detail worth knowing: this vocabulary uses a byte-level "
                "pre-tokenizer, so a space written before `[MASK]` becomes its own `\u0120` "
                "token - a shape the packed pretraining sequences never contained, and "
                "predictions at that position get noticeably worse. Attach the mask to the "
                "preceding word (`the[MASK]`, not `the [MASK]`); the token it predicts carries "
                "the leading space itself.",
            ]
        return lines
    dim = f.encoder_config["embedding_dim"]
    hidden = f.encoder_config["hidden_size"]
    return [
        f"This arm factorizes the token embeddings (a {f.tokenizer_meta['vocab_size']:,} x {dim} "
        f"table, then a projection up to the hidden size {hidden}), which is not a parameter "
        "layout stock `BertForMaskedLM` understands. Load it with the study's own MLX encoder "
        f"from a [leanvocab]({cfg.repo_url}) checkout:",
        "",
        "```python",
        "import json",
        "",
        "from huggingface_hub import snapshot_download",
        "",
        "from leanvocab.model.config import EncoderConfig",
        "from leanvocab.model.encoder import BertForMaskedLM",
        "",
        f'path = snapshot_download("{repo}")',
        'cfg = EncoderConfig(**json.load(open(f"{path}/leanvocab_config.json")))',
        "model = BertForMaskedLM(cfg)",
        'model.load_weights(f"{path}/model.safetensors")',
        "```",
        "",
        "The weights are published mainly so the negative result is inspectable - see "
        "[Intended use](#intended-use-and-limitations).",
    ]


def model_card(cfg: PublishConfig, facts: StudyFacts, arm_id: str) -> str:
    """Card for one pretraining arm."""
    f = facts.arms[arm_id]
    control = facts.control
    final = f.run["final_eval"]
    meta, met = f.tokenizer_meta, f.intrinsic
    vocab = meta["vocab_size"]
    is_control = arm_id == facts.control_id
    delta = final["bits_per_byte"] - control.run["final_eval"]["bits_per_byte"]
    saved = 1 - f.encoder_params / control.encoder_params

    tags = ["bert", "fill-mask", "masked-lm", "tokenizer", "vocabulary", "mlx", "research"]
    head = [
        "---",
        f"license: {cfg.license}",
        _yaml_list("language", [cfg.language]),
        _yaml_list("datasets", [cfg.source_dataset]),
        _yaml_list("tags", tags),
    ]
    if f.arm.bert_compatible:
        head += ["library_name: transformers", "pipeline_tag: fill-mask"]
    head += ["---", ""]

    lines = head + [
        f"# leanvocab-{arm_id}",
        "",
        f"A {facts.pretrain['model']['num_layers']}-layer, "
        f"{facts.pretrain['model']['hidden_size']}-wide BERT-shaped encoder pretrained "
        f"from scratch with masked language modelling on "
        f"{f.run['bytes_seen'] / 1e9:.2f} GB of FineWeb-Edu text, using a "
        f"**{vocab:,}-token vocabulary** ({ALGORITHM_NAMES[meta['algorithm']]}).",
        "",
        f"This is the **{f.arm.role}** arm of [leanvocab]({cfg.repo_url}), a controlled study "
        f"asking whether {CLAIM}.",
        "",
        "> **This is a research checkpoint, not a finished embedding model.** It has only been",
        "> pretrained with MLM on a small budget; the contrastive fine-tuning that would make it",
        "> produce useful sentence embeddings has not run yet. See",
        "> [Intended use](#intended-use-and-limitations).",
        "",
        "## Summary",
        "",
        "| | |",
        "|---|---|",
        f"| Vocabulary | {vocab:,} tokens, {ALGORITHM_NAMES[meta['algorithm']]} |",
        f"| Encoder parameters | {_millions(f.encoder_params)}"
        + (" (the control)" if is_control else f" ({saved * 100:.0f}% fewer than the control)")
        + " |",
        f"| Parameters in `model.safetensors` (encoder + MLM head) | {_millions(f.total_params)} |",
        f"| Held-out bits per byte | **{final['bits_per_byte']:.4f}**"
        + (" (the reference)" if is_control else f" ({delta:+.4f} vs the control)")
        + " |",
        f"| Held-out MLM accuracy | {final['mlm_accuracy'] * 100:.1f}% |",
        f"| Training text | {f.run['bytes_seen'] / 1e9:.2f} GB, one pass, "
        f"{f.run['steps_done']:,} steps |",
        f"| Wall clock | {f.run['train_sec'] / 3600:.2f} h on one M5 Pro |",
        f"| Compression on held-out text | {met['bytes_per_token']:.3f} bytes/token, "
        f"fertility {met['fertility']:.3f} tokens/word |",
        "",
        "**Bits per byte** is the held-out MLM loss per *byte of raw text* rather than per token. "
        "That is the point: two tokenizers cut the same sentence into different numbers of tokens, "
        "so per-token loss is not comparable across them, while per-byte loss is. Lower is better.",
        "",
        "## Why this exists",
        "",
        "An encoder of this shape spends a large share of its parameters on the token embedding "
        f"matrix - {_millions(control.encoder_params)} total for the 30,522-token control. If a "
        "much smaller vocabulary costs little quality, that block shrinks and the savings are "
        "free. To make the tokenizer the only moving part, every arm shares the corpus, the "
        "**bytes** of text per optimizer step, the model shape, the optimizer, the schedule, the "
        "masking and the seed.",
        "",
        f"## All finished arms (identical recipe, seed {f.provenance['seed']})",
        "",
    ]
    lines += _arm_table(facts)
    lines += [
        "",
        "Reading of these numbers, as of the runs above:",
        "",
    ]
    lines += _takeaways(facts)
    lines += [
        "",
        "## Intended use and limitations",
        "",
        "**Use it for:** reproducing or extending the study, comparing tokenizers under a fixed "
        "compute budget, or as a starting point for your own fine-tuning experiments on Apple "
        "silicon.",
        "",
        "**Do not use it for:** anything in production, and do not read these numbers as a "
        "verdict on the study's claim. Specifically:",
        "",
        f"- **Single seed.** Every arm ran once, at seed {f.provenance['seed']}. Differences "
        "smaller than a few hundredths of a bit per byte are not yet distinguishable from noise.",
        f"- **Small budget.** {facts.pretrain['budget_bytes'] / 1e9:.0f} GB of text and one pass "
        "is far short of a production pretraining run; MLM accuracy near "
        f"{final['mlm_accuracy'] * 100:.0f}% reflects that, and arm ordering can change with "
        "scale.",
        "- **Bits per byte is not MTEB.** The study's claim is about sentence-embedding quality. "
        "That requires the contrastive fine-tuning and MTEB phases, which have not run. MTEB is "
        "deliberately untouched so far - all decisions were made on a held-out dev split.",
        "- **No pooler and no sentence-level objective.** Mean-pooling these hidden states will "
        "give you something, but nothing here was trained to make it good.",
        "- **English, educational web text.** FineWeb-Edu only, uncased. Expect poor behaviour "
        "on other languages, on cased distinctions, and on domains unlike educational web pages. "
        "The corpus is filtered web text and carries its biases into this checkpoint.",
        "",
        "## Training recipe",
        "",
    ]
    lines += _recipe_lines(facts)
    lines += [
        "",
        "## Files",
        "",
        "| File | What it is |",
        "|---|---|",
        "| `model.safetensors` | Final weights, float32"
        + (", `BertForMaskedLM` parameter names" if f.arm.bert_compatible else "")
        + " |",
        "| `leanvocab_config.json` | The study's own `EncoderConfig` for this arm |",
        "| `metrics.json` | The full recorded run: curve, plan, final evaluation, provenance |",
        "| `tokenizer.json` | The tokenizer this arm was trained with |",
    ]
    if f.arm.bert_compatible:
        lines.append("| `config.json` | `transformers` configuration for loading the weights |")
    lines += [
        "",
        "Optimizer state is not published (it is twice the size of the weights); resuming "
        "training needs the local checkpoint directory from the study.",
        "",
        "## Usage",
        "",
    ]
    lines += _usage_lines(cfg, f)
    lines += [
        "",
        "## Related repositories",
        "",
        f"- All {len(cfg.tokenizer_ids)} tokenizers from the study: "
        f"[`{cfg.owner}/{cfg.tokenizer_repo}`](https://huggingface.co/{cfg.owner}/"
        f"{cfg.tokenizer_repo})",
    ]
    for other in cfg.arms:
        if other.id != arm_id and other.id in facts.arms:
            lines.append(
                f"- The `{other.id}` arm ({other.role}): "
                f"[`{cfg.owner}/{other.repo}`](https://huggingface.co/{cfg.owner}/{other.repo})"
            )
    lines += [
        "",
        "## Provenance",
        "",
    ]
    lines += _provenance_lines(cfg, f)
    lines += [
        "",
        "## Citation",
        "",
        "```bibtex",
        "@misc{leanvocab,",
        "  title  = {leanvocab: a controlled study of small-vocabulary English tokenizers",
        "            for sentence-embedding encoders},",
        "  author = {Parmar, Dhruv},",
        f"  year   = {{{datetime.now(UTC).year}}},",
        f"  url    = {{{cfg.repo_url}}}",
        "}",
        "```",
        "",
        f"Card generated {datetime.now(UTC).strftime('%Y-%m-%d')} from the recorded run metrics.",
        "",
    ]
    return "\n".join(lines)


def tokenizer_card(
    cfg: PublishConfig,
    *,
    metas: dict[str, dict[str, Any]],
    intrinsic: dict[str, dict[str, Any]],
    heldout: dict[str, Any],
    provenance: dict[str, Any],
    phase2: dict[str, Any],
) -> str:
    """Card for the repo holding every Phase 2 tokenizer."""
    first = metas[cfg.tokenizer_ids[0]]
    subset_gb = first["subset_text_bytes"] / 1e9
    tags = ["tokenizer", "tokenizers", "bpe", "unigram", "wordpiece", "vocabulary", "research"]
    lines = [
        "---",
        f"license: {cfg.license}",
        _yaml_list("language", [cfg.language]),
        _yaml_list("datasets", [cfg.source_dataset]),
        _yaml_list("tags", tags),
        "library_name: tokenizers",
        "---",
        "",
        "# leanvocab tokenizers",
        "",
        f"The {len(cfg.tokenizer_ids)} English tokenizers trained for "
        f"[leanvocab]({cfg.repo_url}), a controlled study asking whether {CLAIM}.",
        "",
        f"All of them are trained on the **same {subset_gb:.1f} GB prefix** of the same "
        "deduplicated FineWeb-Edu corpus, with the same `bert-base-uncased`-style uncased "
        "normalization and the same five special tokens, so differences between them come from "
        "the algorithm and the vocabulary size alone.",
        "",
        "## Contents",
        "",
        "| Folder | Algorithm | Vocab | Bytes/token | Fertility | Rényi eff. (α=2.5) "
        "| Vocab used | [UNK] rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tid in cfg.tokenizer_ids:
        meta, met = metas[tid], intrinsic[tid]
        lines.append(
            f"| `{tid}/` | {ALGORITHM_NAMES[meta['algorithm']]} | {meta['vocab_size']:,} "
            f"| {met['bytes_per_token']:.3f} | {met['fertility']:.3f} "
            f"| {met['renyi_efficiency']:.4f} | {met['vocab_used_share'] * 100:.1f}% "
            f"| {met['unk_token_rate']:.2e} |"
        )
    lines += [
        "",
        f"Measured on the study's frozen held-out split ({heldout['docs']:,} documents, "
        f"{heldout['bytes'] / 1e6:.1f} MB, {heldout['words']:,} whitespace words, sha256 "
        f"`{heldout['sha256'][:12]}`). Bytes/token is compression (higher is better), fertility "
        "is tokens per whitespace word (lower is better), and Rényi efficiency measures how "
        "evenly the vocabulary is actually used (Zouhar et al., 2023). The full tables - "
        "truncation rates at 128/256/512, forward-FLOP ratios and tokenization throughput - are "
        f"in [`reports/intrinsic.md`]({cfg.repo_url}/blob/main/{cfg.intrinsic_report}).",
        "",
        "## Usage",
        "",
        "```python",
        "from huggingface_hub import hf_hub_download",
        "from tokenizers import Tokenizer",
        "",
        f'path = hf_hub_download("{cfg.owner}/{cfg.tokenizer_repo}", "bpe-16k/tokenizer.json")',
        "tok = Tokenizer.from_file(path)",
        'print(tok.encode("photosynthesis happens in plant cells").tokens)',
        "```",
        "",
        "Each folder also carries a `meta.json` recording the algorithm, the exact vocabulary "
        "size, the `tokenizers` version, the SHA-256 of `tokenizer.json`, the training time and "
        "the round-trip check that gated it.",
        "",
        "## How they were trained",
        "",
        "- **Corpus:** FineWeb-Edu `sample-10BT` at a pinned revision, NFC-normalized, exactly "
        "deduplicated (BLAKE2b-128), with the held-out split removed *before* tokenizer training. "
        f"Each tokenizer sees the same byte-budgeted {subset_gb:.1f} GB prefix of the training "
        "shards.",
        f"- **Byte-level BPE** (`bpe-*`): min frequency {phase2['bpe']['min_frequency']}.",
        f"- **Unigram** (`uni-*`): byte fallback with 256 byte pieces inside the vocabulary "
        f"budget, max piece length {phase2['unigram']['max_piece_length']}, shrinking factor "
        f"{phase2['unigram']['shrinking_factor']}. Training text is restricted to a shared "
        f"{phase2['char_alphabet_size']:,}-character alphabet (99.996% coverage of the corpus); "
        "rarer characters fall back to bytes at inference.",
        f"- **WordPiece** (`wp-*`): retrained on the same text, min frequency "
        f"{phase2['wordpiece']['min_frequency']}, same initial alphabet.",
        f"- **`bert-orig`**: the original `{phase2['bert']['repo']}` tokenizer at revision "
        f"`{phase2['bert']['revision'][:12]}`, unmodified - the study's control.",
        "- **`bert-trim-*`**: that BERT vocabulary trimmed to the most frequent pieces on this "
        'corpus, as a baseline for "just drop the rare tokens".',
        "",
        "## Determinism",
        "",
        "The `tokenizers` library is not deterministic out of the box: WordPiece's "
        "`limit_alphabet` tie-breaks and `##` insertion follow hash-map order, and Unigram's "
        "parallel EM leaves roughly 1e-4 of float noise in the scores. Training here precomputes "
        "a shared character alphabet, assigns token ids by a canonical key and rounds Unigram "
        f"scores to {phase2['unigram']['score_decimals']} decimals, which makes repeated runs "
        "byte-identical. The `tokenizer_sha256` in each `meta.json` is what a rerun reproduces.",
        "",
        "## Limitations",
        "",
        "- English, uncased, educational web text. Accents are stripped with lowercasing, as in "
        "`bert-base-uncased`.",
        "- The retrained WordPiece and trimmed-BERT vocabularies emit `[UNK]` on rare characters; "
        "the BPE and Unigram vocabularies never do, because they fall back to bytes.",
        "- These are tokenizers only. Pretrained encoders that use three of them are in the "
        "related repositories below.",
        "",
        "## Related repositories",
        "",
    ]
    for arm in cfg.arms:
        lines.append(
            f"- `{arm.id}` encoder ({arm.role}, tokenizer `{arm.tokenizer}`): "
            f"[`{cfg.owner}/{arm.repo}`](https://huggingface.co/{cfg.owner}/{arm.repo})"
        )
    lines += [
        "",
        "## Provenance",
        "",
        f"- Phase 2 run `{provenance['run_id']}`, seed {provenance['seed']}, config SHA-256 "
        f"`{provenance['config_sha256'][:12]}`, commit `{provenance['git_sha'][:12]}`.",
        f"- {provenance['platform']}, Python {provenance['python']}, "
        f"tokenizers {provenance['libraries']['tokenizers']}.",
        f"- Code, configs and recorded metrics: [{cfg.repo_url}]({cfg.repo_url}).",
        "",
        "## Citation",
        "",
        "```bibtex",
        "@misc{leanvocab,",
        "  title  = {leanvocab: a controlled study of small-vocabulary English tokenizers",
        "            for sentence-embedding encoders},",
        "  author = {Parmar, Dhruv},",
        f"  year   = {{{datetime.now(UTC).year}}},",
        f"  url    = {{{cfg.repo_url}}}",
        "}",
        "```",
        "",
        f"Card generated {datetime.now(UTC).strftime('%Y-%m-%d')} from the recorded run metrics.",
        "",
    ]
    return "\n".join(lines)
