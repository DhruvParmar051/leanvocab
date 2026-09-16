"""Markdown report and Pareto plot for Phase 3."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from leanvocab.intrinsic.metrics import EncoderShape  # noqa: E402

if TYPE_CHECKING:
    from leanvocab.intrinsic.run_intrinsic import Phase3Config

REPORT_FILE = "intrinsic.md"
PLOT_FILE = "intrinsic_pareto.png"

# Categorical slots 1-3 of the validated default palette (all-pairs safe for scatter);
# the BERT reference/baselines are drawn in neutral ink, not a fourth hue.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
FAMILY_STYLE: dict[str, tuple[str, str, str]] = {
    "bpe": ("#2a78d6", "o", "Byte-level BPE"),
    "uni": ("#eb6834", "s", "Unigram (byte fallback)"),
    "wp": ("#1baf7a", "^", "WordPiece (retrained)"),
    "bert": ("#6f6e69", "D", "BERT (original / trimmed)"),
}


# Small horizontal offsets (millions of params) so equal-size tokenizers stay visible.
FAMILY_DODGE = {"bert": -0.27, "bpe": -0.09, "uni": 0.09, "wp": 0.27}
LABEL_OFFSET_X = 0.6
# Points closer than this on x (millions) share one label column.
LABEL_COLUMN_WIDTH = 1.0
LABEL_GAP_FRACTION = 0.04


def _family(tid: str) -> str:
    return tid.split("-")[0]


def plot_pareto(
    path: Path, results: dict[str, dict[str, Any]], reference: str, max_ratio: float
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ref_len = results[reference]["doc_len_mean"]
    ax.axhline(ref_len * max_ratio, color=TEXT_SECONDARY, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(
        0.2,
        ref_len * max_ratio,
        f"length gate: {max_ratio:.2f}× {reference} (candidates above are dropped)",
        color=TEXT_SECONDARY,
        fontsize=8,
        va="top",
    )

    points = sorted(
        (r["token_embedding_params"] / 1e6, r["doc_len_mean"], tid) for tid, r in results.items()
    )
    frontier = []
    best = float("inf")
    for x, y, _ in points:
        if y < best:
            frontier.append((x, y))
            best = y
    ax.step(
        [x for x, _ in frontier],
        [y for _, y in frontier],
        where="post",
        color=GRID,
        lw=2,
        zorder=1,
        label="Pareto frontier",
    )

    y_values = [y for _, y, _ in points] + [ref_len * max_ratio]
    min_gap = LABEL_GAP_FRACTION * (max(y_values) - min(y_values))
    for family, (color, marker, label) in FAMILY_STYLE.items():
        fam = [(x + FAMILY_DODGE[family], y) for x, y, t in points if _family(t) == family]
        if not fam:
            continue
        ax.scatter(
            [x for x, _ in fam],
            [y for _, y in fam],
            s=64,
            color=color,
            marker=marker,
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
            label=label,
        )

    # Tokenizers of equal vocab size share an x position: stack their labels in a
    # column to the right, spaced at least `min_gap` apart, with thin leader lines.
    columns: list[tuple[float, list[tuple[float, str]]]] = []
    for x, y, t in points:  # points are sorted by x
        if columns and x - columns[-1][0] < LABEL_COLUMN_WIDTH:
            columns[-1][1].append((y, t))
        else:
            columns.append((x, [(y, t)]))
    x_of = {t: x for x, _, t in points}
    for _, group in columns:
        x = max(x_of[t] for _, t in group)
        group.sort()
        label_y: list[float] = []
        for y, _ in group:
            label_y.append(y if not label_y else max(y, label_y[-1] + min_gap))
        shift = (label_y[-1] - group[-1][0]) / 2 if len(group) > 1 else 0.0
        for (y, t), ly in zip(group, label_y, strict=True):
            px = x_of[t] + FAMILY_DODGE[_family(t)]
            tx = x + LABEL_OFFSET_X
            ty = ly - shift
            ax.plot([px, tx - 0.05], [y, ty], color=GRID, lw=0.8, zorder=2)
            ax.text(tx, ty, t, fontsize=8, color=TEXT_PRIMARY, va="center")

    ax.set_xlabel("Token-embedding parameters (millions)", color=TEXT_SECONDARY)
    ax.set_ylabel("Mean tokens per held-out document", color=TEXT_SECONDARY)
    ax.set_title(
        "Embedding size vs. sequence length (lower-left is better)",
        color=TEXT_PRIMARY,
        fontsize=11,
        loc="left",
    )
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.set_xlim(0, max(x for x, _, _ in points) + 2.0)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="lower left")
    ax.text(
        1.0,
        -0.12,
        "Markers are nudged sideways by family; equal vocab sizes share one x value.",
        transform=ax.transAxes,
        ha="right",
        fontsize=7,
        color=TEXT_SECONDARY,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def write_report(
    report_dir: Path,
    results: dict[str, dict[str, Any]],
    proposal: dict[str, Any],
    cfg: Phase3Config,
    shape: EncoderShape,
    heldout: dict[str, Any],
    run_id: str,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_pareto(report_dir / PLOT_FILE, results, cfg.reference, cfg.gate.max_len_ratio)
    ids = list(cfg.tokenizers)
    lens = cfg.max_lens
    lines: list[str] = [
        "# Phase 3: intrinsic tokenizer evaluation",
        "",
        f"Run `{run_id}`, generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by "
        "`leanvocab.intrinsic.run_intrinsic`. All numbers are on the frozen held-out split "
        f"({heldout['docs']:,} docs, {heldout['bytes'] / 1e6:.1f} MB, "
        f"{heldout['words']:,} whitespace words, {heldout['paragraphs']:,} paragraphs "
        f">= {cfg.min_paragraph_chars} chars; sha256 `{heldout['sha256'][:12]}`).",
        "",
        "## Size, compression and vocabulary use",
        "",
        "| Tokenizer | Vocab | Total params | Bytes/token | Fertility | "
        f"Words in >={cfg.split_word_min_pieces} pieces | Rényi eff. (α={cfg.renyi_alpha}) | "
        "Vocab used | [UNK] tokens | Docs with [UNK] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tid in ids:
        r = results[tid]
        lines.append(
            f"| {tid} | {r['vocab_size']:,} | {r['total_params'] / 1e6:.2f}M "
            f"({r['params_ratio_vs_reference']:.2f}×) | {r['bytes_per_token']:.3f} | "
            f"{r['fertility']:.3f} | {_pct(r['share_words_split_ge_min_pieces'])} | "
            f"{r['renyi_efficiency']:.4f} | {_pct(r['vocab_used_share'])} | "
            f"{r['unk_token_rate']:.2e} | {_pct(r['unk_doc_share'])} |"
        )
    lines += [
        "",
        "Total params are for the Phase 4 encoder "
        f"({shape.n_layers} layers, d={shape.d_model}, FFN {shape.d_ffn}, "
        f"{shape.max_len} positions, tied embeddings, no MLM head); the ratio is against "
        f"{cfg.reference}. Rényi efficiency divides by log2 of the full vocab size (Zouhar et "
        "al., 2023); `metrics.json` also has the observed-types variant, and both match "
        "`tokenization-scorer` in the unit tests.",
        "",
        "## Sequence length, truncation and compute",
        "",
        "| Tokenizer | Mean doc len | vs ref | p95 doc len | "
        + " | ".join(f"Docs cut @{m}" for m in lens)
        + " | Mean para len | "
        + " | ".join(f"Paras cut @{m}" for m in lens)
        + f" | FLOPs/doc @{shape.max_len} | FLOPs/para | Length gate |",
        "|---|---:|---:|---:|"
        + "---:|" * len(lens)
        + "---:|"
        + "---:|" * len(lens)
        + "---:|---:|:---:|",
    ]
    for tid in ids:
        r = results[tid]
        gate = "pass" if r["passes_len_gate"] else "**drop**"
        lines.append(
            f"| {tid} | {r['doc_len_mean']:.0f} | {r['len_ratio_vs_reference']:.3f}× | "
            f"{r['doc_len_p95']:.0f} | "
            + " | ".join(_pct(r["doc_truncated"][str(m)]) for m in lens)
            + f" | {r['para_len_mean']:.1f} | "
            + " | ".join(_pct(r["para_truncated"][str(m)], 2) for m in lens)
            + f" | {r['flops_doc_ratio_vs_reference']:.3f}× | "
            f"{r['flops_para_ratio_vs_reference']:.3f}× | {gate} |"
        )
    lines += [
        "",
        "Lengths exclude `[CLS]`/`[SEP]`; truncation counts a sequence as cut when its tokens "
        f"plus {cfg.n_special_tokens} specials exceed the limit. FLOPs are forward-pass "
        "estimates (2·N_layers·L + 2·n_layers·L²·d) with L capped at "
        f"{shape.max_len}, relative to {cfg.reference}. Most held-out documents are longer "
        "than any limit, so paragraph-level numbers are the closer proxy for "
        "sentence-embedding inputs.",
        "",
        "## Tokenization throughput (M5 Pro)",
        "",
        "| Tokenizer | Single thread (docs/s) | Single thread (MB/s) | "
        "Batched, all cores (docs/s) | Batched (MB/s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for tid in ids:
        t = results[tid]["throughput"]
        lines.append(
            f"| {tid} | {t['single_thread_docs_per_sec']:.0f} | "
            f"{t['single_thread_mb_per_sec']:.2f} | {t['batched_docs_per_sec']:.0f} | "
            f"{t['batched_mb_per_sec']:.1f} |"
        )
    lines += [
        "",
        f"Single thread: `encode()` over the first {cfg.throughput.single_thread_docs:,} docs; "
        f"batched: `encode_batch()` in chunks of {cfg.throughput.batch_docs}; median of "
        f"{cfg.throughput.repeats} runs after a warm-up. Includes offset computation.",
        "",
        "## Pareto view",
        "",
        f"![Token-embedding parameters vs. mean document length]({PLOT_FILE})",
        "",
        "## Go/no-go #1: proposed finalists",
        "",
        f"Rule applied: drop any candidate whose mean document length exceeds "
        f"{cfg.gate.max_len_ratio}× {cfg.reference}; among the rest, rank each claim vocab "
        "size by bytes/token (inference cost), ties by Rényi efficiency. **This is a "
        "proposal for review, not a decision.**",
        "",
    ]
    for v, p in proposal["by_vocab_size"].items():
        ranked = ", ".join(p["ranked_passing"]) or "none"
        dropped = ", ".join(p["dropped_by_gate"]) or "none"
        lines.append(f"- **{int(v) // 1000}k:** ranked {ranked}; dropped by gate: {dropped}.")
    lines += [
        "",
        f"Always in Phase 4 regardless: the control `{cfg.reference}`, Baseline A "
        "(factorized 30,522×128 embeddings) and Baseline B (`bert-trim-*` at the finalist "
        "sizes). 32k tokenizers have more parameters than the control and are reference "
        "points only.",
        "",
    ]
    path = report_dir / REPORT_FILE
    path.write_text("\n".join(lines))
    return path
