"""SentencePiece-style Unigram with byte fallback.

HF's UnigramTrainer does not create byte pieces, so we train ``vocab_size - 256`` pieces and
append ``<0x00>``..``<0xFF>`` afterwards (as SentencePiece does within its vocab budget), then
rebuild the model with ``byte_fallback=True``. The total vocab is exactly ``vocab_size``.

The trainer's parallel EM is not bit-reproducible: scores vary by ~1e-4 between runs (the
piece set was identical, and segmentation of 3k held-out docs unchanged, in our checks).
Scores are rounded to ``score_decimals`` and ids are assigned by (length, piece), which
depends only on the piece set; Viterbi segmentation uses scores, not ids. The saved
artifact is the reference.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from leanvocab.tokenizers.common import Phase2Config, UnigramConfig, build_normalizer

N_BYTE_PIECES = 256
METASPACE = "▁"
# Byte pieces are only reached through fallback; give them a score below every learned piece.
BYTE_PIECE_SCORE_MARGIN = 1.0


def byte_piece(b: int) -> str:
    return f"<0x{b:02X}>"


def train_unigram(
    cfg: Phase2Config,
    family: UnigramConfig,
    vocab_size: int,
    batches: Iterator[list[str]],
    n: int,
) -> Tokenizer:
    if vocab_size <= N_BYTE_PIECES + len(cfg.special_tokens):
        raise ValueError(f"vocab_size {vocab_size} too small for byte fallback")
    tok = Tokenizer(models.Unigram())
    tok.normalizer = build_normalizer(cfg.normalizer)
    # "never": no space is invented before the first word, so decoding is lossless
    # (mirrors byte-level BPE with add_prefix_space=False).
    tok.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement=METASPACE, prepend_scheme="never", split=True
    )
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size - N_BYTE_PIECES,
        special_tokens=cfg.special_tokens,
        unk_token=cfg.unk_token,
        shrinking_factor=family.shrinking_factor,
        n_sub_iterations=family.n_sub_iterations,
        max_piece_length=family.max_piece_length,
        show_progress=False,
    )
    tok.train_from_iterator(batches, trainer=trainer, length=n)

    model_state = json.loads(tok.to_str())["model"]
    pieces: list[tuple[str, float]] = [(p, float(s)) for p, s in model_state["vocab"]]
    trained = len(pieces)
    if trained != vocab_size - N_BYTE_PIECES:
        raise ValueError(f"Unigram trainer produced {trained} pieces, expected {vocab_size - 256}")
    specials = [(p, s) for p, s in pieces if p in cfg.special_tokens]
    if [p for p, _ in specials] != cfg.special_tokens:
        raise ValueError("Unigram trainer did not place special tokens first, in order")
    learned = sorted(
        ((p, round(s, family.score_decimals)) for p, s in pieces if p not in cfg.special_tokens),
        key=lambda ps: (len(ps[0]), ps[0]),
    )
    pieces = specials + learned
    existing = {p for p, _ in pieces}
    min_score = min(s for _, s in learned)
    for b in range(N_BYTE_PIECES):
        piece = byte_piece(b)
        if piece in existing:
            raise ValueError(f"Learned vocab already contains byte piece {piece}")
        pieces.append((piece, min_score - BYTE_PIECE_SCORE_MARGIN))

    tok.model = models.Unigram(pieces, unk_id=model_state["unk_id"], byte_fallback=True)
    tok.decoder = decoders.Sequence(
        [decoders.Replace(METASPACE, " "), decoders.ByteFallback(), decoders.Fuse()]
    )
    return tok
