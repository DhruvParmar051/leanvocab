"""WordPiece retrained on our corpus with BERT's pre-tokenization and decoding."""

from __future__ import annotations

from collections.abc import Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from leanvocab.tokenizers.common import Phase2Config, WordPieceConfig, build_normalizer

CONTINUATION_PREFIX = "##"
# BERT's value; longer words map to [UNK].
MAX_INPUT_CHARS_PER_WORD = 100


def train_wordpiece(
    cfg: Phase2Config,
    family: WordPieceConfig,
    vocab_size: int,
    batches: Iterator[list[str]],
    n: int,
    alphabet: list[str],
) -> Tokenizer:
    """Train WordPiece with a fixed, precomputed alphabet (see ``alphabet.py``)."""
    tok = Tokenizer(
        models.WordPiece(
            unk_token=cfg.unk_token,
            max_input_chars_per_word=MAX_INPUT_CHARS_PER_WORD,
        )
    )
    tok.normalizer = build_normalizer(cfg.normalizer)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.decoder = decoders.WordPiece(prefix=CONTINUATION_PREFIX, cleanup=True)
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=family.min_frequency,
        special_tokens=cfg.special_tokens,
        initial_alphabet=alphabet,
        limit_alphabet=len(alphabet),
        continuing_subword_prefix=CONTINUATION_PREFIX,
        show_progress=False,
    )
    tok.train_from_iterator(batches, trainer=trainer, length=n)
    return canonicalize_ids(tok, cfg)


def canonicalize_ids(tok: Tokenizer, cfg: Phase2Config) -> Tokenizer:
    """Assign ids from the token set alone: specials, then (piece length, is-continuation, piece).

    The trainer emits tokens in an order that depends on hash-map iteration and on how tied
    merges are popped, so ids differ between runs even when the vocabulary is the same.
    WordPiece segmentation (greedy longest match) depends only on the set, not on ids.
    """
    vocab = tok.get_vocab()
    by_id = sorted(vocab, key=vocab.__getitem__)
    if by_id[: len(cfg.special_tokens)] != cfg.special_tokens:
        raise ValueError("WordPiece trainer did not place special tokens first, in order")
    ordered = list(cfg.special_tokens) + sorted(
        set(vocab) - set(cfg.special_tokens), key=_canonical_key
    )
    out = Tokenizer(
        models.WordPiece(
            {t: i for i, t in enumerate(ordered)},
            unk_token=cfg.unk_token,
            continuing_subword_prefix=CONTINUATION_PREFIX,
            max_input_chars_per_word=MAX_INPUT_CHARS_PER_WORD,
        )
    )
    out.normalizer = tok.normalizer
    out.pre_tokenizer = tok.pre_tokenizer
    out.decoder = tok.decoder
    out.add_special_tokens(cfg.special_tokens)
    return out


def _canonical_key(token: str) -> tuple[int, bool, str]:
    piece = token.removeprefix(CONTINUATION_PREFIX)
    return len(piece), token.startswith(CONTINUATION_PREFIX), piece
