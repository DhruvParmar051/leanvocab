"""WordPiece retrained on our corpus with BERT's pre-tokenization and decoding."""

from __future__ import annotations

from collections.abc import Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from tokstudy.tokenizers.common import Phase2Config, WordPieceConfig, build_normalizer

CONTINUATION_PREFIX = "##"
# BERT's value; longer words map to [UNK].
MAX_INPUT_CHARS_PER_WORD = 100


def train_wordpiece(
    cfg: Phase2Config,
    family: WordPieceConfig,
    vocab_size: int,
    batches: Iterator[list[str]],
    n: int,
) -> Tokenizer:
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
        limit_alphabet=family.limit_alphabet,
        continuing_subword_prefix=CONTINUATION_PREFIX,
        show_progress=False,
    )
    tok.train_from_iterator(batches, trainer=trainer, length=n)
    return tok
