"""Byte-level BPE (GPT-2 style pre-tokenization, no [UNK] thanks to the 256-byte alphabet)."""

from __future__ import annotations

from collections.abc import Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from tokstudy.tokenizers.common import BpeConfig, Phase2Config, build_normalizer


def train_bpe(
    cfg: Phase2Config, family: BpeConfig, vocab_size: int, batches: Iterator[list[str]], n: int
) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.normalizer = build_normalizer(cfg.normalizer)
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=family.min_frequency,
        special_tokens=cfg.special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(batches, trainer=trainer, length=n)
    return tok
