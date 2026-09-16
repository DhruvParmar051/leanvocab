"""Corpus-level text normalization. Anything beyond NFC belongs to the tokenizer families."""

from __future__ import annotations

import unicodedata


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)
