"""Round-trip checks: decode(encode(x)) against the tokenizer's own normalization of x.

- ``exact`` (byte-level BPE, Unigram with byte fallback): strings must be identical.
- ``ignore_whitespace`` (WordPiece, BERT): BERT pre-tokenization drops whitespace, so both
  sides are compared with all whitespace removed; documents containing [UNK] are counted
  separately and not compared (UNK rates are a Phase 3 metric).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tokenizers import Tokenizer

from tokstudy.tokenizers.common import RoundtripMode

MAX_REPORTED_MISMATCHES = 5
MISMATCH_SNIPPET_CHARS = 200


def _strip_ws(s: str) -> str:
    return "".join(s.split())


def roundtrip_report(
    tok: Tokenizer, texts: Sequence[str], mode: RoundtripMode, unk_token: str
) -> dict[str, Any]:
    if tok.normalizer is None:
        raise ValueError("Round-trip check requires a tokenizer with a normalizer")
    if not tok.encode_special_tokens:
        raise ValueError("Set encode_special_tokens=True so literal '[SEP]' text is not special")
    unk_id = tok.token_to_id(unk_token)
    encodings = tok.encode_batch(list(texts), add_special_tokens=False)
    decoded = tok.decode_batch([e.ids for e in encodings], skip_special_tokens=False)

    matched = unk_docs = 0
    mismatches: list[dict[str, str]] = []
    n_mismatch = 0
    for text, enc, dec in zip(texts, encodings, decoded, strict=True):
        expected = tok.normalizer.normalize_str(text)
        if mode == "exact":
            ok = dec == expected
        else:
            if unk_id is not None and unk_id in enc.ids:
                unk_docs += 1
                continue
            ok = _strip_ws(dec) == _strip_ws(expected)
        if ok:
            matched += 1
            continue
        n_mismatch += 1
        if len(mismatches) < MAX_REPORTED_MISMATCHES:
            pos = next(
                (i for i, (a, b) in enumerate(zip(dec, expected, strict=False)) if a != b),
                min(len(dec), len(expected)),
            )
            lo = max(0, pos - 40)
            mismatches.append(
                {
                    "decoded": dec[lo : lo + MISMATCH_SNIPPET_CHARS],
                    "expected": expected[lo : lo + MISMATCH_SNIPPET_CHARS],
                }
            )
    return {
        "mode": mode,
        "docs": len(texts),
        "matched": matched,
        "unk_docs_skipped": unk_docs,
        "mismatched": n_mismatch,
        "passed": n_mismatch == 0 and matched > 0,
        "mismatch_examples": mismatches,
    }
