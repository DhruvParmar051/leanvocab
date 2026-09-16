"""Order-independent held-out assignment."""

from __future__ import annotations

import hashlib

_HASH_BYTES = 8
_HASH_MAX = 2 ** (8 * _HASH_BYTES)


def heldout_key(doc_id: str, seed: int) -> int:
    """Keyed 64-bit hash of ``doc_id``; uniform over [0, 2**64)."""
    key = seed.to_bytes(8, "little")
    digest = hashlib.blake2b(doc_id.encode("utf-8"), digest_size=_HASH_BYTES, key=key).digest()
    return int.from_bytes(digest, "big")


def is_heldout_candidate(key: int, fraction: float) -> bool:
    return key < int(fraction * _HASH_MAX)
