"""Exact deduplication by content hash."""

from __future__ import annotations

import hashlib

DIGEST_BYTES = 16


class ExactDeduper:
    """Tracks 128-bit BLAKE2b digests of normalized text; ``seen`` is True for repeats."""

    def __init__(self) -> None:
        self._digests: set[bytes] = set()
        self.n_checked = 0
        self.n_duplicates = 0
        self.duplicate_bytes = 0

    def seen(self, text_bytes: bytes) -> bool:
        self.n_checked += 1
        digest = hashlib.blake2b(text_bytes, digest_size=DIGEST_BYTES).digest()
        if digest in self._digests:
            self.n_duplicates += 1
            self.duplicate_bytes += len(text_bytes)
            return True
        self._digests.add(digest)
        return False

    def stats(self) -> dict[str, int | float]:
        return {
            "docs_checked": self.n_checked,
            "duplicates_removed": self.n_duplicates,
            "duplicate_bytes_removed": self.duplicate_bytes,
            "duplicate_rate": self.n_duplicates / self.n_checked if self.n_checked else 0.0,
        }
