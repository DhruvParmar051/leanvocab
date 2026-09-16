"""Source corpus reader: downloads pinned Parquet files from the Hub and reads them locally.

Files are fetched one at a time, only when the previous one is exhausted, so a byte budget
smaller than the full sample downloads only the files it needs. Reading local files (instead
of ``datasets`` streaming) is faster and avoids an Arrow I/O pool deadlock at exit.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from tokstudy.data.schema import Document, SourceConfig
from tokstudy.data.shard import sha256_file

logger = logging.getLogger(__name__)


class HubParquetSource:
    """Iterates documents from ``source.file_prefix`` in sorted filename order."""

    def __init__(self, source: SourceConfig, raw_dir: Path) -> None:
        self.source = source
        self.raw_dir = raw_dir
        self.files_used: list[dict[str, Any]] = []
        if source.disable_xet:
            # Read by huggingface_hub at import time, so it must be set before importing it.
            os.environ["HF_HUB_DISABLE_XET"] = "1"
        from huggingface_hub import HfApi

        logger.info("Hub transfer backend: %s", "http" if source.disable_xet else "xet")
        api = HfApi()
        listing = api.list_repo_tree(
            source.path,
            path_in_repo=source.file_prefix.rstrip("/"),
            repo_type="dataset",
            revision=source.revision,
            expand=True,
        )
        self._remote = sorted(
            (f for f in listing if getattr(f, "lfs", None) and f.path.endswith(".parquet")),
            key=lambda f: f.path,
        )
        if not self._remote:
            raise FileNotFoundError(
                f"No parquet files under {source.path}@{source.revision}/{source.file_prefix}"
            )
        logger.info("Source has %d parquet files", len(self._remote))

    def _fetch(self, remote: Any) -> Path:
        from huggingface_hub import hf_hub_download

        logger.info("Fetching %s (%.2f GB)", remote.path, remote.size / 1e9)
        start = time.monotonic()
        local = Path(
            hf_hub_download(
                self.source.path,
                remote.path,
                repo_type="dataset",
                revision=self.source.revision,
                local_dir=self.raw_dir,
            )
        )
        download_sec = time.monotonic() - start
        digest = sha256_file(local)
        if digest != remote.lfs.sha256:
            raise ValueError(
                f"SHA256 mismatch for {remote.path}: got {digest}, expected {remote.lfs.sha256}"
            )
        logger.info(
            "Fetched %s (%.2f GB) in %.0fs, sha256 verified",
            remote.path,
            remote.size / 1e9,
            download_sec,
        )
        self.files_used.append(
            {
                "path": remote.path,
                "size": remote.size,
                "sha256": digest,
                "fetch_sec": round(download_sec, 1),
            }
        )
        return local

    def __iter__(self) -> Generator[Document, None, None]:
        s = self.source
        columns = [s.id_field, s.text_field, s.url_field, s.token_count_field]
        for remote in self._remote:
            local = self._fetch(remote)
            parquet = pq.ParquetFile(local)
            try:
                for batch in parquet.iter_batches(batch_size=s.read_batch_rows, columns=columns):
                    cols = batch.to_pydict()
                    for doc_id, text, url, tokens in zip(
                        cols[s.id_field],
                        cols[s.text_field],
                        cols[s.url_field],
                        cols[s.token_count_field],
                        strict=True,
                    ):
                        if not isinstance(text, str) or not isinstance(doc_id, str):
                            raise TypeError(
                                f"Unexpected row types in {remote.path}: "
                                f"id={type(doc_id)}, text={type(text)}"
                            )
                        yield Document(
                            id=doc_id, text=text, url=url or "", token_count_gpt2=int(tokens)
                        )
            finally:
                parquet.close()

    def provenance(self) -> dict[str, Any]:
        """Files actually opened (a budgeted run reads a prefix of the file list)."""
        return {
            "files_used": [
                {k: v for k, v in f.items() if k != "fetch_sec"} for f in self.files_used
            ]
        }
