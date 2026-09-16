"""Streaming reader for the source corpus."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from tokstudy.data.schema import Document, SourceConfig

logger = logging.getLogger(__name__)


def iter_documents(source: SourceConfig) -> Iterator[Document]:
    """Stream documents from the Hub in the dataset's fixed file order (no shuffling)."""
    from datasets import load_dataset

    logger.info(
        "Streaming %s/%s split=%s revision=%s",
        source.path,
        source.name,
        source.split,
        source.revision,
    )
    columns = [source.id_field, source.text_field, source.url_field, source.token_count_field]
    ds = load_dataset(
        source.path,
        name=source.name,
        split=source.split,
        revision=source.revision,
        streaming=True,
    ).select_columns(columns)
    for row in ds:
        text = row[source.text_field]
        doc_id = row[source.id_field]
        if not isinstance(text, str) or not isinstance(doc_id, str):
            raise TypeError(f"Unexpected row types: id={type(doc_id)}, text={type(text)}")
        yield Document(
            id=doc_id,
            text=text,
            url=row[source.url_field] or "",
            token_count_gpt2=int(row[source.token_count_field]),
        )
