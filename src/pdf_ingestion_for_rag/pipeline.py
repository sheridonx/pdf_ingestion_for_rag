"""End-to-end ingestion: PDF -> parsed blocks -> chunks -> finalized `Chunk`s.

This module owns everything that requires document-level context: hashing and
identity, chunk linking (previous/next), section-context prefixing, optional
language detection, and provenance metadata.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from .chunker import HierarchicalChunker, _PendingChunk
from .config import IngestionConfig
from .models import Chunk, ChunkMetadata
from .pdf_parser import DocumentParser, PDFParseError, open_document
from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)

# Stable namespace so chunk ids are reproducible across runs/machines.
_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


@dataclass
class IngestionResult:
    """Outcome of ingesting one document."""

    doc_id: str
    source_path: str
    file_hash: str
    page_count: int
    chunks: list[Chunk]
    warnings: list[str]

    def __len__(self) -> int:
        return len(self.chunks)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _detect_language(text: str) -> str | None:
    try:
        from langdetect import detect  # optional dependency
        from langdetect.lang_detect_exception import LangDetectException

        try:
            return detect(text)
        except LangDetectException:
            return None
    except ImportError:
        logger.warning(
            "detect_language=True but 'langdetect' is not installed "
            "(pip install pdf-ingestion-for-rag[lang]); skipping."
        )
        return None


def _finalize(
    pending: list[_PendingChunk],
    *,
    doc_id: str,
    source_path: str,
    file_hash: str,
    config: IngestionConfig,
) -> list[Chunk]:
    """Attach ids, links, provenance, and optional prefixes to pending chunks."""
    filename = os.path.basename(source_path)
    total = len(pending)
    chunk_ids = [
        str(uuid.uuid5(_NAMESPACE, f"{doc_id}:{i}")) for i in range(total)
    ]

    chunks: list[Chunk] = []
    for i, pc in enumerate(pending):
        # Stored text stays raw/pristine; section context is composed at embed
        # time in Chunk.to_embedding_input() from the structured section_path.
        text = pc.text

        metadata = ChunkMetadata(
            chunk_id=chunk_ids[i],
            doc_id=doc_id,
            source_path=source_path,
            source_filename=filename,
            file_hash=file_hash,
            chunk_index=i,
            total_chunks=total,
            previous_chunk_id=chunk_ids[i - 1] if i > 0 else None,
            next_chunk_id=chunk_ids[i + 1] if i < total - 1 else None,
            page_start=pc.page_start,
            page_end=pc.page_end,
            section_title=pc.section_path[-1] if pc.section_path else None,
            section_path=pc.section_path,
            heading_level=pc.heading_level,
            contains_tables=pc.contains_tables,
            contains_images=pc.contains_images,
            token_count=pc.token_count,
            char_count=len(text),
            language=_detect_language(pc.text) if config.detect_language else None,
        )
        chunks.append(Chunk(text=text, metadata=metadata))
    return chunks


def _ingest(
    source: str | bytes,
    source_path: str,
    file_bytes: bytes,
    config: IngestionConfig,
) -> IngestionResult:
    tokenizer = Tokenizer(config.tokenizer_encoding)
    parser = DocumentParser(config)
    chunker = HierarchicalChunker(config, tokenizer)

    doc = open_document(source, config)
    try:
        page_count = doc.page_count
        blocks = parser.parse(doc)
    finally:
        doc.close()

    file_hash = _sha256(file_bytes)
    doc_id = str(uuid.uuid5(_NAMESPACE, file_hash))

    warnings: list[str] = []
    if not blocks:
        warnings.append(
            "No text extracted. The PDF may be scanned/image-only; run OCR upstream."
        )

    pending = chunker.chunk(blocks)
    chunks = _finalize(
        pending,
        doc_id=doc_id,
        source_path=source_path,
        file_hash=file_hash,
        config=config,
    )
    logger.info(
        "Ingested %s: %d pages, %d blocks, %d chunks",
        source_path,
        page_count,
        len(blocks),
        len(chunks),
    )
    return IngestionResult(
        doc_id=doc_id,
        source_path=source_path,
        file_hash=file_hash,
        page_count=page_count,
        chunks=chunks,
        warnings=warnings,
    )


def ingest_pdf(
    path: str | os.PathLike, config: IngestionConfig | None = None
) -> IngestionResult:
    """Ingest a PDF file from disk into an `IngestionResult`.

    Raises `PDFParseError` if the file is missing, empty, or unreadable.
    """
    config = config or IngestionConfig()
    p = Path(path)
    if not p.is_file():
        raise PDFParseError(f"File not found: {p}")
    file_bytes = p.read_bytes()
    if not file_bytes:
        raise PDFParseError(f"File is empty: {p}")
    return _ingest(str(p), str(p.resolve()), file_bytes, config)


def ingest_pdf_bytes(
    data: bytes,
    source_name: str = "in-memory.pdf",
    config: IngestionConfig | None = None,
) -> IngestionResult:
    """Ingest a PDF held in memory (e.g. from an upload or object store)."""
    config = config or IngestionConfig()
    if not data:
        raise PDFParseError("Empty PDF bytes.")
    return _ingest(data, source_name, data, config)
