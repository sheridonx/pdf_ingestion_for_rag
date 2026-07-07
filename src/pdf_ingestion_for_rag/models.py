"""Typed data models for the ingestion pipeline.

`Block` is the internal, layout-level unit produced by the parser.
`Chunk` / `ChunkMetadata` are the public, retrieval-level output.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class BlockType(str, enum.Enum):
    """Kind of layout element extracted from a page, in reading order."""

    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    IMAGE = "image"


class Block(BaseModel):
    """A single layout element on a page (internal representation).

    Blocks are ordered in approximate reading order and later grouped into
    sections and chunks. Tables carry their Markdown rendering as `text`;
    images carry an optional caption (or empty text) but still mark presence.
    """

    type: BlockType
    text: str = ""
    page_number: int  # 1-indexed
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) in PDF points
    font_size: float | None = None
    is_bold: bool = False
    heading_level: int | None = None  # set only for HEADING blocks

    model_config = {"frozen": False}


class ChunkMetadata(BaseModel):
    """Retrieval-time metadata attached to every chunk.

    Everything a retriever, reranker, or citation layer might need without
    re-opening the source PDF.
    """

    # Identity & provenance
    chunk_id: str
    doc_id: str
    source_path: str
    source_filename: str
    file_hash: str  # sha256 of the source bytes

    # Ordering within the document
    chunk_index: int
    total_chunks: int
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None

    # Location
    page_start: int
    page_end: int

    # Structure
    section_title: str | None = None
    section_path: list[str] = Field(default_factory=list)  # breadcrumb, root -> leaf
    heading_level: int | None = None

    # Content signals
    contains_tables: bool = False
    contains_images: bool = False
    token_count: int = 0
    char_count: int = 0
    language: str | None = None

    # Extensibility & auditing
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    extra: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrieval unit: text plus its metadata. This is what you embed."""

    text: str
    metadata: ChunkMetadata

    def to_embedding_input(self) -> str:
        """Text as it should be handed to the embedding model."""
        return self.text
