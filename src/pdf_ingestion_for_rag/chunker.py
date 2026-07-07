"""Hierarchical, token-aware chunking of parsed blocks.

Strategy:
  1. Walk blocks in reading order, maintaining a heading stack so every content
     block knows its section breadcrumb (root -> leaf).
  2. Accumulate content into a chunk until adding the next block would exceed
     `max_tokens`; a new heading also starts a new chunk (configurable).
  3. Carry `overlap_tokens` of trailing text into the next chunk of the same
     section for retrieval continuity.
  4. Oversized single blocks (huge paragraphs) are token-windowed; tables are
     kept whole when `keep_tables_whole` is set.
  5. Tiny trailing chunks are merged backward to respect `min_tokens`.

Output is a list of `_PendingChunk` records; the pipeline finalizes them into
`Chunk` objects (assigning ids, links, and provenance metadata).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import IngestionConfig
from .models import Block, BlockType
from .tokenizer import Tokenizer


@dataclass
class _PendingChunk:
    """A chunk before ids/provenance are attached."""

    text: str
    page_start: int
    page_end: int
    section_path: list[str]
    heading_level: int | None
    contains_tables: bool
    contains_images: bool
    token_count: int


@dataclass
class _HeadingStack:
    """Tracks the active section breadcrumb as headings are encountered."""

    stack: list[tuple[int, str]] = field(default_factory=list)

    def push(self, level: int, title: str) -> None:
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, title))

    @property
    def path(self) -> list[str]:
        return [title for _, title in self.stack]

    @property
    def title(self) -> str | None:
        return self.stack[-1][1] if self.stack else None

    @property
    def level(self) -> int | None:
        return self.stack[-1][0] if self.stack else None


class HierarchicalChunker:
    """Groups blocks into overlapping, section-scoped chunks."""

    def __init__(self, config: IngestionConfig, tokenizer: Tokenizer) -> None:
        self.config = config
        self.tok = tokenizer

    def chunk(self, blocks: list[Block]) -> list[_PendingChunk]:
        cfg = self.config
        headings = _HeadingStack()
        chunks: list[_PendingChunk] = []

        # Mutable state for the chunk currently being assembled.
        buf: list[str] = []
        buf_tokens = 0
        page_start: int | None = None
        page_end: int | None = None
        has_table = False
        has_image = False
        section_path: list[str] = []
        heading_level: int | None = None

        def reset_after_flush(carry_overlap: str) -> None:
            nonlocal buf, buf_tokens, page_start, page_end, has_table, has_image
            buf = [carry_overlap] if carry_overlap else []
            buf_tokens = self.tok.count(carry_overlap) if carry_overlap else 0
            page_start = None
            page_end = None
            has_table = False
            has_image = False

        def flush(with_overlap: bool = True) -> None:
            nonlocal section_path, heading_level
            text = "\n\n".join(part for part in buf if part.strip()).strip()
            if not text:
                reset_after_flush("")
                return
            chunks.append(
                _PendingChunk(
                    text=text,
                    page_start=page_start or 1,
                    page_end=page_end or page_start or 1,
                    section_path=list(section_path),
                    heading_level=heading_level,
                    contains_tables=has_table,
                    contains_images=has_image,
                    token_count=self.tok.count(text),
                )
            )
            overlap = (
                self.tok.tail_text(text, cfg.overlap_tokens) if with_overlap else ""
            )
            reset_after_flush(overlap)

        def note_page(block: Block) -> None:
            nonlocal page_start, page_end
            if page_start is None:
                page_start = block.page_number
            page_end = block.page_number

        for block in blocks:
            if block.type == BlockType.HEADING:
                # Close the current chunk at a section boundary.
                if cfg.split_on_section and buf and any(p.strip() for p in buf):
                    flush(with_overlap=False)
                headings.push(block.heading_level or cfg.max_heading_levels, block.text)
                section_path = headings.path
                heading_level = headings.level
                # The heading line itself seeds the new section's first chunk.
                note_page(block)
                buf.append(f"{block.text}")
                buf_tokens += self.tok.count(block.text)
                continue

            if block.type == BlockType.IMAGE:
                has_image = True
                note_page(block)
                # Images carry no text but mark presence; nothing to accumulate.
                continue

            # TEXT or TABLE: accumulate, splitting when needed.
            piece = block.text.strip()
            if not piece:
                continue
            piece_tokens = self.tok.count(piece)

            keep_whole = block.type == BlockType.TABLE and cfg.keep_tables_whole
            oversize = piece_tokens > cfg.max_tokens

            if oversize and not keep_whole:
                # Flush current buffer, then emit token windows for the big block.
                if buf and any(p.strip() for p in buf):
                    flush(with_overlap=False)
                for window in self.tok.split_to_windows(
                    piece, cfg.max_tokens, cfg.overlap_tokens
                ):
                    note_page(block)
                    if block.type == BlockType.TABLE:
                        has_table = True
                    buf.append(window)
                    buf_tokens = self.tok.count("\n\n".join(buf))
                    flush(with_overlap=False)
                continue

            # Would this block overflow the current chunk? Flush first.
            if buf_tokens + piece_tokens > cfg.max_tokens and any(
                p.strip() for p in buf
            ):
                flush(with_overlap=True)

            if block.type == BlockType.TABLE:
                has_table = True
            note_page(block)
            buf.append(piece)
            buf_tokens += piece_tokens

        flush(with_overlap=False)
        return self._merge_small_chunks(chunks)

    def _merge_small_chunks(self, chunks: list[_PendingChunk]) -> list[_PendingChunk]:
        """Merge sub-`min_tokens` chunks into a neighbor within the same section."""
        cfg = self.config
        if cfg.min_tokens <= 0 or len(chunks) < 2:
            return chunks

        merged: list[_PendingChunk] = []
        for chunk in chunks:
            if (
                merged
                and chunk.token_count < cfg.min_tokens
                and merged[-1].section_path == chunk.section_path
                and merged[-1].token_count + chunk.token_count <= cfg.max_tokens
            ):
                prev = merged[-1]
                prev.text = f"{prev.text}\n\n{chunk.text}".strip()
                prev.page_end = max(prev.page_end, chunk.page_end)
                prev.contains_tables = prev.contains_tables or chunk.contains_tables
                prev.contains_images = prev.contains_images or chunk.contains_images
                prev.token_count = self.tok.count(prev.text)
            else:
                merged.append(chunk)
        return merged
