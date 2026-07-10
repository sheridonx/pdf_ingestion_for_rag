"""Hierarchical, token-aware chunking of parsed blocks.

Strategy (packing with soft section boundaries):
  1. Walk blocks in reading order, maintaining a heading stack so every chunk
     records its section breadcrumb (root -> leaf).
  2. Accumulate content into a buffer, packing toward `max_tokens`.
  3. Headings are *soft* boundaries: a heading only starts a new chunk when it
     is "major" (`heading_level <= split_heading_level`) AND the buffer already
     holds at least `min_tokens`. This is what prevents the classic failure mode
     of table-heavy PDFs — hundreds of short/bold lines misread as headings —
     from producing a chunk per line.
  4. A block that would overflow `max_tokens` flushes first, carrying
     `overlap_tokens` of trailing text for continuity.
  5. Oversized single blocks are token-windowed. Tables are kept whole up to
     `hard_max_tokens` (the embedding-request ceiling); larger tables are
     window-split so no chunk can be rejected at embed time.
  6. Any remaining sub-`min_tokens` chunks are merged with a neighbour (forward,
     across sections if allowed).

The chunk's section metadata is snapshotted when the chunk *starts*, so packing
across minor/short sections labels the chunk by the section it opens in.

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
    def level(self) -> int | None:
        return self.stack[-1][0] if self.stack else None


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    out: list[str] = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


class HierarchicalChunker:
    """Groups blocks into overlapping, section-scoped chunks."""

    def __init__(self, config: IngestionConfig, tokenizer: Tokenizer) -> None:
        self.config = config
        self.tok = tokenizer

    def chunk(self, blocks: list[Block]) -> list[_PendingChunk]:
        cfg = self.config
        headings = _HeadingStack()
        chunks: list[_PendingChunk] = []

        # Blocks are joined with this separator; its token cost must be counted
        # or chunks with many small blocks silently exceed max_tokens.
        sep = "\n\n"
        sep_tokens = self.tok.count(sep)

        # Mutable state for the chunk currently being assembled.
        buf: list[str] = []
        buf_tokens = 0
        page_start: int | None = None
        page_end: int | None = None
        has_table = False
        has_image = False
        # Section metadata snapshot for the CURRENT chunk (captured at its start).
        sec_path: list[str] = []
        sec_level: int | None = None
        section_captured = False

        def start_new(carry_overlap: str) -> None:
            nonlocal buf, buf_tokens, page_start, page_end
            nonlocal has_table, has_image, section_captured
            buf = [carry_overlap] if carry_overlap else []
            buf_tokens = self.tok.count(carry_overlap) if carry_overlap else 0
            page_start = None
            page_end = None
            has_table = False
            has_image = False
            section_captured = False  # re-snapshot section for the new chunk

        def capture_section() -> None:
            nonlocal sec_path, sec_level, section_captured
            if not section_captured:
                sec_path = headings.path
                sec_level = headings.level
                section_captured = True

        def has_content() -> bool:
            return any(p.strip() for p in buf)

        def add_piece(text_piece: str, tokens: int) -> None:
            """Append a piece, keeping buf_tokens in sync with the joined text
            (including the separator that will sit between blocks)."""
            nonlocal buf_tokens
            if has_content():
                buf_tokens += sep_tokens + tokens
            else:
                buf_tokens = tokens
            buf.append(text_piece)

        def flush(with_overlap: bool) -> None:
            text = "\n\n".join(part for part in buf if part.strip()).strip()
            if not text:
                start_new("")
                return
            chunks.append(
                _PendingChunk(
                    text=text,
                    page_start=page_start or 1,
                    page_end=page_end or page_start or 1,
                    section_path=list(sec_path),
                    heading_level=sec_level,
                    contains_tables=has_table,
                    contains_images=has_image,
                    token_count=self.tok.count(text),
                )
            )
            overlap = self.tok.tail_text(text, cfg.overlap_tokens) if with_overlap else ""
            start_new(overlap)

        def note_page(block: Block) -> None:
            nonlocal page_start, page_end
            if page_start is None:
                page_start = block.page_number
            page_end = block.page_number

        for block in blocks:
            if block.type == BlockType.HEADING:
                level = block.heading_level or cfg.max_heading_levels
                is_major = level <= cfg.split_heading_level
                # Soft boundary: only break at a major heading once the buffer
                # already holds enough content, else keep packing.
                if (
                    cfg.split_on_section
                    and is_major
                    and has_content()
                    and buf_tokens >= cfg.min_tokens
                ):
                    flush(with_overlap=False)
                headings.push(level, block.text)
                capture_section()  # snapshot section for a freshly started chunk
                note_page(block)
                add_piece(block.text, self.tok.count(block.text))
                continue

            if block.type == BlockType.IMAGE:
                has_image = True
                note_page(block)
                continue

            # TEXT or TABLE.
            piece = block.text.strip()
            if not piece:
                continue
            piece_tokens = self.tok.count(piece)

            keep_whole = (
                block.type == BlockType.TABLE
                and cfg.keep_tables_whole
                and piece_tokens <= cfg.hard_max_tokens
            )

            # A block that cannot fit whole -> window-split into its own chunks.
            if piece_tokens > cfg.max_tokens and not keep_whole:
                if has_content():
                    flush(with_overlap=False)
                capture_section()
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

            # Would this block overflow the current chunk? Flush first (with
            # overlap). The separator cost is included so we break at the right
            # point rather than a bit over the ceiling.
            projected = buf_tokens + (sep_tokens if has_content() else 0) + piece_tokens
            if projected > cfg.max_tokens and has_content():
                flush(with_overlap=True)

            capture_section()
            if block.type == BlockType.TABLE:
                has_table = True
            note_page(block)
            add_piece(piece, piece_tokens)

        flush(with_overlap=False)
        return self._merge_small_chunks(chunks)

    # -- small-chunk merging ------------------------------------------------

    def _merge_small_chunks(self, chunks: list[_PendingChunk]) -> list[_PendingChunk]:
        """Merge sub-`min_tokens` chunks forward into following chunks.

        Look-ahead accumulation so a small chunk can absorb the next one(s) up to
        `max_tokens`. Crosses section boundaries only if `merge_across_sections`,
        in which case the merged section metadata falls back to the common
        ancestor breadcrumb.
        """
        cfg = self.config
        if cfg.min_tokens <= 0 or len(chunks) < 2:
            return chunks

        out: list[_PendingChunk] = []
        i, n = 0, len(chunks)
        while i < n:
            cur = chunks[i]
            j = i + 1
            while cur.token_count < cfg.min_tokens and j < n:
                nxt = chunks[j]
                same_section = cur.section_path == nxt.section_path
                if not (same_section or cfg.merge_across_sections):
                    break
                if cur.token_count + nxt.token_count > cfg.max_tokens:
                    break
                cur = self._merge_two(cur, nxt)
                j += 1
            out.append(cur)
            i = j
        return out

    def _merge_two(self, a: _PendingChunk, b: _PendingChunk) -> _PendingChunk:
        same_section = a.section_path == b.section_path
        if same_section:
            path, level = a.section_path, a.heading_level
        else:
            path = _common_prefix(a.section_path, b.section_path)
            # Only keep a level if the merged breadcrumb still matches `a` exactly.
            level = a.heading_level if path == a.section_path else None
        text = f"{a.text}\n\n{b.text}".strip()
        return _PendingChunk(
            text=text,
            page_start=a.page_start,
            page_end=max(a.page_end, b.page_end),
            section_path=path,
            heading_level=level,
            contains_tables=a.contains_tables or b.contains_tables,
            contains_images=a.contains_images or b.contains_images,
            token_count=self.tok.count(text),
        )
