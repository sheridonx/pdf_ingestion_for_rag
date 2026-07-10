"""Unit tests for chunking logic (no PDF/PyMuPDF dependency)."""

from __future__ import annotations

from pdf_ingestion_for_rag.chunker import (
    HierarchicalChunker,
    _HeadingStack,
    _PendingChunk,
)
from pdf_ingestion_for_rag.config import IngestionConfig
from pdf_ingestion_for_rag.models import Block, BlockType
from pdf_ingestion_for_rag.tokenizer import Tokenizer


def _text_block(text: str, page: int = 1) -> Block:
    return Block(type=BlockType.TEXT, text=text, page_number=page, bbox=(0, 0, 100, 10))


def _heading(text: str, level: int, page: int = 1) -> Block:
    return Block(
        type=BlockType.HEADING,
        text=text,
        page_number=page,
        bbox=(0, 0, 100, 10),
        heading_level=level,
    )


def test_heading_stack_builds_breadcrumb():
    s = _HeadingStack()
    s.push(1, "Chapter 1")
    s.push(2, "Section 1.1")
    assert s.path == ["Chapter 1", "Section 1.1"]
    s.push(2, "Section 1.2")  # same level replaces sibling
    assert s.path == ["Chapter 1", "Section 1.2"]
    s.push(1, "Chapter 2")  # higher level pops back to root
    assert s.path == ["Chapter 2"]


def test_section_boundary_starts_new_chunk():
    cfg = IngestionConfig(max_tokens=512, min_tokens=0, overlap_tokens=0)
    chunker = HierarchicalChunker(cfg, Tokenizer())
    blocks = [
        _heading("Intro", 1),
        _text_block("Alpha content."),
        _heading("Methods", 1),
        _text_block("Beta content."),
    ]
    chunks = chunker.chunk(blocks)
    assert len(chunks) == 2
    assert chunks[0].section_path == ["Intro"]
    assert chunks[1].section_path == ["Methods"]
    assert "Beta" in chunks[1].text


def test_oversized_block_is_windowed():
    cfg = IngestionConfig(max_tokens=64, min_tokens=0, overlap_tokens=8)
    chunker = HierarchicalChunker(cfg, Tokenizer())
    big = " ".join(f"word{i}" for i in range(500))
    chunks = chunker.chunk([_text_block(big)])
    assert len(chunks) > 1
    assert all(c.token_count <= cfg.max_tokens for c in chunks)


def test_table_flag_and_whole_table_kept():
    cfg = IngestionConfig(max_tokens=64, min_tokens=0, overlap_tokens=0, keep_tables_whole=True)
    chunker = HierarchicalChunker(cfg, Tokenizer())
    table = Block(
        type=BlockType.TABLE,
        text="| a | b |\n|---|---|\n" + "\n".join(f"| {i} | {i} |" for i in range(40)),
        page_number=2,
        bbox=(0, 0, 100, 200),
    )
    chunks = chunker.chunk([table])
    assert len(chunks) == 1
    assert chunks[0].contains_tables is True
    assert chunks[0].page_start == 2


def test_soft_boundary_packs_small_sections():
    # Many tiny sections must NOT become one chunk each: headings are soft
    # boundaries and only split once the buffer reaches min_tokens.
    cfg = IngestionConfig(
        max_tokens=512, min_tokens=50, overlap_tokens=0, split_heading_level=2
    )
    chunker = HierarchicalChunker(cfg, Tokenizer())
    blocks = []
    for i in range(8):
        blocks.append(_heading(f"Section {i}", 1))
        blocks.append(_text_block(f"This is a short sentence about topic number {i}."))
    chunks = chunker.chunk(blocks)
    assert len(chunks) < 8  # packed, not one-per-section
    # Every chunk except possibly the last should clear the floor.
    assert all(c.token_count >= cfg.min_tokens for c in chunks[:-1])


def test_deep_headings_do_not_split():
    # Level-3 subheadings (below split_heading_level=2) update the breadcrumb
    # but never start a new chunk.
    cfg = IngestionConfig(
        max_tokens=512, min_tokens=0, overlap_tokens=0, split_heading_level=2
    )
    chunker = HierarchicalChunker(cfg, Tokenizer())
    blocks = [
        _heading("Main", 1),
        _text_block("Intro sentence."),
        _heading("Detail A", 3),
        _text_block("Detail A body."),
        _heading("Detail B", 3),
        _text_block("Detail B body."),
    ]
    chunks = chunker.chunk(blocks)
    assert len(chunks) == 1
    assert chunks[0].section_path == ["Main"]  # snapshotted at chunk start


def test_giant_table_split_under_hard_cap():
    # A table larger than hard_max_tokens is window-split so no chunk can exceed
    # the embedding-request limit, even with keep_tables_whole=True.
    cfg = IngestionConfig(
        max_tokens=256,
        min_tokens=0,
        overlap_tokens=0,
        hard_max_tokens=256,
        keep_tables_whole=True,
    )
    chunker = HierarchicalChunker(cfg, Tokenizer())
    big_table = "| a | b |\n|---|---|\n" + "\n".join(
        f"| value {i} | number {i} |" for i in range(300)
    )
    table = Block(type=BlockType.TABLE, text=big_table, page_number=1, bbox=(0, 0, 100, 500))
    chunks = chunker.chunk([table])
    assert len(chunks) > 1
    assert all(c.token_count <= cfg.max_tokens for c in chunks)
    assert all(c.contains_tables for c in chunks)


def test_many_small_blocks_respect_max_tokens():
    # Regression: a chunk built from many tiny blocks must not exceed max_tokens
    # once the "\n\n" separator tokens are accounted for.
    cfg = IngestionConfig(max_tokens=200, min_tokens=0, overlap_tokens=0)
    chunker = HierarchicalChunker(cfg, Tokenizer())
    # 300 one-to-few-token blocks — like chart labels / stray numbers in PDFs.
    blocks = [_text_block(str(i)) for i in range(300)]
    chunks = chunker.chunk(blocks)
    assert len(chunks) > 1
    assert all(c.token_count <= cfg.max_tokens for c in chunks), [
        c.token_count for c in chunks
    ]


def test_cross_section_merge_uses_common_prefix():
    cfg = IngestionConfig(max_tokens=512, min_tokens=100, merge_across_sections=True)
    chunker = HierarchicalChunker(cfg, Tokenizer())
    a = _PendingChunk("short a", 1, 1, ["Ch1", "A"], 2, False, False, 10)
    b = _PendingChunk("short b", 1, 2, ["Ch1", "B"], 2, False, False, 10)
    merged = chunker._merge_small_chunks([a, b])
    assert len(merged) == 1
    assert merged[0].section_path == ["Ch1"]  # common ancestor
    assert merged[0].page_end == 2
    assert "short a" in merged[0].text and "short b" in merged[0].text


def test_image_sets_flag_without_text():
    cfg = IngestionConfig(min_tokens=0)
    chunker = HierarchicalChunker(cfg, Tokenizer())
    blocks = [
        _text_block("Some prose."),
        Block(type=BlockType.IMAGE, text="", page_number=1, bbox=(0, 0, 50, 50)),
    ]
    chunks = chunker.chunk(blocks)
    assert len(chunks) == 1
    assert chunks[0].contains_images is True
