"""Unit tests for chunking logic (no PDF/PyMuPDF dependency)."""

from __future__ import annotations

from pdf_ingestion_for_rag.chunker import HierarchicalChunker, _HeadingStack
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
    cfg = IngestionConfig(max_tokens=32, min_tokens=0, keep_tables_whole=True)
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
