"""Unit tests for the PyMuPDF4LLM parser's pure logic (no PDF or model needed).

These exercise the segmentation and normalisation the parser does *around*
PyMuPDF4LLM — the parts that turn its Markdown / page_boxes into `Block`s —
without invoking the library itself.
"""

from __future__ import annotations

from pdf_ingestion_for_rag.config import IngestionConfig
from pdf_ingestion_for_rag.layout_parser import LayoutParser, _clean_heading
from pdf_ingestion_for_rag.models import Block, BlockType


def _parser(**overrides) -> LayoutParser:
    return LayoutParser(IngestionConfig(**overrides))


# -- heading cleanup --------------------------------------------------------


def test_clean_heading_strips_wrapping_emphasis():
    assert _clean_heading("**Bold Title**") == "Bold Title"
    assert _clean_heading("__Underlined__") == "Underlined"
    assert _clean_heading("*Italic*") == "Italic"
    # Emphasis only *inside* the line is left alone (not a wrapper).
    assert _clean_heading("A **strong** word") == "A **strong** word"


def test_clean_heading_strips_leading_list_marker():
    assert _clean_heading("- 1 Supporting housing") == "1 Supporting housing"
    assert _clean_heading("**- item**") == "- item"  # emphasis wrapper wins first


# -- ml-engine box handling (_strip_heading_markers) ------------------------


def test_strip_heading_markers_hash_level():
    body, level = _parser()._strip_heading_markers("### Section Title", BlockType.HEADING)
    assert body == "Section Title"
    assert level == 3


def test_strip_heading_markers_unmarked_bold_heading():
    # The model labelled a section-header the Markdown rendered as bold, no '#'.
    body, level = _parser()._strip_heading_markers("**Governance**", BlockType.HEADING)
    assert body == "Governance"
    assert level is None  # ranking assigns the deepest level later


def test_strip_heading_markers_ignored_for_non_heading():
    body, level = _parser()._strip_heading_markers("# not a heading", BlockType.TEXT)
    assert body == "# not a heading"
    assert level is None


# -- heading level normalisation --------------------------------------------


def _heading(level: int | None) -> Block:
    return Block(type=BlockType.HEADING, text="H", page_number=1, bbox=(0, 0, 0, 0),
                 heading_level=level)


def test_normalize_heading_levels_ranks_sparse_levels_densely():
    # Raw markdown used only '#' (1) and '######' (6); rank to a dense 1..N.
    parser = _parser(max_heading_levels=4)
    blocks = [_heading(1), _heading(6), _heading(6), _heading(1)]
    parser._normalize_heading_levels(blocks)
    assert [b.heading_level for b in blocks] == [1, 2, 2, 1]


def test_normalize_heading_levels_clamps_past_cap():
    parser = _parser(max_heading_levels=2)
    blocks = [_heading(1), _heading(2), _heading(3), _heading(4)]
    parser._normalize_heading_levels(blocks)
    # ranks 1,2,3,4 -> clamp at cap 2 -> 1,2,2,2
    assert [b.heading_level for b in blocks] == [1, 2, 2, 2]


def test_normalize_heading_levels_unmarked_heading_gets_deepest():
    parser = _parser(max_heading_levels=4)
    blocks = [_heading(1), _heading(None)]
    parser._normalize_heading_levels(blocks)
    assert blocks[0].heading_level == 1
    assert blocks[1].heading_level == 4  # no '#' -> deepest


def test_normalize_leaves_non_heading_blocks_untouched():
    parser = _parser()
    text = Block(type=BlockType.TEXT, text="body", page_number=1, bbox=(0, 0, 0, 0))
    blocks = [_heading(1), text]
    parser._normalize_heading_levels(blocks)
    assert text.heading_level is None


# -- heuristic markdown segmentation ----------------------------------------


def test_blocks_from_markdown_segments_headings_paragraphs_tables():
    md = (
        "# Title\n"
        "\n"
        "First paragraph line one.\n"
        "line two.\n"
        "\n"
        "## Subsection\n"
        "\n"
        "| A | B |\n"
        "| --- | --- |\n"
        "| 1 | 2 |\n"
        "\n"
        "Closing paragraph.\n"
    )
    blocks = _parser()._blocks_from_markdown(md, page_number=3)
    types = [b.type for b in blocks]
    assert types == [
        BlockType.HEADING,
        BlockType.TEXT,
        BlockType.HEADING,
        BlockType.TABLE,
        BlockType.TEXT,
    ]
    assert blocks[0].text == "Title" and blocks[0].heading_level == 1
    assert blocks[1].text == "First paragraph line one.\nline two."
    assert blocks[3].text.startswith("| A | B |")
    assert all(b.page_number == 3 for b in blocks)


def test_blocks_from_markdown_table_demoted_when_tables_disabled():
    md = "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
    blocks = _parser(extract_tables=False)._blocks_from_markdown(md, page_number=1)
    assert len(blocks) == 1
    assert blocks[0].type is BlockType.TEXT  # kept as text, not a TABLE block


def test_blocks_from_markdown_splits_list_items_from_paragraph():
    md = "Intro sentence.\n- first item\n- second item\n"
    blocks = _parser()._blocks_from_markdown(md, page_number=1)
    # Intro paragraph is separated from the list block(s) that follow it.
    assert blocks[0].text == "Intro sentence."
    assert all(b.type is BlockType.TEXT for b in blocks)
