"""Unit tests for table rendering / filtering (no PDF or PyMuPDF needed)."""

from __future__ import annotations

from pdf_ingestion_for_rag.config import IngestionConfig
from pdf_ingestion_for_rag.table_extractor import _is_meaningful, cells_to_markdown


def test_cells_to_markdown_basic_shape():
    md = cells_to_markdown([["Year", "Revenue"], ["2024", "100"], ["2025", "120"]])
    lines = md.splitlines()
    assert lines[0] == "| Year | Revenue |"
    assert lines[1] == "| --- | --- |"  # header separator
    assert lines[2] == "| 2024 | 100 |"
    assert len(lines) == 4  # header + separator + 2 body rows


def test_cells_to_markdown_escapes_pipes_and_collapses_newlines():
    md = cells_to_markdown([["a|b", "c"], ["multi\nline", None]])
    assert "a\\|b" in md  # literal pipe escaped so it can't break the table
    assert "multi line" in md  # internal newline collapsed to a space
    # None and short rows are padded to the column count, never dropped.
    assert md.splitlines()[-1] == "| multi line |  |"


def test_cells_to_markdown_pads_ragged_rows():
    md = cells_to_markdown([["a", "b", "c"], ["x"]])
    assert md.splitlines()[-1] == "| x |  |  |"


def test_cells_to_markdown_empty():
    assert cells_to_markdown([]) == ""
    assert cells_to_markdown([[]]) == ""


def test_is_meaningful_rejects_degenerate_tables():
    cfg = IngestionConfig(table_min_rows=2, table_min_cols=2)
    assert _is_meaningful([["a", "b"], ["c", "d"]], cfg) is True
    assert _is_meaningful([["only one row and col"]], cfg) is False  # too small
    assert _is_meaningful([["a", "b"]], cfg) is False  # one row < min_rows
    assert _is_meaningful([[None, ""], ["", None]], cfg) is False  # all empty


def test_is_meaningful_respects_config_thresholds():
    cfg = IngestionConfig(table_min_rows=1, table_min_cols=1)
    assert _is_meaningful([["single"]], cfg) is True
