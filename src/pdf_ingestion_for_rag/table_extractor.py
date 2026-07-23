"""Table detection and rendering, decoupled from the rest of the parser.

The parser owns *layout*; this module owns *tables* specifically, behind a
single interface so the detection engine can be swapped without touching
`pdf_parser` or the chunker. Two backends are provided:

* ``pymupdf`` (default, no extra deps) — PyMuPDF's ``page.find_tables``. The
  built-in ``lines_strict`` strategy only sees tables drawn with ruled lines,
  so the ``auto`` strategy here retries with the ``text`` strategy to catch
  **borderless / whitespace-delimited** tables (common in financial & ESG
  disclosures) before giving up on a page.
* ``pdfplumber`` (optional, ``[tables]`` extra, imported lazily) — purpose-built
  table extraction with tunable line/text strategies; generally stronger on
  borderless and irregular tables. Selected via ``config.table_backend``.

Both backends funnel through :func:`cells_to_markdown`, so a table renders to
the same GitHub-flavoured Markdown regardless of engine, and the downstream
`Block`/chunker contract (tables are just Markdown text) is unchanged.

Coordinates are page points with a top-left origin (y increases downward),
matching PyMuPDF text-line bboxes so the parser can suppress raw text that
overlaps a detected table.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import fitz  # PyMuPDF

from .config import IngestionConfig

logger = logging.getLogger(__name__)

# PyMuPDF find_tables strategy names, mapped from our config vocabulary.
_PYMUPDF_STRATEGIES = {
    "lines": "lines",
    "lines_strict": "lines_strict",
    "text": "text",
}


@dataclass(frozen=True)
class ExtractedTable:
    """A detected table: its Markdown rendering plus geometry and shape."""

    markdown: str
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1), top-left origin
    n_rows: int
    n_cols: int


def cells_to_markdown(rows: list[list[object]]) -> str:
    """Render a grid of cells to a GitHub-flavoured Markdown table.

    Row 0 is treated as the header. Cell values are stringified; ``None`` and
    missing trailing cells become empty strings. Pipes are escaped and internal
    newlines collapsed so a cell can never break the table structure.
    """
    if not rows:
        return ""
    n_cols = max((len(r) for r in rows), default=0)
    if n_cols == 0:
        return ""

    def _cell(value: object) -> str:
        if value is None:
            return ""
        text = str(value).replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
        return text.replace("|", "\\|").strip()

    def _row(cells: list[object]) -> str:
        padded = [cells[i] if i < len(cells) else "" for i in range(n_cols)]
        return "| " + " | ".join(_cell(c) for c in padded) + " |"

    lines = [_row(rows[0]), "| " + " | ".join(["---"] * n_cols) + " |"]
    lines.extend(_row(r) for r in rows[1:])
    return "\n".join(lines)


def _shape(rows: list[list[object]]) -> tuple[int, int]:
    return len(rows), max((len(r) for r in rows), default=0)


def _is_meaningful(rows: list[list[object]], config: IngestionConfig) -> bool:
    """Reject degenerate 'tables' (single cell, too few rows/cols, all empty)."""
    n_rows, n_cols = _shape(rows)
    if n_rows < config.table_min_rows or n_cols < config.table_min_cols:
        return False
    non_empty = sum(
        1 for r in rows for c in r if c is not None and str(c).strip()
    )
    # Need at least one populated cell per row on average, else it's noise.
    return non_empty >= n_rows


class TableExtractor:
    """Per-document table detector; dispatches to the configured backend.

    Lifecycle mirrors the document: :meth:`open` once, :meth:`extract` per page,
    :meth:`close` at the end. It is a context manager for convenience.
    """

    def __init__(self, config: IngestionConfig) -> None:
        self.config = config
        self._plumber = None  # lazily-opened pdfplumber.PDF, if that backend

    # -- lifecycle ----------------------------------------------------------

    def open(self, doc: fitz.Document) -> None:
        if not self.config.extract_tables:
            return
        if self.config.table_backend == "pdfplumber":
            self._open_pdfplumber(doc)

    def close(self) -> None:
        if self._plumber is not None:
            try:
                self._plumber.close()
            finally:
                self._plumber = None

    def __enter__(self) -> "TableExtractor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- extraction ---------------------------------------------------------

    def extract(self, page: fitz.Page) -> list[ExtractedTable]:
        if not self.config.extract_tables:
            return []
        try:
            if self._plumber is not None:
                return self._extract_pdfplumber(page)
            return self._extract_pymupdf(page)
        except Exception as exc:  # detection is best-effort, never fatal
            logger.debug(
                "Table extraction failed on page %d: %s", page.number + 1, exc
            )
            return []

    # -- pymupdf backend ----------------------------------------------------

    def _extract_pymupdf(self, page: fitz.Page) -> list[ExtractedTable]:
        strategy = self.config.table_detection_strategy
        if strategy == "auto":
            # Ruled tables first; fall back to the text strategy for borderless
            # tables only if the strict pass found nothing on this page.
            tables = self._find_pymupdf(page, "lines_strict")
            if not tables:
                tables = self._find_pymupdf(page, "text")
            return tables
        return self._find_pymupdf(page, _PYMUPDF_STRATEGIES.get(strategy, "lines_strict"))

    def _find_pymupdf(self, page: fitz.Page, strategy: str) -> list[ExtractedTable]:
        try:
            finder = page.find_tables(
                vertical_strategy=strategy, horizontal_strategy=strategy
            )
        except Exception as exc:  # table finder can be finicky on odd PDFs
            logger.debug(
                "find_tables(%s) failed on page %d: %s",
                strategy,
                page.number + 1,
                exc,
            )
            return []
        out: list[ExtractedTable] = []
        for table in getattr(finder, "tables", []):
            try:
                rows = table.extract()
            except Exception:
                continue
            if not _is_meaningful(rows, self.config):
                continue
            markdown = cells_to_markdown(rows)
            if not markdown.strip():
                continue
            n_rows, n_cols = _shape(rows)
            out.append(
                ExtractedTable(
                    markdown=markdown,
                    bbox=tuple(round(float(v), 2) for v in table.bbox),
                    n_rows=n_rows,
                    n_cols=n_cols,
                )
            )
        return out

    # -- pdfplumber backend -------------------------------------------------

    def _open_pdfplumber(self, doc: fitz.Document) -> None:
        try:
            import pdfplumber
        except ImportError:
            logger.warning(
                "table_backend='pdfplumber' but pdfplumber is not installed; "
                "falling back to the pymupdf backend. Install the '[tables]' extra."
            )
            return
        try:
            # Re-serialise the (possibly decrypted) document so pdfplumber and
            # PyMuPDF read identical bytes and page indices line up.
            self._plumber = pdfplumber.open(io.BytesIO(doc.tobytes()))
        except Exception as exc:
            logger.warning(
                "Could not open document with pdfplumber (%s); using pymupdf backend.",
                exc,
            )
            self._plumber = None

    def _plumber_settings(self) -> dict:
        strategy = self.config.table_detection_strategy
        if strategy in ("lines", "lines_strict"):
            return {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
        if strategy == "text":
            return {"vertical_strategy": "text", "horizontal_strategy": "text"}
        return {}  # 'auto' -> pdfplumber defaults (lines) then text fallback

    def _extract_pdfplumber(self, page: fitz.Page) -> list[ExtractedTable]:
        index = page.number
        if index >= len(self._plumber.pages):
            return []
        plumber_page = self._plumber.pages[index]
        settings = self._plumber_settings()

        found = self._find_plumber(plumber_page, settings)
        if not found and self.config.table_detection_strategy == "auto":
            # Borderless fallback: retry with the text strategy.
            found = self._find_plumber(
                plumber_page,
                {"vertical_strategy": "text", "horizontal_strategy": "text"},
            )
        return found

    def _find_plumber(self, plumber_page, settings: dict) -> list[ExtractedTable]:
        try:
            tables = plumber_page.find_tables(table_settings=settings or None)
        except Exception as exc:
            logger.debug(
                "pdfplumber.find_tables failed on page %d: %s",
                plumber_page.page_number,
                exc,
            )
            return []
        out: list[ExtractedTable] = []
        for table in tables:
            try:
                rows = table.extract()
            except Exception:
                continue
            if not _is_meaningful(rows, self.config):
                continue
            markdown = cells_to_markdown(rows)
            if not markdown.strip():
                continue
            n_rows, n_cols = _shape(rows)
            out.append(
                ExtractedTable(
                    markdown=markdown,
                    bbox=tuple(round(float(v), 2) for v in table.bbox),
                    n_rows=n_rows,
                    n_cols=n_cols,
                )
            )
        return out
