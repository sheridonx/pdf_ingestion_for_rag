"""Layout-aware PDF parsing using PyMuPDF.

Produces an ordered stream of `Block`s (text / heading / table / image) in
approximate reading order. Heading detection is font-driven: we estimate the
document's body font size, then classify larger/bold short lines as headings
and map their sizes to hierarchy levels (H1..Hn).

Notes / limitations:
* Reading order is derived from PyMuPDF's `sort=True`, which handles single- and
  simple multi-column layouts well but is not a full layout-analysis engine.
* Table text is emitted as Markdown; overlapping raw text blocks are dropped to
  avoid duplicating table content in the surrounding prose.
* Scanned/image-only PDFs yield no text; wire an OCR step upstream if needed
  (see README).
"""

from __future__ import annotations

import logging
from collections import Counter

import fitz  # PyMuPDF

from .config import IngestionConfig
from .models import Block, BlockType

logger = logging.getLogger(__name__)

# PyMuPDF span flag bit for bold text.
_FLAG_BOLD = 1 << 4


class PDFParseError(RuntimeError):
    """Raised when a PDF cannot be opened or is unusable."""


def _bbox_overlaps(a: tuple, b: tuple, tol: float = 2.0) -> bool:
    """True if rectangles a and b overlap (with a small tolerance)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 - tol or bx1 < ax0 - tol or ay1 < by0 - tol or by1 < ay0 - tol)


def open_document(source: str | bytes, config: IngestionConfig) -> fitz.Document:
    """Open a PDF from a path or raw bytes, handling encryption."""
    try:
        if isinstance(source, bytes):
            doc = fitz.open(stream=source, filetype="pdf")
        else:
            doc = fitz.open(source)
    except Exception as exc:
        raise PDFParseError(f"Could not open PDF: {exc}") from exc

    if doc.needs_pass:
        if not doc.authenticate(config.pdf_password or ""):
            doc.close()
            raise PDFParseError("PDF is encrypted and the supplied password was rejected.")

    if doc.page_count == 0:
        doc.close()
        raise PDFParseError("PDF contains no pages.")
    return doc


class DocumentParser:
    """Turns a PDF document into an ordered list of `Block`s."""

    def __init__(self, config: IngestionConfig) -> None:
        self.config = config

    def parse(self, doc: fitz.Document) -> list[Block]:
        # First pass: collect raw line records to estimate body font size.
        page_lines: list[list[dict]] = []
        for page in doc:
            try:
                page_lines.append(self._extract_lines(page))
            except Exception as exc:
                if self.config.skip_pages_on_error:
                    logger.warning("Skipping page %d during line scan: %s", page.number + 1, exc)
                    page_lines.append([])
                else:
                    raise PDFParseError(f"Failed to read page {page.number + 1}: {exc}") from exc

        body_size = self._estimate_body_font_size(page_lines)
        heading_levels = self._build_heading_level_map(page_lines, body_size)

        # Second pass: build blocks per page in reading order.
        blocks: list[Block] = []
        for page_index, page in enumerate(doc):
            try:
                blocks.extend(
                    self._blocks_for_page(page, page_lines[page_index], heading_levels)
                )
            except Exception as exc:
                if self.config.skip_pages_on_error:
                    logger.warning("Skipping page %d during block build: %s", page_index + 1, exc)
                else:
                    raise PDFParseError(f"Failed to parse page {page_index + 1}: {exc}") from exc
        return blocks

    # -- line extraction ----------------------------------------------------

    def _extract_lines(self, page: fitz.Page) -> list[dict]:
        """Flatten a page into line records with dominant size/bold/bbox/text."""
        data = page.get_text("dict", sort=True)
        lines: list[dict] = []
        for block in data.get("blocks", []):
            if block.get("type") != 0:  # 0 == text block
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue
                # Dominant span = the one contributing the most characters.
                dominant = max(spans, key=lambda s: len(s.get("text", "")), default=None)
                size = round(float(dominant.get("size", 0.0)), 1) if dominant else 0.0
                is_bold = bool(dominant and (int(dominant.get("flags", 0)) & _FLAG_BOLD))
                lines.append(
                    {
                        "text": text,
                        "size": size,
                        "is_bold": is_bold,
                        "bbox": tuple(round(float(v), 2) for v in line["bbox"]),
                        "page_number": page.number + 1,
                    }
                )
        return lines

    def _estimate_body_font_size(self, page_lines: list[list[dict]]) -> float:
        """Most common font size weighted by character count = body text size."""
        weighted: Counter[float] = Counter()
        for lines in page_lines:
            for ln in lines:
                weighted[ln["size"]] += len(ln["text"])
        if not weighted:
            return 0.0
        return weighted.most_common(1)[0][0]

    def _build_heading_level_map(
        self, page_lines: list[list[dict]], body_size: float
    ) -> dict[float, int]:
        """Map distinct heading font sizes -> hierarchy level (1 = largest)."""
        cfg = self.config
        if body_size <= 0:
            return {}
        threshold = body_size * cfg.heading_size_ratio
        heading_sizes = {
            ln["size"]
            for lines in page_lines
            for ln in lines
            if ln["size"] >= threshold and len(ln["text"].split()) <= cfg.heading_max_words
        }
        ordered = sorted(heading_sizes, reverse=True)[: cfg.max_heading_levels]
        return {size: level for level, size in enumerate(ordered, start=1)}

    def _is_heading(self, line: dict, body_size: float) -> bool:
        cfg = self.config
        words = len(line["text"].split())
        if words > cfg.heading_max_words:
            return False
        if body_size <= 0:
            return False
        if line["size"] >= body_size * cfg.heading_size_ratio:
            return True
        # Bold, same-or-larger size, short line -> likely a sub-heading.
        return line["is_bold"] and line["size"] >= body_size and words <= cfg.heading_max_words

    # -- per-page block assembly -------------------------------------------

    def _blocks_for_page(
        self, page: fitz.Page, lines: list[dict], heading_levels: dict[float, int]
    ) -> list[Block]:
        page_number = page.number + 1

        table_bboxes, table_blocks = self._extract_tables(page)
        image_blocks = self._extract_images(page)

        # Drop text lines that fall inside a detected table (avoid duplication).
        text_blocks: list[Block] = []
        # Local body size for the heading test; the global heading_levels map
        # assigns the actual hierarchy level.
        global_body = self._page_body_size(lines)
        for ln in lines:
            if any(_bbox_overlaps(ln["bbox"], tb) for tb in table_bboxes):
                continue
            is_heading = self._is_heading(ln, global_body)
            level = heading_levels.get(ln["size"]) if is_heading else None
            # A heading whose size isn't in the map (e.g. only bold) -> deepest level.
            if is_heading and level is None:
                level = self.config.max_heading_levels
            text_blocks.append(
                Block(
                    type=BlockType.HEADING if is_heading else BlockType.TEXT,
                    text=ln["text"],
                    page_number=page_number,
                    bbox=ln["bbox"],
                    font_size=ln["size"],
                    is_bold=ln["is_bold"],
                    heading_level=level,
                )
            )

        # Merge everything and sort by reading order (top-to-bottom, left-to-right).
        page_blocks = text_blocks + table_blocks + image_blocks
        page_blocks.sort(key=lambda b: (round(b.bbox[1], 1), round(b.bbox[0], 1)))
        return page_blocks

    def _page_body_size(self, lines: list[dict]) -> float:
        weighted: Counter[float] = Counter()
        for ln in lines:
            weighted[ln["size"]] += len(ln["text"])
        return weighted.most_common(1)[0][0] if weighted else 0.0

    def _extract_tables(self, page: fitz.Page) -> tuple[list[tuple], list[Block]]:
        if not self.config.extract_tables:
            return [], []
        bboxes: list[tuple] = []
        blocks: list[Block] = []
        try:
            finder = page.find_tables()
        except Exception as exc:  # table finder can be finicky on odd PDFs
            logger.debug("find_tables failed on page %d: %s", page.number + 1, exc)
            return [], []
        for table in getattr(finder, "tables", []):
            try:
                markdown = table.to_markdown()
            except Exception:
                markdown = ""
            if not markdown.strip():
                continue
            bbox = tuple(round(float(v), 2) for v in table.bbox)
            bboxes.append(bbox)
            blocks.append(
                Block(
                    type=BlockType.TABLE,
                    text=markdown.strip(),
                    page_number=page.number + 1,
                    bbox=bbox,
                )
            )
        return bboxes, blocks

    def _extract_images(self, page: fitz.Page) -> list[Block]:
        if not self.config.extract_images:
            return []
        blocks: list[Block] = []
        try:
            infos = page.get_image_info()
        except Exception as exc:
            logger.debug("get_image_info failed on page %d: %s", page.number + 1, exc)
            return []
        for info in infos:
            bbox = info.get("bbox")
            if not bbox:
                continue
            # Ignore hairline/decorative images (rules, bullets, backgrounds).
            x0, y0, x1, y1 = bbox
            if (x1 - x0) < 24 or (y1 - y0) < 24:
                continue
            blocks.append(
                Block(
                    type=BlockType.IMAGE,
                    text="",
                    page_number=page.number + 1,
                    bbox=tuple(round(float(v), 2) for v in bbox),
                )
            )
        return blocks
