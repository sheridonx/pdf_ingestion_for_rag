"""Layout-model PDF parsing via PyMuPDF4LLM.

An alternative to `pdf_parser.DocumentParser` that satisfies the same contract
(`parse(doc) -> list[Block]`), so the chunker and pipeline are untouched. It
exists because the font-heuristic parser has two structural blind spots that
matter for the documents this project targets (bank CSR / climate / Pillar 3
reports):

* **Multi-column pages.** `DocumentParser` sorts blocks by ``(y, x)``, which
  interleaves columns line-by-line on a two-column page. PyMuPDF4LLM resolves
  columns properly — via an ONNX layout model (``pymupdf-layout``) or, failing
  that, PyMuPDF4LLM's own ``column_boxes`` geometry pass.
* **Semantic block classes.** The layout model labels every region
  (``text``, ``section-header``, ``table``, ``list-item``, ``page-header`` …),
  so running headers/footers can be dropped instead of polluting chunks, and
  tables are found without a separate detection pass.

Two engines, selected by `config.layout_engine`:

* ``ml`` — ``pymupdf-layout``'s model. ``to_markdown(page_chunks=True)`` returns
  a ``page_boxes`` list per page, each entry carrying ``class``, ``bbox`` and
  ``pos`` — a ``(start, end)`` character span into that page's Markdown. Slicing
  by ``pos`` gives an exact, non-overlapping block segmentation for free.
* ``heuristic`` — PyMuPDF4LLM without the model. No ``page_boxes``, so the
  page Markdown is re-parsed into blocks by its own syntax (``#`` headings,
  ``|`` table rows, ``-`` list items, blank-line paragraphs).

Heading levels come from the Markdown ``#`` count, then are **rank-normalised**
per document into ``1..max_heading_levels`` — the same idea as
`DocumentParser._build_heading_level_map`, and necessary because the raw counts
are font-size driven and often bunch up at ``######``, which would flatten the
chunker's `_HeadingStack` breadcrumb.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import fitz  # PyMuPDF

from .config import IngestionConfig
from .models import Block, BlockType

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^\s*(#{1,6})\s*(.*)$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_LIST_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+")
# Surrounding Markdown emphasis on a heading line (** __ * _), stripped so the
# breadcrumb / section_title metadata carries plain text, not markup.
_EMPHASIS_RE = re.compile(r"^(\*{1,3}|_{1,3})(.+?)\1$")


def _clean_heading(title: str) -> str:
    """Strip wrapping emphasis and stray leading list markers from a heading."""
    title = _LIST_RE.sub("", title.strip(), count=1).strip()
    match = _EMPHASIS_RE.match(title)
    return match.group(2).strip() if match else title

# page_boxes 'class' -> BlockType. Classes absent here are dropped (see
# _DISCARDED); an unknown/new class falls back to TEXT rather than vanishing.
_CLASS_TO_TYPE: dict[str, BlockType] = {
    "title": BlockType.HEADING,
    "section-header": BlockType.HEADING,
    "text": BlockType.TEXT,
    "list-item": BlockType.TEXT,
    "footnote": BlockType.TEXT,
    "caption": BlockType.TEXT,
    "formula": BlockType.TEXT,
    "code": BlockType.TEXT,
    "table": BlockType.TABLE,
    "picture": BlockType.IMAGE,
    "figure": BlockType.IMAGE,
}

# Running headers/footers: repeated boilerplate that adds nothing to a chunk
# and pollutes the breadcrumb. Dropped when config.drop_running_headers.
_RUNNING = {"page-header", "page-footer"}


def layout_parser_available() -> tuple[bool, str]:
    """Whether PyMuPDF4LLM can be imported, plus a reason when it cannot.

    PyMuPDF4LLM pins an *exact* PyMuPDF version (it raises ImportError on a
    mismatch), so a version skew is a realistic failure here, not just a
    missing package. Both surface as a caller-visible reason string.
    """
    try:
        import pymupdf4llm  # noqa: F401
    except ImportError as exc:
        return False, str(exc)
    return True, ""


def _ml_engine_available() -> bool:
    """Whether the ONNX layout model (`pymupdf-layout`) is installed."""
    try:
        import pymupdf.layout  # noqa: F401
    except ImportError:
        return False
    return True


class LayoutParser:
    """Turns a PDF into an ordered list of `Block`s using PyMuPDF4LLM."""

    def __init__(self, config: IngestionConfig) -> None:
        self.config = config

    # -- public API ---------------------------------------------------------

    def parse(self, doc: fitz.Document) -> list[Block]:
        pages = self._to_markdown(doc)
        use_ml = self._resolve_engine() == "ml"

        blocks: list[Block] = []
        for page in pages:
            page_number = self._page_number(page)
            try:
                if use_ml and page.get("page_boxes"):
                    blocks.extend(self._blocks_from_boxes(page, page_number))
                else:
                    blocks.extend(
                        self._blocks_from_markdown(page.get("text", ""), page_number)
                    )
            except Exception as exc:
                if self.config.skip_pages_on_error:
                    logger.warning("Skipping page %d during block build: %s", page_number, exc)
                else:
                    raise

        self._normalize_heading_levels(blocks)
        return blocks

    # -- PyMuPDF4LLM invocation ---------------------------------------------

    def _resolve_engine(self) -> str:
        """Resolve `layout_engine='auto'` against what is actually installed."""
        requested = self.config.layout_engine
        if requested == "ml":
            if not _ml_engine_available():
                logger.warning(
                    "layout_engine='ml' but 'pymupdf-layout' is not installed; "
                    "falling back to the heuristic engine."
                )
                return "heuristic"
            return "ml"
        if requested == "heuristic":
            return "heuristic"
        return "ml" if _ml_engine_available() else "heuristic"

    def _to_markdown(self, doc: fitz.Document) -> list[dict[str, Any]]:
        """Run PyMuPDF4LLM over the whole document, returning per-page dicts.

        `use_layout` is a module-level global in PyMuPDF4LLM, so it is set on
        every call rather than once at import — a second parser using the other
        engine must not inherit this one's setting.
        """
        import pymupdf4llm

        engine = self._resolve_engine()
        pymupdf4llm.use_layout(engine == "ml")

        kwargs: dict[str, Any] = {"page_chunks": True, "show_progress": False}
        if engine == "heuristic":
            # Only the heuristic engine exposes PyMuPDF's table strategies; the
            # ML engine detects tables with the layout model instead.
            kwargs["table_strategy"] = (
                "lines_strict"
                if self.config.table_detection_strategy == "auto"
                else self.config.table_detection_strategy
            )
        try:
            pages = pymupdf4llm.to_markdown(doc, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"pymupdf4llm.to_markdown failed: {exc}") from exc
        return list(pages or [])

    @staticmethod
    def _page_number(page: dict[str, Any]) -> int:
        meta = page.get("metadata") or {}
        return int(meta.get("page_number", 0)) or 1

    # -- ML engine: structured page_boxes -----------------------------------

    def _blocks_from_boxes(self, page: dict[str, Any], page_number: int) -> list[Block]:
        text = page.get("text", "")
        blocks: list[Block] = []

        for box in page["page_boxes"]:
            cls = str(box.get("class", "")).lower()
            if cls in _RUNNING and self.config.drop_running_headers:
                continue

            start, end = box.get("pos", (0, 0))
            raw = text[start:end]
            btype = _CLASS_TO_TYPE.get(cls, BlockType.TEXT)

            if btype is BlockType.IMAGE:
                if not self.config.extract_images:
                    continue
                body, level = "", None
            else:
                if btype is BlockType.TABLE and not self.config.extract_tables:
                    # Tables disabled: keep the content, drop the special handling.
                    btype = BlockType.TEXT
                body, level = self._strip_heading_markers(raw, btype)
                if not body:
                    continue

            blocks.append(
                Block(
                    type=btype,
                    text=body,
                    page_number=page_number,
                    bbox=self._bbox(box.get("bbox")),
                    heading_level=level,
                )
            )
        return blocks

    def _strip_heading_markers(self, raw: str, btype: BlockType) -> tuple[str, int | None]:
        """Return (text, raw heading level) for a box's Markdown slice.

        For headings the leading ``#``s are consumed and their count returned as
        the *raw* level; `_normalize_heading_levels` ranks it later. A heading
        box with no ``#`` (the model saw a header the font pass did not) keeps
        its text and gets no level, so ranking assigns it the deepest one.
        """
        if btype is not BlockType.HEADING:
            return raw.strip(), None

        # A heading box is normally one line, but guard against stray trailing
        # content so the title used in the breadcrumb stays a single line.
        first, _, rest = raw.strip().partition("\n")
        match = _HEADING_RE.match(first)
        if not match:
            # A heading box the model found but Markdown left unmarked (e.g.
            # rendered as bold): keep the text, let ranking assign the level.
            return _clean_heading(raw.strip()), None
        title = _clean_heading(match.group(2))
        if rest.strip():
            title = f"{title} {rest.strip()}".strip()
        return title, len(match.group(1))

    @staticmethod
    def _bbox(value: Any) -> tuple[float, float, float, float]:
        try:
            x0, y0, x1, y1 = (float(v) for v in value)
        except (TypeError, ValueError):
            return (0.0, 0.0, 0.0, 0.0)
        return (round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2))

    # -- heuristic engine: re-parse the page Markdown -----------------------

    def _blocks_from_markdown(self, text: str, page_number: int) -> list[Block]:
        """Segment a page's Markdown into blocks by its own syntax.

        Used when the ML engine is unavailable, where PyMuPDF4LLM returns text
        only. Bboxes are unavailable at this granularity and are reported as
        zeros — nothing downstream of the parser reads `Block.bbox`.
        """
        blocks: list[Block] = []
        para: list[str] = []
        table: list[str] = []

        def flush_para() -> None:
            body = "\n".join(para).strip()
            para.clear()
            if body:
                blocks.append(self._plain(BlockType.TEXT, body, page_number))

        def flush_table() -> None:
            body = "\n".join(table).strip()
            table.clear()
            if not body:
                return
            btype = BlockType.TABLE if self.config.extract_tables else BlockType.TEXT
            blocks.append(self._plain(btype, body, page_number))

        for line in text.splitlines():
            if _TABLE_ROW_RE.match(line):
                flush_para()
                table.append(line.rstrip())
                continue
            flush_table()

            heading = _HEADING_RE.match(line)
            if heading and heading.group(2).strip():
                flush_para()
                blocks.append(
                    self._plain(
                        BlockType.HEADING,
                        _clean_heading(heading.group(2)),
                        page_number,
                        level=len(heading.group(1)),
                    )
                )
                continue

            if not line.strip():
                flush_para()
                continue

            # A list item ends the running paragraph but is body text itself.
            if _LIST_RE.match(line) and para:
                flush_para()
            para.append(line.rstrip())

        flush_table()
        flush_para()
        return blocks

    @staticmethod
    def _plain(
        btype: BlockType, text: str, page_number: int, level: int | None = None
    ) -> Block:
        return Block(
            type=btype,
            text=text,
            page_number=page_number,
            bbox=(0.0, 0.0, 0.0, 0.0),
            heading_level=level,
        )

    # -- heading level normalisation ----------------------------------------

    def _normalize_heading_levels(self, blocks: list[Block]) -> None:
        """Rank raw ``#`` counts into a dense ``1..max_heading_levels`` scale.

        Raw Markdown levels are font-size driven and sparse — a document may use
        only ``#`` and ``######``. Feeding those to the chunker would make
        `split_heading_level` (default 2) match the title and nothing else, and
        would leave the breadcrumb two levels deep with a gap. Ranking the
        *distinct* observed levels restores a usable hierarchy; levels past the
        cap clamp onto the deepest one. Mutates `blocks` in place.
        """
        cap = self.config.max_heading_levels
        raw = sorted({b.heading_level for b in blocks if b.heading_level is not None})
        ranking = {lvl: min(rank, cap) for rank, lvl in enumerate(raw, start=1)}

        for block in blocks:
            if block.type is not BlockType.HEADING:
                continue
            # A heading with no '#' marker sorts to the deepest level.
            block.heading_level = (
                cap if block.heading_level is None else ranking[block.heading_level]
            )
