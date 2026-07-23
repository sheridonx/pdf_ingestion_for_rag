"""Configuration for the ingestion pipeline.

All tunables live here so behaviour is explicit, reproducible, and reviewable.
Defaults target general-purpose retrieval with ~512-token chunks, which suits
most modern embedding models (e.g. text-embedding-3-*, bge, e5, Voyage).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class IngestionConfig(BaseModel):
    """Immutable-ish settings object driving parsing and chunking.

    Tune `max_tokens` / `overlap_tokens` to your embedding model's context and
    your retrieval granularity. Smaller chunks = sharper retrieval but more
    rows; larger chunks = more context per hit but coarser matching.
    """

    # --- Tokenization -------------------------------------------------------
    tokenizer_encoding: str = Field(
        default="cl100k_base",
        description="tiktoken encoding used for token counting and splitting.",
    )

    # --- Chunk sizing (in tokens) ------------------------------------------
    # Defaults tuned for text-embedding-3-large semantic search: chunks land
    # in the ~200-800 token band, which carries enough context per vector
    # without diluting topical focus.
    max_tokens: int = Field(default=800, ge=64, le=8192)
    min_tokens: int = Field(
        default=200,
        ge=0,
        description="Target floor. A heading does NOT start a new chunk until the "
        "current buffer holds at least this many tokens, and leftover sub-floor "
        "chunks are merged with neighbours. This is the main lever against "
        "over-fragmentation.",
    )
    overlap_tokens: int = Field(
        default=100,
        ge=0,
        description="Token overlap carried between consecutive chunks of a section.",
    )
    hard_max_tokens: int = Field(
        default=8000,
        ge=256,
        le=8191,
        description="Absolute per-chunk ceiling (incl. whole tables), kept below the "
        "text-embedding-3 8191-token request limit. Anything larger is window-split "
        "regardless of keep_tables_whole, so no chunk can be rejected at embed time.",
    )

    # --- Structure handling -------------------------------------------------
    split_on_section: bool = Field(
        default=True,
        description="Allow headings to act as (soft) chunk boundaries at all.",
    )
    split_heading_level: int = Field(
        default=2,
        ge=1,
        le=6,
        description="Only headings at or above this level (1=top) are candidate "
        "chunk boundaries. Deeper subheadings update the breadcrumb but never split, "
        "which prevents fragmentation from many small subsections.",
    )
    merge_across_sections: bool = Field(
        default=True,
        description="Allow sub-min_tokens chunks to merge with a neighbour even across "
        "a section boundary (metadata falls back to the common ancestor section).",
    )
    keep_tables_whole: bool = Field(
        default=True,
        description="Never split a table across chunks, up to hard_max_tokens.",
    )

    # --- Heading detection heuristics --------------------------------------
    heading_size_ratio: float = Field(
        default=1.15,
        ge=1.0,
        description="A line is a heading candidate if its font size exceeds the body "
        "font size by at least this ratio.",
    )
    heading_max_words: int = Field(
        default=18,
        ge=1,
        description="Heading candidates must be short; longer lines are treated as body.",
    )
    max_heading_levels: int = Field(default=4, ge=1, le=6)

    # --- Extraction toggles -------------------------------------------------
    extract_tables: bool = Field(default=True)
    extract_images: bool = Field(default=True)

    # --- Parser backend -----------------------------------------------------
    parser_backend: Literal["pymupdf4llm", "pymupdf"] = Field(
        default="pymupdf4llm",
        description="Layout parser. 'pymupdf4llm' (default) is a layout-model "
        "parser that resolves multi-column pages natively and labels block "
        "classes (headings/tables/headers); 'pymupdf' is the legacy font-heuristic "
        "parser with the pluggable table backend below. The 'pymupdf' parser is "
        "used automatically if pymupdf4llm cannot be imported.",
    )
    layout_engine: Literal["auto", "ml", "heuristic"] = Field(
        default="auto",
        description="Engine for the 'pymupdf4llm' parser. 'ml' uses the "
        "'pymupdf-layout' ONNX model for column/block detection; 'heuristic' uses "
        "pymupdf4llm's geometry-only column detection (no extra model); 'auto' "
        "prefers 'ml' when 'pymupdf-layout' is installed, else 'heuristic'.",
    )
    drop_running_headers: bool = Field(
        default=True,
        description="Drop page-header/page-footer regions (running headers, page "
        "numbers) detected by the layout model. Only applies to the 'pymupdf4llm' "
        "parser with the 'ml' engine.",
    )

    # --- Table detection (legacy 'pymupdf' parser only) ---------------------
    table_backend: Literal["pymupdf", "pdfplumber"] = Field(
        default="pymupdf",
        description="Table detection engine for the legacy 'pymupdf' parser "
        "(ignored by 'pymupdf4llm', which finds tables via its layout pass). "
        "'pymupdf' needs no extra deps; "
        "'pdfplumber' (install the '[tables]' extra) is stronger on borderless / "
        "irregular tables and falls back to 'pymupdf' if the library is missing.",
    )
    table_detection_strategy: Literal["auto", "lines", "lines_strict", "text"] = Field(
        default="auto",
        description="How table gridlines are inferred. 'lines'/'lines_strict' rely on "
        "ruled borders; 'text' infers columns from text alignment (borderless tables); "
        "'auto' tries ruled lines first, then retries with the text strategy per page.",
    )
    table_min_rows: int = Field(
        default=2,
        ge=1,
        description="Detected tables with fewer rows are discarded as false positives.",
    )
    table_min_cols: int = Field(
        default=2,
        ge=1,
        description="Detected tables with fewer columns are discarded as false positives.",
    )
    detect_language: bool = Field(
        default=False,
        description="Per-chunk language detection (requires the 'lang' extra).",
    )

    # --- Robustness ---------------------------------------------------------
    pdf_password: str | None = Field(default=None, description="Password for encrypted PDFs.")
    skip_pages_on_error: bool = Field(
        default=True,
        description="If a single page fails to parse, log and continue instead of aborting.",
    )

    @model_validator(mode="after")
    def _check_invariants(self) -> "IngestionConfig":
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        if self.min_tokens >= self.max_tokens:
            raise ValueError("min_tokens must be smaller than max_tokens")
        if self.hard_max_tokens < self.max_tokens:
            raise ValueError("hard_max_tokens must be >= max_tokens")
        if self.split_heading_level > self.max_heading_levels:
            raise ValueError("split_heading_level must be <= max_heading_levels")
        return self
