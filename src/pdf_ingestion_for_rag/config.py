"""Configuration for the ingestion pipeline.

All tunables live here so behaviour is explicit, reproducible, and reviewable.
Defaults target general-purpose retrieval with ~512-token chunks, which suits
most modern embedding models (e.g. text-embedding-3-*, bge, e5, Voyage).
"""

from __future__ import annotations

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
    max_tokens: int = Field(default=512, ge=64, le=8192)
    min_tokens: int = Field(
        default=64,
        ge=0,
        description="Chunks smaller than this are merged forward when possible.",
    )
    overlap_tokens: int = Field(
        default=64,
        ge=0,
        description="Token overlap carried between consecutive chunks of a section.",
    )

    # --- Structure handling -------------------------------------------------
    split_on_section: bool = Field(
        default=True,
        description="Start a new chunk when a heading introduces a new section.",
    )
    keep_tables_whole: bool = Field(
        default=True,
        description="Never split a table across chunks, even if it exceeds max_tokens.",
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
        return self
