# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`pdf-ingestion-for-rag` turns PDFs into embedding-ready chunks for RAG. It does
**layout-aware parsing** (PyMuPDF) followed by **hierarchical, structure-aware
chunking** (deterministic, no embedding calls at ingest time, never splits
mid-table). See `README.md` for the rationale vs. semantic chunking.

## Commands

```bash
# Install (editable, with dev + optional extras)
pip install -e ".[dev,lang]"      # add ,azure for the embedding backend

# Tests — chunker/tokenizer tests run without any PDF or PyMuPDF
pytest
pytest tests/test_chunker.py::test_oversized_block_is_windowed   # single test

# Lint (line-length 100, target py310)
ruff check .

# CLI
pdf-ingest-rag report.pdf --out chunks.jsonl
pdf-ingest-rag ./docs --glob "*.pdf" --out ./out --max-tokens 384 --overlap-tokens 48
```

Requires Python ≥ 3.10. Note: the original author's machine had no Python
installed, so the suite may not have been run there — run `pytest` after setup.

## Architecture

The pipeline is a strict one-way flow; each stage has a single owner module.
Read them in this order to understand the whole:

```
open_document ─▶ DocumentParser.parse ─▶ HierarchicalChunker.chunk ─▶ _finalize
 (pdf_parser)      (pdf_parser)              (chunker)                  (pipeline)
   fitz.Document      list[Block]           list[_PendingChunk]        list[Chunk]
```

- **`models.py`** — the two-tier data model that everything hinges on.
  `Block` is the *internal* layout unit (text/heading/table/image, with bbox,
  font size, reading order). `Chunk` + `ChunkMetadata` are the *public*
  retrieval output. The boundary between them is the key design seam: parsing
  produces `Block`s, chunking consumes them, and only `Chunk` is exported.

- **`pdf_parser.py`** — `DocumentParser` does a **two-pass** parse. Pass 1
  scans every line to estimate the document's body font size (most common size
  weighted by character count) and builds a size→heading-level map. Pass 2
  emits `Block`s per page. Heading detection is purely font-driven (size ratio
  vs. body, or bold + short line). Tables render to Markdown and their bboxes
  suppress overlapping raw text lines (avoids duplication). Blocks are sorted
  into reading order by `(y, x)`.

- **`chunker.py`** — `HierarchicalChunker` walks blocks maintaining a
  `_HeadingStack` (the section breadcrumb, root→leaf). It accumulates a buffer
  until the next block would exceed `max_tokens`, flushes with `overlap_tokens`
  of carried tail text, starts a new chunk at each heading (`split_on_section`),
  windows oversized single blocks, keeps tables whole (`keep_tables_whole`),
  and finally merges sub-`min_tokens` chunks backward. Output is
  `_PendingChunk` — no ids/provenance yet.

- **`pipeline.py`** — owns everything needing *document-level* context:
  sha256 hashing, deterministic UUID5 ids (stable across runs via a fixed
  `_NAMESPACE`), prev/next chunk linking, section-context prefixing, optional
  language detection, and provenance metadata. `ingest_pdf` / `ingest_pdf_bytes`
  are the public entry points; both funnel through `_ingest`.

- **`tokenizer.py`** — wraps tiktoken (`cl100k_base` default) with a
  **char-based fallback** if tiktoken/encoding data is unavailable, so the
  pipeline always runs. Exact token-boundary windowing when tiktoken is present;
  sentence-greedy packing otherwise (never cuts mid-word).

- **`config.py`** — `IngestionConfig` (pydantic) is the single source of all
  tunables; a `model_validator` enforces `overlap_tokens < max_tokens` and
  `min_tokens < max_tokens`. Threading a new knob through means adding it here.

- **`embeddings.py`** — optional Azure OpenAI backend (`[azure]` extra), fully
  decoupled from ingestion. Batches by item count **and** an 8000-token/request
  budget, retries transient errors with tenacity backoff, supports dimension
  shortening. `Chunk` is the seam between ingestion and this stage.

## Conventions that matter here

- **Determinism is a feature.** Chunk ids are UUID5 over `file_hash` + index,
  so re-ingesting the same bytes yields identical ids. Don't introduce
  randomness or wall-clock into ids/chunk boundaries (`created_at` is the only
  timestamp, and it's audit-only).
- **Graceful degradation over hard failure.** Missing tiktoken → approximate
  counts; a page that fails to parse → skipped with a warning
  (`skip_pages_on_error`); scanned/image-only PDF → empty result + warning, not
  an exception. Preserve this posture when touching parser/tokenizer.
- **Optional deps are imported lazily** inside functions (`langdetect`,
  `openai`, `tenacity`), never at module top level, so core install stays lean.
  `fitz` (PyMuPDF) is the one required import in `pdf_parser.py`.
- **Extension points** are deliberate: swap the parser for a layout model
  without touching the chunker; add OCR upstream of `ingest_pdf`; layer a
  semantic refinement pass over `Chunk`s. Keep these seams intact.
