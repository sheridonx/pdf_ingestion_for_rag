# pdf-ingestion-for-rag

Production-grade PDF → chunk ingestion for RAG. Layout-aware parsing
(**PyMuPDF4LLM**, with multi-column support) + **hierarchical, structure-aware
chunking** with rich, typed metadata.

## Why hierarchical (not semantic) chunking?

PDFs already encode structure (headings, sections, tables). Hierarchical
chunking exploits it: it is **deterministic and reproducible**, needs **no
embedding calls at ingest time**, and **won't split mid-table**. Semantic
(embedding-boundary) chunking is non-deterministic, costs an embedding call per
sentence, and gives marginal retrieval gains for most corpora. This library
does structure-aware splitting by default; a semantic refinement pass can be
layered on later if a corpus genuinely needs it.

Pipeline:

```
PDF ─▶ parse (text│heading│table│image, reading order)
    ─▶ heading-hierarchy sectioning
    ─▶ token-aware split within sections (+overlap)
    ─▶ finalize: ids, prev/next links, provenance, metadata
```

## Install

```bash
cd pdf_ingestion_for_rag
python -m venv .venv && .venv\Scripts\activate     # Windows
# source .venv/bin/activate                        # macOS/Linux
pip install -e ".[dev,lang]"
# For the best multi-column detection, add the ONNX layout model:
pip install -e ".[dev,lang,layout]"
```

Requires Python ≥ 3.10. Core deps: `pymupdf`, `pymupdf4llm`, `pydantic`,
`tiktoken`. The `[layout]` extra adds `pymupdf-layout` (the ONNX layout model);
without it the parser still resolves columns with a geometry-only heuristic.

> Note: this machine has no Python interpreter installed, so the test suite has
> not been run here. After installing, run `pytest` to validate.

## Usage — library

```python
from pdf_ingestion_for_rag import ingest_pdf, IngestionConfig

# Defaults (max_tokens=800, min_tokens=200, overlap=100) target
# text-embedding-3-large; chunks land in a ~200-800 token band.
config = IngestionConfig()
result = ingest_pdf("report.pdf", config)

for chunk in result.chunks:
    m = chunk.metadata
    print(m.chunk_index, m.section_path, m.page_start, m.page_end)
    embed(chunk.to_embedding_input())   # your embedding call
```

`chunk.text` is always the **raw** passage. `to_embedding_input()` prepends the
section breadcrumb (e.g. `[A > B > C]`) so an isolated chunk stays
self-describing for the embedder — context lives at embed time, not in the
stored text. Pass `to_embedding_input(prepend_section_context=False)` to embed
the bare text.

In-memory (uploads, object storage):

```python
from pdf_ingestion_for_rag import ingest_pdf_bytes
result = ingest_pdf_bytes(pdf_bytes, source_name="s3://bucket/report.pdf")
```

## Usage — CLI

```bash
pdf-ingest-rag report.pdf --out chunks.jsonl
pdf-ingest-rag ./docs --glob "*.pdf" --out ./out --max-tokens 384 --overlap-tokens 48
```

Each JSONL line is `{"text": ..., <all metadata fields>}`.

## Embedding with Azure OpenAI (optional)

`text-embedding-3-*` uses the `cl100k_base` encoding — already the ingestion
default, so chunk token counts are exact for it. The optional embedder batches
within the 8191-token/request limit, retries on throttling, and supports
dimension shortening.

```bash
pip install -e ".[azure]"
# env: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION
```

```python
from pdf_ingestion_for_rag import ingest_pdf
from pdf_ingestion_for_rag.embeddings import AzureOpenAIEmbedder, AzureEmbedderConfig

result = ingest_pdf("report.pdf")
embedder = AzureOpenAIEmbedder(
    AzureEmbedderConfig(deployment="text-embedding-3-large", dimensions=1024)
)
embedded = embedder.embed_chunks(result.chunks)   # -> [EmbeddedChunk(chunk, embedding), ...]
for e in embedded:
    upsert(id=e.chunk.metadata.chunk_id, vector=e.embedding, payload=e.chunk.metadata.model_dump())
```

`deployment` is your **Azure deployment name**, not the base model name. Omit
`api_key` to use Entra ID / managed-identity auth (`AZURE_OPENAI_AD_TOKEN`).

## Chunk metadata

| Field | Meaning |
|---|---|
| `chunk_id` | Deterministic UUID5 (stable across runs) |
| `doc_id` | UUID5 of the file hash |
| `source_path`, `source_filename`, `file_hash` | Provenance (sha256) |
| `chunk_index`, `total_chunks` | Position within the document |
| `previous_chunk_id`, `next_chunk_id` | Sibling links for windowed retrieval |
| `page_start`, `page_end` | Page span (1-indexed) for citations |
| `section_title`, `section_path`, `heading_level` | Heading hierarchy breadcrumb |
| `contains_tables`, `contains_images` | Content signals for routing/filtering |
| `token_count`, `char_count` | Sizing |
| `language` | Optional per-chunk language (needs `[lang]` extra) |
| `created_at`, `extra` | Audit timestamp + extension bag |

## Configuration

See `IngestionConfig` in `src/pdf_ingestion_for_rag/config.py`. Key knobs:

* **Sizing** — `max_tokens` (800), `min_tokens` (200, the packing floor),
  `overlap_tokens` (100), `hard_max_tokens` (8000, the embedding-safety ceiling).
* **Structure** — `split_on_section`, `split_heading_level` (only headings this
  major are chunk boundaries), `merge_across_sections`, `keep_tables_whole`.
* **Heading detection** — `heading_size_ratio`, `heading_max_words`,
  `max_heading_levels`.
* **Tables** — `table_backend` (`pymupdf` default, or `pdfplumber` via the
  `[tables]` extra), `table_detection_strategy` (`auto` tries ruled lines then a
  borderless-table text fallback), `table_min_rows` / `table_min_cols`.
* **Robustness** — `skip_pages_on_error`, `pdf_password`.

Headings are **soft boundaries**: a chunk only breaks at a major heading once it
already holds `min_tokens`, so table-heavy PDFs don't fragment into one-line
chunks. Raise `min_tokens` for coarser retrieval, lower it for sharper. Section
context is applied at embed time via `Chunk.to_embedding_input()`, not config.

## Scope & extension points

* **Scanned / image-only PDFs** produce no text (a warning is emitted). Add an
  OCR step (e.g. `ocrmypdf`, Tesseract, or a vision model) upstream, then feed
  the OCR'd PDF/bytes to `ingest_pdf`.
* **Parser backend** is pluggable via `config.parser_backend`. The default
  `pymupdf4llm` (`layout_parser.py`) resolves **multi-column** pages natively and
  labels block classes (headings / tables / running headers) with a layout
  model; `config.layout_engine` chooses the ONNX model (`ml`, needs the
  `[layout]` extra) or a geometry-only heuristic. The legacy `pymupdf`
  (`pdf_parser.py`) font-heuristic parser remains available and is used
  automatically if `pymupdf4llm` can't be imported. Both emit the same `Block`
  stream, so the chunker and pipeline are untouched either way.
* **Tables** — the `pymupdf4llm` parser finds tables in its layout pass. The
  legacy `pymupdf` parser detects them in `table_extractor.py` behind a backend
  interface: borderless / whitespace-delimited tables (common in financial & ESG
  disclosures) are handled by the `auto` strategy's text-based fallback, or more
  robustly by the optional `pdfplumber` backend (`pip install -e ".[tables]"`).
  All paths render to the same Markdown, so the chunker is unaffected.
* **Semantic refinement**: run an embedding-similarity merge/split pass over the
  hierarchical chunks if a corpus needs it — the `Chunk` model is the seam.

## Tests

```bash
pytest        # chunker/tokenizer tests run without any PDF
```
