"""Production-grade PDF ingestion for RAG applications.

Pipeline: PDF -> layout-aware parse (text/headings/tables/images)
-> heading-hierarchy sectioning -> token-aware hierarchical chunking
-> richly-typed, linked chunks ready for embedding.

Public API:
    >>> from pdf_ingestion_for_rag import ingest_pdf, IngestionConfig
    >>> chunks = ingest_pdf("report.pdf")
    >>> chunks[0].text, chunks[0].metadata.section_title
"""

from .config import IngestionConfig
from .models import Chunk, ChunkMetadata, BlockType
from .pipeline import ingest_pdf, ingest_pdf_bytes, IngestionResult

__all__ = [
    "IngestionConfig",
    "Chunk",
    "ChunkMetadata",
    "BlockType",
    "ingest_pdf",
    "ingest_pdf_bytes",
    "IngestionResult",
]

__version__ = "0.1.0"
