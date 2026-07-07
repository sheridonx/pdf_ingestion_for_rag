"""Command-line entry point.

Usage:
    pdf-ingest-rag report.pdf --out chunks.jsonl
    pdf-ingest-rag ./docs --glob "*.pdf" --out out_dir --max-tokens 384
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import IngestionConfig
from .pdf_parser import PDFParseError
from .pipeline import IngestionResult, ingest_pdf


def _build_config(args: argparse.Namespace) -> IngestionConfig:
    return IngestionConfig(
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        min_tokens=args.min_tokens,
        detect_language=args.detect_language,
        extract_tables=not args.no_tables,
        extract_images=not args.no_images,
        pdf_password=args.password,
    )


def _write_jsonl(result: IngestionResult, out: Path) -> None:
    with out.open("w", encoding="utf-8") as fh:
        for chunk in result.chunks:
            record = {"text": chunk.text, **chunk.metadata.model_dump()}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _resolve_inputs(target: Path, pattern: str) -> list[Path]:
    if target.is_dir():
        return sorted(target.glob(pattern))
    return [target]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest PDF(s) into RAG chunks.")
    parser.add_argument("input", type=Path, help="PDF file or directory of PDFs.")
    parser.add_argument("--out", type=Path, help="Output .jsonl file (single input) "
                        "or output directory (directory input).")
    parser.add_argument("--glob", default="*.pdf", help="Glob when input is a directory.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--detect-language", action="store_true")
    parser.add_argument("--no-tables", action="store_true", help="Disable table extraction.")
    parser.add_argument("--no-images", action="store_true", help="Disable image detection.")
    parser.add_argument("--password", default=None, help="Password for encrypted PDFs.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = _build_config(args)
    inputs = _resolve_inputs(args.input, args.glob)
    if not inputs:
        print(f"No PDFs found at {args.input}", file=sys.stderr)
        return 1

    exit_code = 0
    for pdf in inputs:
        try:
            result = ingest_pdf(pdf, config)
        except PDFParseError as exc:
            print(f"[FAIL] {pdf}: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        for w in result.warnings:
            print(f"[WARN] {pdf}: {w}", file=sys.stderr)

        if args.out is None:
            print(f"[OK] {pdf}: {len(result)} chunks (doc_id={result.doc_id})")
            continue

        if args.input.is_dir():
            args.out.mkdir(parents=True, exist_ok=True)
            out_path = args.out / (pdf.stem + ".jsonl")
        else:
            out_path = args.out
        _write_jsonl(result, out_path)
        print(f"[OK] {pdf}: {len(result)} chunks -> {out_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
