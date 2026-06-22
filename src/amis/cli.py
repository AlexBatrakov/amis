"""Command-line entry point for AMIS."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from amis.chunking import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
    DEFAULT_TARGET_CHARS,
    ChunkingError,
    ChunkPolicy,
    chunk_document,
)
from amis.ingestion import IngestionError, ingest_epub

READY_MESSAGE = "AMIS repository foundation is ready."


def main(argv: Sequence[str] | None = None) -> int:
    """Run the AMIS command-line interface."""
    parser = argparse.ArgumentParser(
        prog="amis",
        description="Ankh-Morpork Intelligence System",
    )
    subparsers = parser.add_subparsers(dest="command")
    ingest_parser = subparsers.add_parser(
        "ingest", help="normalize one local EPUB 2 source"
    )
    ingest_parser.add_argument("source", type=Path, help="local EPUB file")
    ingest_parser.add_argument(
        "--output", type=Path, required=True, help="normalized output root"
    )
    chunk_parser = subparsers.add_parser(
        "chunk", help="chunk one normalized document directory"
    )
    chunk_parser.add_argument(
        "input_document_directory",
        type=Path,
        help="directory containing document.json and sections.jsonl",
    )
    chunk_parser.add_argument(
        "--output", type=Path, required=True, help="separate chunk output root"
    )
    chunk_parser.add_argument(
        "--target-chars",
        type=int,
        default=DEFAULT_TARGET_CHARS,
        help="preferred chunk size",
    )
    chunk_parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="absolute chunk size limit",
    )
    chunk_parser.add_argument(
        "--overlap-chars",
        type=int,
        default=DEFAULT_OVERLAP_CHARS,
        help="maximum source overlap",
    )
    arguments = parser.parse_args(argv)

    if arguments.command is None:
        print(READY_MESSAGE)
        return 0

    if arguments.command == "ingest":
        try:
            result = ingest_epub(arguments.source, arguments.output)
        except IngestionError as error:
            print(f"amis ingest: error: {error}", file=sys.stderr)
            return 1

        print(
            f"Ingested {result.document_id} with "
            f"{result.section_count} ordered sections."
        )
        return 0

    try:
        policy = ChunkPolicy(
            target_chars=arguments.target_chars,
            max_chars=arguments.max_chars,
            overlap_chars=arguments.overlap_chars,
        )
        chunk_result = chunk_document(
            arguments.input_document_directory,
            arguments.output,
            policy,
        )
    except ChunkingError as error:
        print(f"amis chunk: error: {error}", file=sys.stderr)
        return 1

    print(
        f"Chunked {chunk_result.document_id} with {chunk_result.chunk_count} chunks "
        f"under {chunk_result.policy_id}."
    )
    return 0
