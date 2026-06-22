"""Command-line entry point for AMIS."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

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
    arguments = parser.parse_args(argv)

    if arguments.command is None:
        print(READY_MESSAGE)
        return 0

    try:
        result = ingest_epub(arguments.source, arguments.output)
    except IngestionError as error:
        print(f"amis ingest: error: {error}", file=sys.stderr)
        return 1

    print(
        f"Ingested {result.document_id} with {result.section_count} ordered sections."
    )
    return 0
