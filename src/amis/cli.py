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
from amis.model_cache import (
    ModelCacheError,
    acquire_model,
    default_model_cache,
    snapshot_directory,
    verify_model_snapshot,
)

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
    model_parser = subparsers.add_parser(
        "model", help="acquire or verify the pinned embedding model"
    )
    model_subparsers = model_parser.add_subparsers(dest="model_command", required=True)
    acquire_parser = model_subparsers.add_parser(
        "acquire", help="download and verify the exact pinned model revision"
    )
    _add_cache_argument(acquire_parser)
    verify_parser = model_subparsers.add_parser(
        "verify", help="verify a local model snapshot without network access"
    )
    _add_cache_argument(verify_parser)
    verify_parser.add_argument(
        "--model-snapshot",
        type=Path,
        help="explicit snapshot contained by the selected cache root",
    )
    index_parser = subparsers.add_parser("index", help="build a local semantic index")
    index_subparsers = index_parser.add_subparsers(dest="index_command", required=True)
    build_parser = index_subparsers.add_parser(
        "build", help="build one semantic index with network access disabled"
    )
    build_parser.add_argument(
        "chunk_policy_directory",
        type=Path,
        help="directory containing chunk_manifest.json and chunks.jsonl",
    )
    build_parser.add_argument(
        "--output", type=Path, required=True, help="separate index output root"
    )
    _add_cache_argument(build_parser)
    build_parser.add_argument(
        "--model-snapshot",
        type=Path,
        help="explicit snapshot contained by the selected cache root",
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

    if arguments.command == "model":
        cache_root = arguments.cache_root or default_model_cache()
        try:
            if arguments.model_command == "acquire":
                verified = acquire_model(cache_root)
                action = "Acquired and verified"
            else:
                selected_snapshot = arguments.model_snapshot or snapshot_directory(
                    cache_root
                )
                verified = verify_model_snapshot(
                    selected_snapshot, cache_root=cache_root
                )
                action = "Verified"
        except ModelCacheError as error:
            print(f"amis model: error: {error}", file=sys.stderr)
            return 1
        print(f"{action} pinned model {verified.spec.spec_id}.")
        return 0

    if arguments.command == "index":
        from amis.embeddings import EmbeddingError, SentenceTransformerEmbedder
        from amis.semantic_index import SemanticIndexError, build_semantic_index

        cache_root = arguments.cache_root or default_model_cache()
        selected_snapshot = arguments.model_snapshot or snapshot_directory(cache_root)
        try:
            verified = verify_model_snapshot(selected_snapshot, cache_root=cache_root)
            embedder = SentenceTransformerEmbedder(verified.snapshot_directory)
            result = build_semantic_index(
                arguments.chunk_policy_directory, arguments.output, embedder
            )
        except (ModelCacheError, EmbeddingError, SemanticIndexError) as error:
            print(f"amis index: error: {error}", file=sys.stderr)
            return 1
        print(
            f"Indexed {result.document_id} with {result.chunk_count} vectors "
            f"under {result.index_config_id}."
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


def _add_cache_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="model cache root (overrides AMIS_MODEL_CACHE and the user-local default)",
    )
