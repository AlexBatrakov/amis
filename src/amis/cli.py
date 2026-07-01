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
from amis.retrieval import DEFAULT_EXCERPT_CHARS, DEFAULT_TOP_K

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
    search_parser = subparsers.add_parser(
        "search", help="search one local semantic index for one query"
    )
    search_parser.add_argument("query", help="query text")
    search_parser.add_argument(
        "--index",
        type=Path,
        required=True,
        help="directory containing index_manifest.json, metadata.jsonl, vectors.npy",
    )
    search_parser.add_argument(
        "--chunks",
        type=Path,
        required=True,
        help="matching chunk-policy directory for citation text",
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="number of ranked citations to display",
    )
    search_parser.add_argument(
        "--excerpt-chars",
        type=int,
        default=DEFAULT_EXCERPT_CHARS,
        help="maximum display characters per excerpt",
    )
    _add_cache_argument(search_parser)
    search_parser.add_argument(
        "--model-snapshot",
        type=Path,
        help="explicit snapshot contained by the selected cache root",
    )
    lexical_search_parser = subparsers.add_parser(
        "lexical-search", help="search one chunk-policy directory with BM25"
    )
    lexical_search_parser.add_argument("query", help="query text")
    lexical_search_parser.add_argument(
        "--chunks",
        type=Path,
        required=True,
        help="chunk-policy directory containing chunk_manifest.json and chunks.jsonl",
    )
    lexical_search_parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="number of ranked citations to display",
    )
    lexical_search_parser.add_argument(
        "--excerpt-chars",
        type=int,
        default=DEFAULT_EXCERPT_CHARS,
        help="maximum display characters per excerpt",
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

    if arguments.command == "search":
        from amis.embeddings import EmbeddingError, SentenceTransformerEmbedder
        from amis.retrieval import (
            RetrievalError,
            search_citations,
            validate_search_request,
        )

        cache_root = arguments.cache_root or default_model_cache()
        selected_snapshot = arguments.model_snapshot or snapshot_directory(cache_root)
        try:
            validate_search_request(
                arguments.query,
                top_k=arguments.top_k,
                excerpt_chars=arguments.excerpt_chars,
            )
            verified = verify_model_snapshot(selected_snapshot, cache_root=cache_root)
            embedder = SentenceTransformerEmbedder(verified.snapshot_directory)
            result = search_citations(
                arguments.query,
                index_directory=arguments.index,
                chunk_policy_directory=arguments.chunks,
                embedder=embedder,
                top_k=arguments.top_k,
                excerpt_chars=arguments.excerpt_chars,
            )
        except (ModelCacheError, EmbeddingError, RetrievalError) as error:
            print(f"amis search: error: {error}", file=sys.stderr)
            return 1
        _print_search_result(result)
        return 0

    if arguments.command == "lexical-search":
        from amis.lexical import LexicalRetrievalError, search_lexical_citations

        try:
            result = search_lexical_citations(
                arguments.query,
                chunk_policy_directory=arguments.chunks,
                top_k=arguments.top_k,
                excerpt_chars=arguments.excerpt_chars,
            )
        except LexicalRetrievalError as error:
            print(f"amis lexical-search: error: {error}", file=sys.stderr)
            return 1
        _print_lexical_search_result(result)
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


def _print_search_result(result: object) -> None:
    citations = result.citations
    for index, citation in enumerate(citations):
        if index:
            print()
        print(f"Rank {citation.rank} | score {citation.score:.6f}")
        print(f"source: {citation.source_path}")
        print(f"document_id: {citation.document_id}")
        print(f"chunk_id: {citation.chunk_id}")
        print(f"document_chunk_index: {citation.document_chunk_index}")
        print(f"section_id: {citation.section_id}")
        print(f"section_chunk_index: {citation.section_chunk_index}")
        print(
            f"coordinates: start_char={citation.start_char} "
            f"end_char={citation.end_char}"
        )
        print(f"text_sha256: {citation.text_sha256}")
        print(f"excerpt: {citation.excerpt}")


def _print_lexical_search_result(result: object) -> None:
    citations = result.citations
    for index, citation in enumerate(citations):
        if index:
            print()
        print(f"Rank {citation.rank} | lexical_score {citation.lexical_score:.6f}")
        print(f"source: {citation.source_path}")
        print(f"document_id: {citation.document_id}")
        print(f"chunk_id: {citation.chunk_id}")
        print(f"document_chunk_index: {citation.document_chunk_index}")
        print(f"section_id: {citation.section_id}")
        print(f"section_chunk_index: {citation.section_chunk_index}")
        print(
            f"coordinates: start_char={citation.start_char} "
            f"end_char={citation.end_char}"
        )
        print(f"text_sha256: {citation.text_sha256}")
        print(f"excerpt: {citation.excerpt}")


def _add_cache_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="model cache root (overrides AMIS_MODEL_CACHE and the user-local default)",
    )
