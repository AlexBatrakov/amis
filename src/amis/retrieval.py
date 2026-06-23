"""Query retrieval and citation display over one validated semantic index."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from amis.embeddings import Embedder, EmbeddingOutput
from amis.model_spec import EMBEDDING_GEMMA
from amis.semantic_index import (
    SemanticIndexError,
    load_semantic_index,
    load_validated_chunk_texts,
)

DEFAULT_TOP_K = 5
DEFAULT_EXCERPT_CHARS = 320
MAX_EXCERPT_CHARS = 1200
QUERY_ITEM_ID = "query-1"

_WHITESPACE_RE = re.compile(r"\s+")


class RetrievalError(Exception):
    """Raised when query retrieval or citation display cannot proceed safely."""


@dataclass(frozen=True)
class Citation:
    """One ranked citation row with a bounded runtime excerpt."""

    rank: int
    score: float
    document_id: str
    chunk_id: str
    document_chunk_index: int
    section_id: str
    section_chunk_index: int
    source_path: str
    start_char: int
    end_char: int
    text_sha256: str
    excerpt: str


@dataclass(frozen=True)
class RetrievalResult:
    """Complete result for one query."""

    query_token_count: int
    citations: tuple[Citation, ...]


def search_citations(
    query: str,
    *,
    index_directory: Path | str,
    chunk_policy_directory: Path | str,
    embedder: Embedder,
    top_k: int = DEFAULT_TOP_K,
    excerpt_chars: int = DEFAULT_EXCERPT_CHARS,
) -> RetrievalResult:
    """Embed one query, search one exact index, and resolve bounded citations."""
    validate_search_request(query, top_k=top_k, excerpt_chars=excerpt_chars)
    try:
        index = load_semantic_index(
            index_directory, expected_chunks=chunk_policy_directory
        )
        chunks = load_validated_chunk_texts(chunk_policy_directory)
        chunk_texts = {chunk.chunk_id: chunk for chunk in chunks}
        if top_k > len(index.metadata):
            raise RetrievalError("top-k must be between 1 and the index size")
        token_counts = tuple(embedder.preflight_queries([QUERY_ITEM_ID], [query]))
        if (
            len(token_counts) != 1
            or type(token_counts[0]) is not int
            or token_counts[0] < 1
        ):
            raise RetrievalError("embedder returned invalid query token counts")
        if token_counts[0] > EMBEDDING_GEMMA.effective_token_limit:
            raise RetrievalError(
                "query token limit exceeded: "
                f"{token_counts[0]} > {EMBEDDING_GEMMA.effective_token_limit}"
            )
        embedded = embedder.embed_queries([QUERY_ITEM_ID], [query])
        query_vector = _validated_query_vector(embedded)
        matches = index.top_k(query_vector, top_k)
    except RetrievalError:
        raise
    except SemanticIndexError as error:
        raise RetrievalError(str(error)) from error
    except Exception as error:
        raise RetrievalError(str(error)) from error

    citations: list[Citation] = []
    for rank, match in enumerate(matches, 1):
        metadata = match.metadata
        chunk = chunk_texts.get(metadata["chunk_id"])
        if chunk is None or chunk.text_sha256 != metadata["text_sha256"]:
            raise RetrievalError("citation text does not match validated metadata")
        citations.append(
            _citation_from_match(
                rank,
                match.score,
                metadata,
                chunk.text,
                excerpt_chars,
            )
        )
    return RetrievalResult(token_counts[0], tuple(citations))


def _validate_query(query: str) -> None:
    if not isinstance(query, str) or not query.strip():
        raise RetrievalError("query must not be empty")


def validate_search_request(query: str, *, top_k: int, excerpt_chars: int) -> None:
    """Validate cheap search inputs before model initialization."""
    _validate_query(query)
    _validate_search_options(top_k, excerpt_chars)


def _validate_search_options(top_k: int, excerpt_chars: int) -> None:
    if type(top_k) is not int or top_k < 1:
        raise RetrievalError("top-k must be a positive integer")
    if (
        type(excerpt_chars) is not int
        or excerpt_chars < 1
        or excerpt_chars > MAX_EXCERPT_CHARS
    ):
        raise RetrievalError(
            f"excerpt-chars must be an integer between 1 and {MAX_EXCERPT_CHARS}"
        )


def _validated_query_vector(embedded: EmbeddingOutput) -> np.ndarray:
    if embedded.item_ids != (QUERY_ITEM_ID,):
        raise RetrievalError("embedder output order does not match the query")
    array = np.asarray(embedded.vectors)
    if array.shape != (1, EMBEDDING_GEMMA.dimension):
        raise RetrievalError("query embedding has the wrong shape")
    if array.dtype != np.dtype(np.float32):
        raise RetrievalError("query embedding must have dtype float32")
    if not np.isfinite(array).all():
        raise RetrievalError("query embedding contains nonfinite values")
    norm = float(np.linalg.norm(array[0]))
    if not math.isfinite(norm) or norm == 0.0:
        raise RetrievalError("query embedding must have a nonzero finite norm")
    return np.asarray(array[0], dtype=np.float32)


def _citation_from_match(
    rank: int,
    score: float,
    metadata: dict[str, Any],
    text: str,
    excerpt_chars: int,
) -> Citation:
    excerpt = _bounded_excerpt(text, excerpt_chars)
    return Citation(
        rank=rank,
        score=score,
        document_id=metadata["document_id"],
        chunk_id=metadata["chunk_id"],
        document_chunk_index=metadata["document_chunk_index"],
        section_id=metadata["section_id"],
        section_chunk_index=metadata["section_chunk_index"],
        source_path=metadata["source_path"],
        start_char=metadata["start_char"],
        end_char=metadata["end_char"],
        text_sha256=metadata["text_sha256"],
        excerpt=excerpt,
    )


def _bounded_excerpt(text: str, excerpt_chars: int) -> str:
    collapsed = _WHITESPACE_RE.sub(" ", text).strip()
    if len(collapsed) <= excerpt_chars:
        return collapsed
    if excerpt_chars <= 3:
        return collapsed[:excerpt_chars]
    return collapsed[: excerpt_chars - 3].rstrip() + "..."
