"""Deterministic BM25 lexical retrieval over one validated chunk directory."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from amis.retrieval import DEFAULT_EXCERPT_CHARS, DEFAULT_TOP_K, MAX_EXCERPT_CHARS
from amis.semantic_index import (
    SemanticIndexError,
    ValidatedChunkText,
    load_validated_chunk_texts,
)

BM25_K1 = 1.2
BM25_B = 0.75
LEXICAL_RANKING_RULE = "bm25_score_desc_document_chunk_index_asc_v1"

_WHITESPACE_RE = re.compile(r"\s+")


class LexicalRetrievalError(Exception):
    """Raised when lexical retrieval cannot proceed safely."""


@dataclass(frozen=True)
class LexicalCitation:
    """One ranked lexical citation row with a bounded runtime excerpt."""

    rank: int
    lexical_score: float
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
class LexicalSearchResult:
    """Complete BM25 result for one query."""

    query_token_count: int
    citations: tuple[LexicalCitation, ...]


@dataclass(frozen=True)
class _LexicalDocument:
    chunk: ValidatedChunkText
    term_counts: Counter[str]
    length: int


def tokenize_lexical_text(text: str) -> tuple[str, ...]:
    """Tokenize text with the public lexical retrieval analyzer."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    current: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if category[:1] in {"L", "M", "N"}:
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


def search_lexical_citations(
    query: str,
    *,
    chunk_policy_directory: Path | str,
    top_k: int = DEFAULT_TOP_K,
    excerpt_chars: int = DEFAULT_EXCERPT_CHARS,
) -> LexicalSearchResult:
    """Search one validated chunk-policy directory with BM25 Okapi."""
    validate_lexical_search_request(query, top_k=top_k, excerpt_chars=excerpt_chars)
    query_tokens = tokenize_lexical_text(query)
    if not query_tokens:
        raise LexicalRetrievalError("query must contain at least one lexical token")

    try:
        chunks = load_validated_chunk_texts(chunk_policy_directory)
    except SemanticIndexError as error:
        raise LexicalRetrievalError(str(error)) from error
    if top_k > len(chunks):
        raise LexicalRetrievalError("top-k must be between 1 and the chunk count")

    documents, idf_by_term, average_length = _build_corpus(chunks)
    query_counts = Counter(query_tokens)
    citations = _score_query(
        query_counts,
        documents,
        idf_by_term,
        average_length,
        top_k,
        excerpt_chars,
    )
    return LexicalSearchResult(len(query_tokens), citations)


def validate_lexical_search_request(
    query: str, *, top_k: int, excerpt_chars: int
) -> None:
    """Validate cheap lexical search inputs before loading chunks."""
    if not isinstance(query, str) or not query.strip():
        raise LexicalRetrievalError("query must not be empty")
    if type(top_k) is not int or top_k < 1:
        raise LexicalRetrievalError("top-k must be a positive integer")
    if (
        type(excerpt_chars) is not int
        or excerpt_chars < 1
        or excerpt_chars > MAX_EXCERPT_CHARS
    ):
        raise LexicalRetrievalError(
            f"excerpt-chars must be an integer between 1 and {MAX_EXCERPT_CHARS}"
        )


def _build_corpus(
    chunks: tuple[ValidatedChunkText, ...],
) -> tuple[tuple[_LexicalDocument, ...], dict[str, float], float]:
    ordered = tuple(sorted(chunks, key=lambda chunk: chunk.document_chunk_index))
    if [chunk.document_chunk_index for chunk in ordered] != list(range(len(ordered))):
        raise LexicalRetrievalError("chunk document order must be contiguous")

    documents: list[_LexicalDocument] = []
    document_frequency: Counter[str] = Counter()
    total_length = 0
    for chunk in ordered:
        tokens = tokenize_lexical_text(chunk.text)
        term_counts = Counter(tokens)
        document_frequency.update(term_counts.keys())
        total_length += len(tokens)
        documents.append(_LexicalDocument(chunk, term_counts, len(tokens)))

    if not documents:
        raise LexicalRetrievalError("lexical corpus is empty")
    average_length = total_length / len(documents)
    if average_length <= 0.0:
        raise LexicalRetrievalError("lexical corpus has no tokens")

    total_documents = len(documents)
    idf_by_term = {
        term: math.log(1.0 + (total_documents - df + 0.5) / (df + 0.5))
        for term, df in document_frequency.items()
    }
    return tuple(documents), idf_by_term, average_length


def _score_query(
    query_counts: Counter[str],
    documents: tuple[_LexicalDocument, ...],
    idf_by_term: dict[str, float],
    average_length: float,
    top_k: int,
    excerpt_chars: int,
) -> tuple[LexicalCitation, ...]:
    scored: list[tuple[float, int, _LexicalDocument]] = []
    for document in documents:
        score = 0.0
        length_factor = 1.0 - BM25_B + BM25_B * (document.length / average_length)
        for term, query_frequency in query_counts.items():
            term_frequency = document.term_counts.get(term, 0)
            if term_frequency == 0:
                continue
            numerator = term_frequency * (BM25_K1 + 1.0)
            denominator = term_frequency + BM25_K1 * length_factor
            score += (
                query_frequency * idf_by_term.get(term, 0.0) * numerator / denominator
            )
        scored.append((score, document.chunk.document_chunk_index, document))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(
        _citation_from_document(rank, score, document, excerpt_chars)
        for rank, (score, _document_index, document) in enumerate(scored[:top_k], 1)
    )


def _citation_from_document(
    rank: int,
    score: float,
    document: _LexicalDocument,
    excerpt_chars: int,
) -> LexicalCitation:
    chunk = document.chunk
    return LexicalCitation(
        rank=rank,
        lexical_score=score,
        document_id=chunk.document_id,
        chunk_id=chunk.chunk_id,
        document_chunk_index=chunk.document_chunk_index,
        section_id=chunk.section_id,
        section_chunk_index=chunk.section_chunk_index,
        source_path=chunk.source_path,
        start_char=chunk.start_char,
        end_char=chunk.end_char,
        text_sha256=chunk.text_sha256,
        excerpt=_bounded_excerpt(chunk.text, excerpt_chars),
    )


def _bounded_excerpt(text: str, excerpt_chars: int) -> str:
    collapsed = _WHITESPACE_RE.sub(" ", text).strip()
    if len(collapsed) <= excerpt_chars:
        return collapsed
    if excerpt_chars <= 3:
        return collapsed[:excerpt_chars]
    return collapsed[: excerpt_chars - 3].rstrip() + "..."
