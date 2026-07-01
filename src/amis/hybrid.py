"""Deterministic hybrid vector-plus-lexical retrieval with RRF fusion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from amis import lexical, retrieval
from amis.embeddings import Embedder

DEFAULT_HYBRID_MODE_ID = "rrf_vector2_lexical1_k20"
DEFAULT_CANDIDATE_K = 20


class HybridRetrievalError(Exception):
    """Raised when hybrid retrieval cannot proceed safely."""


SourceMembership = Literal["both", "vector_only", "lexical_only"]


@dataclass(frozen=True)
class HybridSearchConfig:
    """Supported rank-fusion settings for public hybrid retrieval."""

    mode_id: str
    vector_weight: float
    lexical_weight: float
    rank_constant: int
    default_candidate_k: int = DEFAULT_CANDIDATE_K


DEFAULT_HYBRID_CONFIG = HybridSearchConfig(
    mode_id=DEFAULT_HYBRID_MODE_ID,
    vector_weight=2.0,
    lexical_weight=1.0,
    rank_constant=20,
)


@dataclass(frozen=True)
class HybridCitation:
    """One ranked hybrid citation row with source-specific evidence."""

    rank: int
    fused_score: float
    source_membership: SourceMembership
    vector_rank: int | None
    vector_score: float | None
    lexical_rank: int | None
    lexical_score: float | None
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
class HybridSearchResult:
    """Complete hybrid result for one query."""

    mode_id: str
    query_token_count: int
    lexical_query_token_count: int
    candidate_k: int
    citations: tuple[HybridCitation, ...]


@dataclass
class _HybridCandidate:
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
    vector_rank: int | None = None
    vector_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    fused_score: float = 0.0


def validate_hybrid_search_request(
    query: str,
    *,
    top_k: int,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    excerpt_chars: int = retrieval.DEFAULT_EXCERPT_CHARS,
    config: HybridSearchConfig = DEFAULT_HYBRID_CONFIG,
) -> None:
    """Validate cheap hybrid search inputs before model initialization."""
    _validate_config(config)
    try:
        retrieval.validate_search_request(
            query, top_k=top_k, excerpt_chars=excerpt_chars
        )
    except retrieval.RetrievalError as error:
        raise HybridRetrievalError(str(error)) from error
    if type(candidate_k) is not int or candidate_k < 1:
        raise HybridRetrievalError("candidate-k must be a positive integer")
    if top_k > candidate_k:
        raise HybridRetrievalError("top-k must be no larger than candidate-k")
    if not lexical.tokenize_lexical_text(query):
        raise HybridRetrievalError("query must contain at least one lexical token")


def search_hybrid_citations(
    query: str,
    *,
    index_directory: Path | str,
    chunk_policy_directory: Path | str,
    embedder: Embedder,
    top_k: int = retrieval.DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    excerpt_chars: int = retrieval.DEFAULT_EXCERPT_CHARS,
    config: HybridSearchConfig = DEFAULT_HYBRID_CONFIG,
) -> HybridSearchResult:
    """Search vector and lexical sources, then fuse candidates with RRF."""
    validate_hybrid_search_request(
        query,
        top_k=top_k,
        candidate_k=candidate_k,
        excerpt_chars=excerpt_chars,
        config=config,
    )

    try:
        vector_result = retrieval.search_citations(
            query,
            index_directory=index_directory,
            chunk_policy_directory=chunk_policy_directory,
            embedder=embedder,
            top_k=candidate_k,
            excerpt_chars=excerpt_chars,
        )
    except retrieval.RetrievalError as error:
        raise HybridRetrievalError(f"vector retrieval failed: {error}") from error

    try:
        lexical_result = lexical.search_lexical_citations(
            query,
            chunk_policy_directory=chunk_policy_directory,
            top_k=candidate_k,
            excerpt_chars=excerpt_chars,
        )
    except lexical.LexicalRetrievalError as error:
        raise HybridRetrievalError(f"lexical retrieval failed: {error}") from error

    candidates = _fuse_candidates(vector_result.citations, lexical_result.citations)
    _score_candidates(candidates, config)
    ranked = sorted(candidates.values(), key=_candidate_sort_key)
    citations = tuple(
        _citation_from_candidate(rank, candidate)
        for rank, candidate in enumerate(ranked[:top_k], 1)
    )
    return HybridSearchResult(
        mode_id=config.mode_id,
        query_token_count=vector_result.query_token_count,
        lexical_query_token_count=lexical_result.query_token_count,
        candidate_k=candidate_k,
        citations=citations,
    )


def _validate_config(config: HybridSearchConfig) -> None:
    if config != DEFAULT_HYBRID_CONFIG:
        raise HybridRetrievalError(
            f"unsupported hybrid search mode: {getattr(config, 'mode_id', '<invalid>')}"
        )


def _fuse_candidates(
    vector_citations: tuple[retrieval.Citation, ...],
    lexical_citations: tuple[lexical.LexicalCitation, ...],
) -> dict[str, _HybridCandidate]:
    candidates: dict[str, _HybridCandidate] = {}
    for citation in vector_citations:
        candidate = candidates.setdefault(citation.chunk_id, _candidate_from(citation))
        _validate_same_metadata(candidate, citation)
        candidate.vector_rank = citation.rank
        candidate.vector_score = citation.score
    for citation in lexical_citations:
        candidate = candidates.setdefault(citation.chunk_id, _candidate_from(citation))
        _validate_same_metadata(candidate, citation)
        candidate.lexical_rank = citation.rank
        candidate.lexical_score = citation.lexical_score
    return candidates


def _candidate_from(
    citation: retrieval.Citation | lexical.LexicalCitation,
) -> _HybridCandidate:
    return _HybridCandidate(
        document_id=citation.document_id,
        chunk_id=citation.chunk_id,
        document_chunk_index=citation.document_chunk_index,
        section_id=citation.section_id,
        section_chunk_index=citation.section_chunk_index,
        source_path=citation.source_path,
        start_char=citation.start_char,
        end_char=citation.end_char,
        text_sha256=citation.text_sha256,
        excerpt=citation.excerpt,
    )


def _validate_same_metadata(
    candidate: _HybridCandidate,
    citation: retrieval.Citation | lexical.LexicalCitation,
) -> None:
    for key in (
        "document_id",
        "chunk_id",
        "document_chunk_index",
        "section_id",
        "section_chunk_index",
        "source_path",
        "start_char",
        "end_char",
        "text_sha256",
    ):
        if getattr(candidate, key) != getattr(citation, key):
            raise HybridRetrievalError(
                f"candidate metadata mismatch for {candidate.chunk_id}: {key}"
            )


def _score_candidates(
    candidates: dict[str, _HybridCandidate], config: HybridSearchConfig
) -> None:
    for candidate in candidates.values():
        score = 0.0
        if candidate.vector_rank is not None:
            score += config.vector_weight / (
                config.rank_constant + candidate.vector_rank
            )
        if candidate.lexical_rank is not None:
            score += config.lexical_weight / (
                config.rank_constant + candidate.lexical_rank
            )
        candidate.fused_score = round(score, 12)


def _candidate_sort_key(candidate: _HybridCandidate) -> tuple[float, int, int]:
    source_ranks = [
        rank for rank in (candidate.vector_rank, candidate.lexical_rank) if rank
    ]
    best_rank = min(source_ranks) if source_ranks else DEFAULT_CANDIDATE_K + 1
    return (-candidate.fused_score, best_rank, candidate.document_chunk_index)


def _citation_from_candidate(rank: int, candidate: _HybridCandidate) -> HybridCitation:
    return HybridCitation(
        rank=rank,
        fused_score=candidate.fused_score,
        source_membership=_source_membership(candidate),
        vector_rank=candidate.vector_rank,
        vector_score=candidate.vector_score,
        lexical_rank=candidate.lexical_rank,
        lexical_score=candidate.lexical_score,
        document_id=candidate.document_id,
        chunk_id=candidate.chunk_id,
        document_chunk_index=candidate.document_chunk_index,
        section_id=candidate.section_id,
        section_chunk_index=candidate.section_chunk_index,
        source_path=candidate.source_path,
        start_char=candidate.start_char,
        end_char=candidate.end_char,
        text_sha256=candidate.text_sha256,
        excerpt=candidate.excerpt,
    )


def _source_membership(candidate: _HybridCandidate) -> SourceMembership:
    if candidate.vector_rank is not None and candidate.lexical_rank is not None:
        return "both"
    if candidate.vector_rank is not None:
        return "vector_only"
    return "lexical_only"
