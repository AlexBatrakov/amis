"""Public tests for deterministic hybrid RRF retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest

import amis.hybrid as hybrid
from amis.semantic_index import build_semantic_index
from tests.semantic_factory import FakeEmbedder, write_chunk_policy_from_texts


def _built_index(tmp_path: Path) -> tuple[Path, Path]:
    chunks = write_chunk_policy_from_texts(
        tmp_path / "input",
        [
            "Alpha beta appears in the first synthetic notice.",
            "Gamma appears in the second synthetic notice.",
            "Delta appears in the third synthetic notice.",
        ],
    )
    result = build_semantic_index(chunks, tmp_path / "indexes", FakeEmbedder())
    return chunks, result.output_directory


def test_hybrid_search_returns_weighted_rrf_citations(tmp_path: Path) -> None:
    chunks, index = _built_index(tmp_path)

    result = hybrid.search_hybrid_citations(
        "gamma",
        index_directory=index,
        chunk_policy_directory=chunks,
        embedder=FakeEmbedder(),
        top_k=2,
        candidate_k=3,
        excerpt_chars=32,
    )

    assert result.mode_id == hybrid.DEFAULT_HYBRID_MODE_ID
    assert result.query_token_count == 9
    assert result.lexical_query_token_count == 1
    assert result.candidate_k == 3
    assert [citation.rank for citation in result.citations] == [1, 2]
    assert [citation.document_chunk_index for citation in result.citations] == [0, 1]
    assert result.citations[0].source_membership == "both"
    assert result.citations[0].vector_rank == 1
    assert result.citations[0].lexical_rank == 2
    assert result.citations[0].fused_score == pytest.approx(2.0 / 21 + 1.0 / 22)
    assert result.citations[1].vector_rank == 2
    assert result.citations[1].lexical_rank == 1
    assert result.citations[1].fused_score == pytest.approx(2.0 / 22 + 1.0 / 21)
    assert len(result.citations[0].excerpt) <= 32
    assert result.citations[0].source_path.startswith("OPS/Text/")
    assert not Path(result.citations[0].source_path).is_absolute()


def test_fusion_uses_source_membership_and_tie_breaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector_rows = (
        _vector_citation("a", 1, 10, score=0.9),
        _vector_citation("b", 22, 20, score=0.8),
    )
    lexical_rows = (
        _lexical_citation("c", 1, 30, score=3.0),
        _lexical_citation("a", 2, 10, score=2.0),
    )

    monkeypatch.setattr(
        hybrid.retrieval,
        "search_citations",
        lambda *args, **kwargs: hybrid.retrieval.RetrievalResult(7, vector_rows),
    )
    monkeypatch.setattr(
        hybrid.lexical,
        "search_lexical_citations",
        lambda *args, **kwargs: hybrid.lexical.LexicalSearchResult(1, lexical_rows),
    )

    result = hybrid.search_hybrid_citations(
        "alpha",
        index_directory="unused-index",
        chunk_policy_directory="unused-chunks",
        embedder=FakeEmbedder(),
        top_k=3,
        candidate_k=3,
    )

    assert [citation.chunk_id for citation in result.citations] == ["a", "c", "b"]
    assert [citation.source_membership for citation in result.citations] == [
        "both",
        "lexical_only",
        "vector_only",
    ]
    assert result.citations[0].fused_score == pytest.approx(2.0 / 21 + 1.0 / 22)
    assert result.citations[1].fused_score == pytest.approx(
        result.citations[2].fused_score
    )
    assert result.citations[1].lexical_rank == 1
    assert result.citations[2].vector_rank == 22


def test_metadata_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hybrid.retrieval,
        "search_citations",
        lambda *args, **kwargs: hybrid.retrieval.RetrievalResult(
            7, (_vector_citation("a", 1, 0, score=0.9),)
        ),
    )
    monkeypatch.setattr(
        hybrid.lexical,
        "search_lexical_citations",
        lambda *args, **kwargs: hybrid.lexical.LexicalSearchResult(
            1,
            (_lexical_citation("a", 1, 0, score=3.0, text_sha256="1" * 64),),
        ),
    )

    with pytest.raises(hybrid.HybridRetrievalError, match="metadata mismatch"):
        hybrid.search_hybrid_citations(
            "alpha",
            index_directory="unused-index",
            chunk_policy_directory="unused-chunks",
            embedder=FakeEmbedder(),
            top_k=1,
            candidate_k=1,
        )


@pytest.mark.parametrize(
    "top_k,candidate_k", [(0, 1), (2, 1), (True, 1), (1, 0), (1, True)]
)
def test_invalid_options_fail_before_sources(
    monkeypatch: pytest.MonkeyPatch, top_k: object, candidate_k: object
) -> None:
    def source_should_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("retrieval sources must not run")

    monkeypatch.setattr(hybrid.retrieval, "search_citations", source_should_not_run)
    monkeypatch.setattr(
        hybrid.lexical, "search_lexical_citations", source_should_not_run
    )

    with pytest.raises(hybrid.HybridRetrievalError):
        hybrid.search_hybrid_citations(
            "alpha",
            index_directory="unused-index",
            chunk_policy_directory="unused-chunks",
            embedder=FakeEmbedder(),
            top_k=top_k,  # type: ignore[arg-type]
            candidate_k=candidate_k,  # type: ignore[arg-type]
        )


def test_untokenizable_query_fails_before_embedding(tmp_path: Path) -> None:
    chunks, index = _built_index(tmp_path)
    embedder = FakeEmbedder()

    with pytest.raises(hybrid.HybridRetrievalError, match="lexical token"):
        hybrid.search_hybrid_citations(
            "?! --",
            index_directory=index,
            chunk_policy_directory=chunks,
            embedder=embedder,
        )

    assert embedder.query_embed_calls == 0


def test_vector_errors_are_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_vector(*args: object, **kwargs: object) -> object:
        raise hybrid.retrieval.RetrievalError("synthetic vector failure")

    monkeypatch.setattr(hybrid.retrieval, "search_citations", fail_vector)

    with pytest.raises(hybrid.HybridRetrievalError, match="vector retrieval failed"):
        hybrid.search_hybrid_citations(
            "alpha",
            index_directory="unused-index",
            chunk_policy_directory="unused-chunks",
            embedder=FakeEmbedder(),
            top_k=1,
            candidate_k=1,
        )


def test_unsupported_config_is_rejected() -> None:
    config = hybrid.HybridSearchConfig(
        mode_id="rrf_equal_k20",
        vector_weight=1.0,
        lexical_weight=1.0,
        rank_constant=20,
    )

    with pytest.raises(hybrid.HybridRetrievalError, match="unsupported"):
        hybrid.validate_hybrid_search_request(
            "alpha", top_k=1, candidate_k=1, config=config
        )


def _vector_citation(
    chunk_id: str,
    rank: int,
    document_chunk_index: int,
    *,
    score: float,
    text_sha256: str = "0" * 64,
) -> hybrid.retrieval.Citation:
    return hybrid.retrieval.Citation(
        rank=rank,
        score=score,
        document_id="doc_sha256_" + "a" * 64,
        chunk_id=chunk_id,
        document_chunk_index=document_chunk_index,
        section_id="sec_sha256_" + "b" * 64,
        section_chunk_index=document_chunk_index,
        source_path=f"OPS/Text/section-{document_chunk_index}.xhtml",
        start_char=0,
        end_char=10,
        text_sha256=text_sha256,
        excerpt="Synthetic excerpt.",
    )


def _lexical_citation(
    chunk_id: str,
    rank: int,
    document_chunk_index: int,
    *,
    score: float,
    text_sha256: str = "0" * 64,
) -> hybrid.lexical.LexicalCitation:
    return hybrid.lexical.LexicalCitation(
        rank=rank,
        lexical_score=score,
        document_id="doc_sha256_" + "a" * 64,
        chunk_id=chunk_id,
        document_chunk_index=document_chunk_index,
        section_id="sec_sha256_" + "b" * 64,
        section_chunk_index=document_chunk_index,
        source_path=f"OPS/Text/section-{document_chunk_index}.xhtml",
        start_char=0,
        end_char=10,
        text_sha256=text_sha256,
        excerpt="Synthetic excerpt.",
    )
