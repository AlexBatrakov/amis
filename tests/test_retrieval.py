"""Public tests for one-query retrieval and citation display."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import amis.retrieval as retrieval
from amis.semantic_index import build_semantic_index
from tests.semantic_factory import FakeEmbedder, write_chunk_policy


def _built_index(tmp_path: Path) -> tuple[Path, Path]:
    chunks = write_chunk_policy(tmp_path / "input")
    result = build_semantic_index(chunks, tmp_path / "indexes", FakeEmbedder())
    return chunks, result.output_directory


def test_search_citations_returns_ranked_bounded_rows(tmp_path: Path) -> None:
    chunks, index = _built_index(tmp_path)

    result = retrieval.search_citations(
        "synthetic query",
        index_directory=index,
        chunk_policy_directory=chunks,
        embedder=FakeEmbedder(),
        top_k=2,
        excerpt_chars=24,
    )

    assert result.query_token_count == 10
    assert [citation.rank for citation in result.citations] == [1, 2]
    assert [citation.document_chunk_index for citation in result.citations] == [0, 1]
    assert result.citations[0].score == pytest.approx(1.0)
    assert result.citations[1].score == pytest.approx(0.0)
    assert len(result.citations[0].excerpt) <= 24
    assert result.citations[0].excerpt.endswith("...")
    assert result.citations[0].source_path.startswith("OPS/Text/")
    assert not Path(result.citations[0].source_path).is_absolute()


def test_empty_query_fails_before_embedding(tmp_path: Path) -> None:
    chunks, index = _built_index(tmp_path)
    embedder = FakeEmbedder()

    with pytest.raises(retrieval.RetrievalError, match="empty"):
        retrieval.search_citations(
            "   ",
            index_directory=index,
            chunk_policy_directory=chunks,
            embedder=embedder,
        )

    assert embedder.query_embed_calls == 0


@pytest.mark.parametrize("top_k", [0, True])
def test_invalid_top_k_fails_before_embedding(tmp_path: Path, top_k: object) -> None:
    chunks, index = _built_index(tmp_path)
    embedder = FakeEmbedder()

    with pytest.raises(retrieval.RetrievalError, match="top-k"):
        retrieval.search_citations(
            "synthetic query",
            index_directory=index,
            chunk_policy_directory=chunks,
            embedder=embedder,
            top_k=top_k,  # type: ignore[arg-type]
        )

    assert embedder.query_embed_calls == 0


@pytest.mark.parametrize("excerpt_chars", [0, 1201, True])
def test_invalid_excerpt_size_fails_before_embedding(
    tmp_path: Path, excerpt_chars: object
) -> None:
    chunks, index = _built_index(tmp_path)
    embedder = FakeEmbedder()

    with pytest.raises(retrieval.RetrievalError, match="excerpt-chars"):
        retrieval.search_citations(
            "synthetic query",
            index_directory=index,
            chunk_policy_directory=chunks,
            embedder=embedder,
            excerpt_chars=excerpt_chars,  # type: ignore[arg-type]
        )

    assert embedder.query_embed_calls == 0


def test_top_k_larger_than_index_size_is_rejected(tmp_path: Path) -> None:
    chunks, index = _built_index(tmp_path)

    with pytest.raises(retrieval.RetrievalError, match="top-k"):
        retrieval.search_citations(
            "synthetic query",
            index_directory=index,
            chunk_policy_directory=chunks,
            embedder=FakeEmbedder(),
            top_k=4,
        )


@pytest.mark.parametrize(
    "embedder",
    [
        FakeEmbedder(query_token_counts=()),
        FakeEmbedder(query_token_counts=(10, 11)),
        FakeEmbedder(query_token_counts=(1985,)),
        FakeEmbedder(query_preflight_error="backend token counts do not match"),
    ],
)
def test_query_preflight_failures_stop_before_embedding(
    tmp_path: Path, embedder: FakeEmbedder
) -> None:
    chunks, index = _built_index(tmp_path)

    with pytest.raises(retrieval.RetrievalError):
        retrieval.search_citations(
            "synthetic query",
            index_directory=index,
            chunk_policy_directory=chunks,
            embedder=embedder,
        )

    assert embedder.query_embed_calls == 0


def test_query_embedding_failure_is_mapped(tmp_path: Path) -> None:
    chunks, index = _built_index(tmp_path)

    with pytest.raises(retrieval.RetrievalError, match="synthetic embedding failed"):
        retrieval.search_citations(
            "synthetic query",
            index_directory=index,
            chunk_policy_directory=chunks,
            embedder=FakeEmbedder(query_embed_error="synthetic embedding failed"),
            top_k=1,
        )


@pytest.mark.parametrize(
    "vectors",
    [
        np.ones((2, 768), dtype=np.float32),
        np.ones((1, 767), dtype=np.float32),
        np.ones((1, 768), dtype=np.float64),
        np.full((1, 768), np.nan, dtype=np.float32),
        np.zeros((1, 768), dtype=np.float32),
    ],
)
def test_malformed_query_vectors_are_rejected(
    tmp_path: Path, vectors: np.ndarray
) -> None:
    chunks, index = _built_index(tmp_path)

    with pytest.raises(retrieval.RetrievalError, match="query embedding"):
        retrieval.search_citations(
            "synthetic query",
            index_directory=index,
            chunk_policy_directory=chunks,
            embedder=FakeEmbedder(query_vectors=vectors),
            top_k=1,
        )


def test_reordered_query_output_is_rejected(tmp_path: Path) -> None:
    chunks, index = _built_index(tmp_path)

    with pytest.raises(retrieval.RetrievalError, match="order"):
        retrieval.search_citations(
            "synthetic query",
            index_directory=index,
            chunk_policy_directory=chunks,
            embedder=FakeEmbedder(query_output_ids=("other-query",)),
            top_k=1,
        )


def test_stale_chunks_cannot_be_cited(tmp_path: Path) -> None:
    chunks, index = _built_index(tmp_path)
    manifest_path = chunks / "chunk_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["input_document_sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )

    with pytest.raises(retrieval.RetrievalError, match="stale"):
        retrieval.search_citations(
            "synthetic query",
            index_directory=index,
            chunk_policy_directory=chunks,
            embedder=FakeEmbedder(),
        )


def test_corrupt_chunks_cannot_be_cited(tmp_path: Path) -> None:
    chunks, index = _built_index(tmp_path)
    (chunks / "chunks.jsonl").write_text("not json\n")

    with pytest.raises(retrieval.RetrievalError, match="invalid JSON"):
        retrieval.search_citations(
            "synthetic query",
            index_directory=index,
            chunk_policy_directory=chunks,
            embedder=FakeEmbedder(),
        )


def test_symlinked_chunks_are_rejected(tmp_path: Path) -> None:
    chunks, index = _built_index(tmp_path / "real")
    linked = tmp_path / "linked-chunks"
    linked.symlink_to(chunks, target_is_directory=True)

    with pytest.raises(retrieval.RetrievalError, match="symbolic link"):
        retrieval.search_citations(
            "synthetic query",
            index_directory=index,
            chunk_policy_directory=linked,
            embedder=FakeEmbedder(),
        )
