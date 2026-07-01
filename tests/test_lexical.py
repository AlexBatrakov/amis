"""Public tests for deterministic BM25 lexical retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import amis.lexical as lexical
from tests.semantic_factory import write_chunk_policy_from_texts


def _lexical_chunks(tmp_path: Path) -> Path:
    return write_chunk_policy_from_texts(
        tmp_path / "input",
        [
            "Alpha alpha beta appears in the first synthetic notice.",
            "Alpha gamma appears in the second synthetic notice.",
            "Delta epsilon appears in the third synthetic notice.",
        ],
    )


def test_tokenizer_uses_nfkc_casefold_and_unicode_lmn_categories() -> None:
    assert lexical.tokenize_lexical_text(
        "Stra\u00dfe \uff2112 cafe\u0301 \u0661\u0662!"
    ) == ("strasse", "a12", "café", "\u0661\u0662")


def test_lexical_search_returns_ranked_bounded_citations(tmp_path: Path) -> None:
    chunks = _lexical_chunks(tmp_path)

    result = lexical.search_lexical_citations(
        "alpha beta",
        chunk_policy_directory=chunks,
        top_k=2,
        excerpt_chars=32,
    )

    assert result.query_token_count == 2
    assert [citation.rank for citation in result.citations] == [1, 2]
    assert [citation.document_chunk_index for citation in result.citations] == [0, 1]
    assert result.citations[0].lexical_score > result.citations[1].lexical_score
    assert result.citations[0].lexical_score > 0.0
    assert len(result.citations[0].excerpt) <= 32
    assert result.citations[0].excerpt.endswith("...")
    assert result.citations[0].source_path.startswith("OPS/Text/")
    assert not Path(result.citations[0].source_path).is_absolute()


def test_duplicate_query_terms_increase_bm25_score(tmp_path: Path) -> None:
    chunks = _lexical_chunks(tmp_path)

    single = lexical.search_lexical_citations(
        "beta",
        chunk_policy_directory=chunks,
        top_k=1,
    )
    repeated = lexical.search_lexical_citations(
        "beta beta",
        chunk_policy_directory=chunks,
        top_k=1,
    )

    assert repeated.citations[0].document_chunk_index == 0
    assert repeated.citations[0].lexical_score == pytest.approx(
        single.citations[0].lexical_score * 2
    )


def test_zero_score_rows_tie_break_by_document_chunk_index(tmp_path: Path) -> None:
    chunks = _lexical_chunks(tmp_path)

    result = lexical.search_lexical_citations(
        "missing",
        chunk_policy_directory=chunks,
        top_k=3,
    )

    assert [citation.document_chunk_index for citation in result.citations] == [
        0,
        1,
        2,
    ]
    assert [citation.lexical_score for citation in result.citations] == [0.0] * 3


@pytest.mark.parametrize("top_k", [0, True])
def test_invalid_top_k_fails_before_loading_chunks(
    tmp_path: Path, top_k: object
) -> None:
    with pytest.raises(lexical.LexicalRetrievalError, match="top-k"):
        lexical.search_lexical_citations(
            "alpha",
            chunk_policy_directory=tmp_path / "missing",
            top_k=top_k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("excerpt_chars", [0, 1201, True])
def test_invalid_excerpt_size_fails_before_loading_chunks(
    tmp_path: Path, excerpt_chars: object
) -> None:
    with pytest.raises(lexical.LexicalRetrievalError, match="excerpt-chars"):
        lexical.search_lexical_citations(
            "alpha",
            chunk_policy_directory=tmp_path / "missing",
            excerpt_chars=excerpt_chars,  # type: ignore[arg-type]
        )


def test_empty_or_untokenizable_query_is_rejected(tmp_path: Path) -> None:
    chunks = _lexical_chunks(tmp_path)

    with pytest.raises(lexical.LexicalRetrievalError, match="empty"):
        lexical.search_lexical_citations("   ", chunk_policy_directory=chunks)
    with pytest.raises(lexical.LexicalRetrievalError, match="lexical token"):
        lexical.search_lexical_citations("?! --", chunk_policy_directory=chunks)


def test_top_k_larger_than_chunk_count_is_rejected(tmp_path: Path) -> None:
    chunks = _lexical_chunks(tmp_path)

    with pytest.raises(lexical.LexicalRetrievalError, match="top-k"):
        lexical.search_lexical_citations(
            "alpha", chunk_policy_directory=chunks, top_k=4
        )


def test_stale_chunk_manifest_is_rejected(tmp_path: Path) -> None:
    chunks = _lexical_chunks(tmp_path)
    manifest_path = chunks / "chunk_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["chunks_sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )

    with pytest.raises(lexical.LexicalRetrievalError, match="chunk stream hash"):
        lexical.search_lexical_citations("alpha", chunk_policy_directory=chunks)


def test_corrupt_chunks_are_rejected(tmp_path: Path) -> None:
    chunks = _lexical_chunks(tmp_path)
    (chunks / "chunks.jsonl").write_text("not json\n")

    with pytest.raises(lexical.LexicalRetrievalError, match="invalid JSON"):
        lexical.search_lexical_citations("alpha", chunk_policy_directory=chunks)


def test_symlinked_chunks_are_rejected(tmp_path: Path) -> None:
    chunks = _lexical_chunks(tmp_path / "real")
    linked = tmp_path / "linked-chunks"
    linked.symlink_to(chunks, target_is_directory=True)

    with pytest.raises(lexical.LexicalRetrievalError, match="symbolic link"):
        lexical.search_lexical_citations("alpha", chunk_policy_directory=linked)
