"""Public behavior tests for exact semantic-index construction and loading."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pytest import MonkeyPatch

import amis.embeddings as embeddings
import amis.semantic_index as semantic_index
from amis.embeddings import EmbeddingError, SentenceTransformerEmbedder
from amis.model_spec import EMBEDDING_GEMMA
from amis.semantic_index import (
    SemanticIndexError,
    build_semantic_index,
    load_semantic_index,
)
from tests.normalized_factory import read_records
from tests.semantic_factory import FakeEmbedder, write_chunk_policy


def _rewrite_metadata(
    index_directory: Path, change: Callable[[dict[str, Any]], None]
) -> tuple[bytes, bytes]:
    metadata_path = index_directory / "metadata.jsonl"
    rows = read_records(metadata_path)
    change(rows[0])
    content = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    metadata_path.write_bytes(content)
    manifest_path = index_directory / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["metadata_sha256"] = hashlib.sha256(content).hexdigest()
    manifest_content = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    manifest_path.write_bytes(manifest_content)
    return content, manifest_content


def test_end_to_end_build_load_and_exact_top_k(tmp_path: Path) -> None:
    chunks = write_chunk_policy(tmp_path / "input")
    result = build_semantic_index(
        chunks,
        tmp_path / "indexes",
        FakeEmbedder(),
        timestamp=datetime(2026, 6, 22, tzinfo=UTC),
    )

    loaded = load_semantic_index(result.output_directory, expected_chunks=chunks)
    assert loaded.vectors.shape == (3, 768)
    assert loaded.vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(loaded.vectors, axis=1), 1.0, atol=1e-5)
    assert result.maximum_token_count == 12
    assert set(path.name for path in result.output_directory.iterdir()) == {
        "index_manifest.json",
        "metadata.jsonl",
        "vectors.npy",
    }
    assert all("text" not in row and "prompt" not in row for row in loaded.metadata)
    assert [
        item.metadata["document_chunk_index"]
        for item in loaded.top_k(np.eye(1, 768, dtype=np.float32)[0], 3)
    ] == [0, 1, 2]
    assert loaded.top_k(np.eye(1, 768, dtype=np.float32)[0], 1)[
        0
    ].score == pytest.approx(1.0)


def test_timestamp_is_nonidentity_and_outputs_repeat(tmp_path: Path) -> None:
    chunks = write_chunk_policy(tmp_path / "input")
    first = build_semantic_index(
        chunks,
        tmp_path / "first",
        FakeEmbedder(),
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    second = build_semantic_index(
        chunks,
        tmp_path / "second",
        FakeEmbedder(),
        timestamp=datetime(2026, 2, 1, tzinfo=UTC),
    )
    first_manifest = json.loads(
        (first.output_directory / "index_manifest.json").read_text()
    )
    second_manifest = json.loads(
        (second.output_directory / "index_manifest.json").read_text()
    )

    assert first.index_config_id == second.index_config_id
    assert first_manifest["outputs"] == second_manifest["outputs"]
    assert first_manifest["build"] != second_manifest["build"]
    assert first_manifest["build"]["timestamp_identity"] is False


def test_equivalent_existing_index_is_idempotent_without_embedding(
    tmp_path: Path,
) -> None:
    chunks = write_chunk_policy(tmp_path / "input")
    output = tmp_path / "indexes"
    first_embedder = FakeEmbedder()
    first = build_semantic_index(chunks, output, first_embedder)
    second_embedder = FakeEmbedder(preflight_error="must not run")

    repeated = build_semantic_index(chunks, output, second_embedder)

    assert repeated.output_directory == first.output_directory
    assert first_embedder.embed_calls == 1
    assert second_embedder.embed_calls == 0


@pytest.mark.parametrize(
    "vectors",
    [
        np.ones((2, 768), dtype=np.float32),
        np.ones((3, 768), dtype=np.float64),
        np.vstack(
            [
                np.ones((1, 768), dtype=np.float32),
                np.full((1, 768), np.nan, dtype=np.float32),
                np.ones((1, 768), dtype=np.float32),
            ]
        ),
        np.vstack(
            [
                np.ones((1, 768), dtype=np.float32),
                np.zeros((1, 768), dtype=np.float32),
                np.ones((1, 768), dtype=np.float32),
            ]
        ),
    ],
)
def test_malformed_complete_matrix_is_rejected_before_output(
    tmp_path: Path, vectors: np.ndarray
) -> None:
    chunks = write_chunk_policy(tmp_path / "input")
    with pytest.raises(SemanticIndexError):
        build_semantic_index(
            chunks, tmp_path / "indexes", FakeEmbedder(vectors=vectors)
        )
    assert not (tmp_path / "indexes").exists()


def test_reordered_output_is_rejected(tmp_path: Path) -> None:
    chunks = write_chunk_policy(tmp_path / "input")
    records = read_records(chunks / "chunks.jsonl")
    reversed_ids = tuple(row["chunk_id"] for row in reversed(records))

    with pytest.raises(SemanticIndexError, match="order"):
        build_semantic_index(
            chunks,
            tmp_path / "indexes",
            FakeEmbedder(output_ids=reversed_ids),
        )


@pytest.mark.parametrize(
    "embedder",
    [
        FakeEmbedder(token_counts=(10, 11)),
        FakeEmbedder(token_counts=(10, 1985, 12)),
        FakeEmbedder(
            preflight_error="backend token counts do not match untruncated counts"
        ),
    ],
)
def test_token_preflight_failures_happen_before_embedding(
    tmp_path: Path, embedder: FakeEmbedder
) -> None:
    chunks = write_chunk_policy(tmp_path / "input")
    with pytest.raises(SemanticIndexError):
        build_semantic_index(chunks, tmp_path / "indexes", embedder)
    assert embedder.embed_calls == 0
    assert not (tmp_path / "indexes").exists()


def test_query_preflight_contract_rejects_overflow() -> None:
    embedder = FakeEmbedder()
    with pytest.raises(EmbeddingError, match="query token"):
        embedder.preflight_queries(["query-1"], ["word " * 2000])


def test_production_preflight_uses_supported_preprocess_api() -> None:
    class Count:
        def item(self) -> int:
            return 3

    class MaskRow:
        def count_nonzero(self) -> Count:
            return Count()

    class Model:
        def preprocess(self, values: list[str]) -> dict[str, list[MaskRow]]:
            return {"attention_mask": [MaskRow() for _ in values]}

        def tokenize(self, values: list[str]) -> None:
            raise AssertionError("deprecated tokenize API must not be called")

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
            return {"input_ids": [1, 2, 3]}

    embedder = SentenceTransformerEmbedder.__new__(SentenceTransformerEmbedder)
    embedder.spec = EMBEDDING_GEMMA
    embedder._model = Model()
    embedder._tokenizer = Tokenizer()

    assert embedder.preflight_documents(["chunk-1"], ["synthetic"]) == (3,)


def test_missing_optional_runtime_is_actionable(monkeypatch: MonkeyPatch) -> None:
    def missing(package: str) -> str:
        raise embeddings.metadata.PackageNotFoundError(package)

    monkeypatch.setattr(embeddings.metadata, "version", missing)
    with pytest.raises(EmbeddingError, match="optional dependencies"):
        embeddings._validate_runtime_versions()


@pytest.mark.parametrize("k", [0, 4, True])
def test_top_k_rejects_invalid_k(tmp_path: Path, k: object) -> None:
    chunks = write_chunk_policy(tmp_path / "input")
    result = build_semantic_index(chunks, tmp_path / "indexes", FakeEmbedder())
    loaded = load_semantic_index(result.output_directory)
    with pytest.raises(SemanticIndexError, match="k"):
        loaded.top_k(np.ones(768, dtype=np.float32), k)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "query",
    [
        np.ones(767, dtype=np.float32),
        np.zeros(768, dtype=np.float32),
        np.full(768, np.inf, dtype=np.float32),
        np.asarray(["x"] * 768),
    ],
)
def test_top_k_rejects_invalid_query_vector(tmp_path: Path, query: np.ndarray) -> None:
    chunks = write_chunk_policy(tmp_path / "input")
    result = build_semantic_index(chunks, tmp_path / "indexes", FakeEmbedder())
    with pytest.raises(SemanticIndexError, match="query vector"):
        load_semantic_index(result.output_directory).top_k(query, 1)


def test_corrupt_existing_index_is_preserved_as_a_conflict(tmp_path: Path) -> None:
    chunks = write_chunk_policy(tmp_path / "input")
    output = tmp_path / "indexes"
    result = build_semantic_index(chunks, output, FakeEmbedder())
    manifest = result.output_directory / "index_manifest.json"
    manifest.write_text("synthetic conflict\n")

    with pytest.raises(SemanticIndexError):
        build_semantic_index(chunks, output, FakeEmbedder())

    assert manifest.read_text() == "synthetic conflict\n"


def test_load_rejects_corrupt_vectors(tmp_path: Path) -> None:
    chunks = write_chunk_policy(tmp_path / "input")
    result = build_semantic_index(chunks, tmp_path / "indexes", FakeEmbedder())
    vectors = result.output_directory / "vectors.npy"
    vectors.write_bytes(vectors.read_bytes() + b"corrupt")

    with pytest.raises(SemanticIndexError, match="hash"):
        load_semantic_index(result.output_directory)


def test_load_rejects_metadata_with_passage_text_even_when_rehashed(
    tmp_path: Path,
) -> None:
    chunks = write_chunk_policy(tmp_path / "input")
    result = build_semantic_index(chunks, tmp_path / "indexes", FakeEmbedder())
    metadata_path = result.output_directory / "metadata.jsonl"
    rows = read_records(metadata_path)
    rows[0]["text"] = "synthetic forbidden passage"
    content = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    metadata_path.write_bytes(content)
    manifest_path = result.output_directory / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["metadata_sha256"] = hashlib.sha256(content).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )

    with pytest.raises(SemanticIndexError, match="passage"):
        load_semantic_index(result.output_directory)


@pytest.mark.parametrize(
    "change",
    [
        lambda row: row.__setitem__("start_char", row["start_char"] + 1),
        lambda row: row.__setitem__("section_id", "sec_sha256_" + "0" * 64),
        lambda row: row.__setitem__("source_path", "synthetic/changed.xhtml"),
        lambda row: row.__setitem__("source_content_sha256", "0" * 64),
        lambda row: row.__setitem__("text_sha256", "0" * 64),
    ],
    ids=["coordinate", "section", "source-path", "source-hash", "text-hash"],
)
def test_rehashed_metadata_provenance_must_match_expected_chunks(
    tmp_path: Path, change: Callable[[dict[str, Any]], None]
) -> None:
    chunks = write_chunk_policy(tmp_path / "input")
    result = build_semantic_index(chunks, tmp_path / "indexes", FakeEmbedder())
    _rewrite_metadata(result.output_directory, change)

    with pytest.raises(SemanticIndexError):
        load_semantic_index(result.output_directory, expected_chunks=chunks)


def test_rehashed_coordinate_is_structurally_rejected_without_chunks(
    tmp_path: Path,
) -> None:
    chunks = write_chunk_policy(tmp_path / "input")
    result = build_semantic_index(chunks, tmp_path / "indexes", FakeEmbedder())
    _rewrite_metadata(
        result.output_directory,
        lambda row: row.__setitem__("start_char", row["start_char"] + 1),
    )

    with pytest.raises(SemanticIndexError, match="chunk identity"):
        load_semantic_index(result.output_directory)


def test_rehashed_unexpected_metadata_key_is_rejected(tmp_path: Path) -> None:
    chunks = write_chunk_policy(tmp_path / "input")
    result = build_semantic_index(chunks, tmp_path / "indexes", FakeEmbedder())
    _rewrite_metadata(
        result.output_directory,
        lambda row: row.__setitem__("content", "synthetic forbidden passage"),
    )

    with pytest.raises(SemanticIndexError, match="schema"):
        load_semantic_index(result.output_directory)


def test_rehashed_provenance_conflict_is_preserved(tmp_path: Path) -> None:
    chunks = write_chunk_policy(tmp_path / "input")
    output = tmp_path / "indexes"
    result = build_semantic_index(chunks, output, FakeEmbedder())
    expected_metadata, expected_manifest = _rewrite_metadata(
        result.output_directory,
        lambda row: row.__setitem__("source_path", "synthetic/changed.xhtml"),
    )

    with pytest.raises(SemanticIndexError, match="supplied chunk input"):
        build_semantic_index(chunks, output, FakeEmbedder())

    assert (
        result.output_directory / "metadata.jsonl"
    ).read_bytes() == expected_metadata
    assert (
        result.output_directory / "index_manifest.json"
    ).read_bytes() == expected_manifest


def test_chunk_input_and_output_symlinks_are_rejected(tmp_path: Path) -> None:
    chunks = write_chunk_policy(tmp_path / "input")
    linked_input = tmp_path / "linked-input"
    linked_input.symlink_to(chunks, target_is_directory=True)
    with pytest.raises(SemanticIndexError, match="symbolic link"):
        build_semantic_index(linked_input, tmp_path / "indexes", FakeEmbedder())

    real_output = tmp_path / "real-output"
    real_output.mkdir()
    linked_output = tmp_path / "linked-output"
    linked_output.symlink_to(real_output, target_is_directory=True)
    with pytest.raises(SemanticIndexError, match="symbolic link"):
        build_semantic_index(chunks, linked_output, FakeEmbedder())
    assert list(real_output.iterdir()) == []


def test_intermediate_chunk_input_and_index_load_symlinks_are_rejected(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    chunks = write_chunk_policy(real / "input")
    result = build_semantic_index(chunks, real / "indexes", FakeEmbedder())
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    linked_chunks = linked / chunks.relative_to(real)
    with pytest.raises(SemanticIndexError, match="symbolic link"):
        build_semantic_index(linked_chunks, tmp_path / "other-indexes", FakeEmbedder())

    linked_index = linked / result.output_directory.relative_to(real)
    with pytest.raises(SemanticIndexError, match="symbolic link"):
        load_semantic_index(linked_index)


@pytest.mark.parametrize("existing", [True, False])
def test_intermediate_output_symlink_is_rejected_before_writes(
    tmp_path: Path, existing: bool
) -> None:
    chunks = write_chunk_policy(tmp_path / "input")
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    relative_output = Path("output") if existing else Path("missing/output")
    if existing:
        (real / relative_output).mkdir()

    with pytest.raises(SemanticIndexError, match="symbolic link"):
        build_semantic_index(chunks, linked / relative_output, FakeEmbedder())

    assert not any((real / relative_output).rglob("vectors.npy"))
    assert not any(path.name.startswith(".") for path in real.rglob("*"))


@pytest.mark.parametrize("relation", ["equal", "ancestor", "descendant"])
def test_output_must_be_disjoint_from_chunk_input(
    tmp_path: Path, relation: str
) -> None:
    chunks = write_chunk_policy(tmp_path / "input")
    if relation == "equal":
        output = chunks
    elif relation == "ancestor":
        output = chunks.parent
    else:
        output = chunks / "index"
    with pytest.raises(SemanticIndexError, match="disjoint"):
        build_semantic_index(chunks, output, FakeEmbedder())


def test_injected_publication_failure_leaves_no_final_or_stage(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    chunks = write_chunk_policy(tmp_path / "input")
    output = tmp_path / "indexes"

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(semantic_index.os, "replace", fail_replace)
    with pytest.raises(SemanticIndexError, match="atomically"):
        build_semantic_index(chunks, output, FakeEmbedder())

    assert not output.exists() or list(output.rglob("*")) == []


def test_failed_sibling_publication_preserves_prior_valid_index(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    class AlternateFake(FakeEmbedder):
        @property
        def identity(self) -> dict[str, object]:
            return {**super().identity, "backend": "alternate_fake_v1"}

    chunks = write_chunk_policy(tmp_path / "input")
    output = tmp_path / "indexes"
    prior = build_semantic_index(chunks, output, FakeEmbedder())

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(semantic_index.os, "replace", fail_replace)
    with pytest.raises(SemanticIndexError, match="atomically"):
        build_semantic_index(chunks, output, AlternateFake())

    assert load_semantic_index(prior.output_directory).vectors.shape == (3, 768)
    assert not any(
        path.name.startswith(".") for path in prior.output_directory.parent.iterdir()
    )


def test_load_rejects_stale_chunk_identity(tmp_path: Path) -> None:
    chunks = write_chunk_policy(tmp_path / "input")
    result = build_semantic_index(chunks, tmp_path / "indexes", FakeEmbedder())
    manifest_path = chunks / "chunk_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["input_document_sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )

    with pytest.raises(SemanticIndexError, match="stale"):
        load_semantic_index(result.output_directory, expected_chunks=chunks)
