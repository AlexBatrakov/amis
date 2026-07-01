"""Build, persist, load, and search one exact AMIS semantic index."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from amis._path_safety import UnsafePathError, reject_symlink_components
from amis.chunking import (
    CHUNK_MANIFEST_SCHEMA_VERSION,
    CHUNK_SCHEMA_VERSION,
    CHUNKER_VERSION,
)
from amis.embeddings import DOCUMENT_TRANSFORM, QUERY_TRANSFORM, Embedder
from amis.model_spec import EMBEDDING_GEMMA, canonical_json

INDEX_CONFIG_SCHEMA_VERSION = "amis.semantic_index_config.v1"
INDEX_MANIFEST_SCHEMA_VERSION = "amis.semantic_index_manifest.v1"
INDEX_ROW_SCHEMA_VERSION = "amis.semantic_index_row.v1"
INDEXER_VERSION = "amis.semantic_indexer.v1"
RANKING_RULE = "score_desc_document_chunk_index_asc_v1"
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_ID_RE = re.compile(r"[a-z_]+_sha256_[0-9a-f]{64}\Z")
_INDEX_FILES = frozenset({"index_manifest.json", "metadata.jsonl", "vectors.npy"})
_METADATA_KEYS = frozenset(
    {
        "chunk_id",
        "document_chunk_index",
        "document_id",
        "end_char",
        "row_index",
        "schema_version",
        "section_chunk_index",
        "section_id",
        "section_text_sha256",
        "source_content_sha256",
        "source_path",
        "spine_index",
        "start_char",
        "text_sha256",
    }
)


class SemanticIndexError(Exception):
    """Raised when a semantic index operation violates its public contract."""


class _DuplicateKeyError(ValueError):
    pass


@dataclass(frozen=True)
class SemanticIndexResult:
    """Summary of a completed or idempotent build."""

    document_id: str
    policy_id: str
    index_config_id: str
    chunk_count: int
    output_directory: Path
    maximum_token_count: int


@dataclass(frozen=True)
class SearchResult:
    """One passage-free exact-search result."""

    score: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ValidatedChunkText:
    """One validated chunk record with passage text for runtime citation display."""

    chunk_id: str
    document_chunk_index: int
    document_id: str
    end_char: int
    section_chunk_index: int
    section_id: str
    source_path: str
    start_char: int
    text: str
    text_sha256: str


@dataclass(frozen=True)
class LoadedSemanticIndex:
    """A fully validated exact index."""

    directory: Path
    manifest: dict[str, Any]
    vectors: NDArray[np.float32]
    metadata: tuple[dict[str, Any], ...]

    def top_k(self, query_vector: ArrayLike, k: int) -> tuple[SearchResult, ...]:
        """Return exact cosine results with stable document-order ties."""
        if type(k) is not int or not 1 <= k <= len(self.metadata):
            raise SemanticIndexError("k must be an integer between 1 and index size")
        query = np.asarray(query_vector)
        if query.shape != (self.vectors.shape[1],):
            raise SemanticIndexError("query vector has the wrong dimension")
        if not np.issubdtype(query.dtype, np.number):
            raise SemanticIndexError("query vector must be numeric")
        query = np.asarray(query, dtype=np.float32)
        if not np.isfinite(query).all():
            raise SemanticIndexError("query vector must contain only finite values")
        norm = float(np.linalg.norm(query))
        if not math.isfinite(norm) or norm == 0.0:
            raise SemanticIndexError("query vector must have a nonzero finite norm")
        query /= norm
        scores = self.vectors @ query
        document_order = np.asarray(
            [row["document_chunk_index"] for row in self.metadata], dtype=np.int64
        )
        order = np.lexsort((document_order, -scores))[:k]
        return tuple(
            SearchResult(float(scores[index]), dict(self.metadata[index]))
            for index in order
        )


@dataclass(frozen=True)
class _ChunkInput:
    directory: Path
    manifest: dict[str, Any]
    chunks: tuple[dict[str, Any], ...]
    manifest_sha256: str
    chunks_sha256: str
    ordered_chunk_ids_sha256: str


def build_semantic_index(
    chunk_policy_directory: Path | str,
    output_root: Path | str,
    embedder: Embedder,
    *,
    timestamp: datetime | None = None,
) -> SemanticIndexResult:
    """Build and atomically publish one offline semantic index."""
    input_directory = Path(chunk_policy_directory)
    output_directory = Path(output_root)
    _validate_output_isolation(input_directory, output_directory)
    source = _load_chunk_input(input_directory)
    config = _index_configuration(embedder.identity)
    config_id = (
        "index_config_sha256_" + hashlib.sha256(canonical_json(config)).hexdigest()
    )
    destination = (
        output_directory
        / source.manifest["document_id"]
        / source.manifest["policy_id"]
        / config_id
    )
    if destination.exists() or destination.is_symlink():
        existing = load_semantic_index(destination, expected_chunks=input_directory)
        if existing.manifest["index_config_id"] != config_id:
            raise SemanticIndexError("index output already contains conflicting data")
        return SemanticIndexResult(
            source.manifest["document_id"],
            source.manifest["policy_id"],
            config_id,
            len(source.chunks),
            destination,
            existing.manifest["tokens"]["maximum_document_tokens"],
        )

    item_ids = [chunk["chunk_id"] for chunk in source.chunks]
    exact_texts = [chunk["text"] for chunk in source.chunks]
    try:
        token_counts = tuple(embedder.preflight_documents(item_ids, exact_texts))
    except Exception as error:
        if isinstance(error, SemanticIndexError):
            raise
        raise SemanticIndexError(str(error)) from error
    if len(token_counts) != len(source.chunks) or any(
        type(count) is not int or count < 1 for count in token_counts
    ):
        raise SemanticIndexError("embedder returned invalid token counts")
    effective_limit = EMBEDDING_GEMMA.effective_token_limit
    for chunk, count in zip(source.chunks, token_counts, strict=True):
        if count > effective_limit:
            raise SemanticIndexError(
                f"token limit exceeded for {chunk['chunk_id']}: {count} > {effective_limit}"
            )

    try:
        embedded = embedder.embed_documents(item_ids, exact_texts)
    except Exception as error:
        raise SemanticIndexError(str(error)) from error
    if embedded.item_ids != tuple(item_ids):
        raise SemanticIndexError("embedder output order does not match chunk order")
    vectors = _validate_and_normalize_vectors(embedded.vectors, len(source.chunks))
    metadata = tuple(
        _metadata_row(index, chunk) for index, chunk in enumerate(source.chunks)
    )
    metadata_bytes = b"".join(_json_line(row) for row in metadata)

    stage: Path | None = None
    created_directories: list[Path] = []
    try:
        parent = destination.parent
        _make_output_parents(output_directory, parent, created_directories)
        stage = Path(tempfile.mkdtemp(prefix=f".{config_id}.tmp-", dir=parent))
        vectors_path = stage / "vectors.npy"
        with vectors_path.open("xb") as output_file:
            np.save(output_file, vectors, allow_pickle=False)
            output_file.flush()
            os.fsync(output_file.fileno())
        _write_synced(stage / "metadata.jsonl", metadata_bytes)
        manifest = _index_manifest(
            source,
            config,
            config_id,
            embedder.identity,
            vectors,
            _sha256(vectors_path),
            hashlib.sha256(metadata_bytes).hexdigest(),
            max(token_counts),
            timestamp or datetime.now(UTC),
        )
        _write_synced(stage / "index_manifest.json", _json_line(manifest))
        _sync_directory(stage)
        load_semantic_index(stage, expected_chunks=input_directory)
        if destination.exists() or destination.is_symlink():
            raise SemanticIndexError("index output already contains conflicting data")
        os.replace(stage, destination)
        stage = None
        _sync_directory(parent)
    except SemanticIndexError:
        raise
    except OSError as error:
        raise SemanticIndexError("index could not be published atomically") from error
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        for directory in reversed(created_directories):
            _remove_empty(directory)

    return SemanticIndexResult(
        source.manifest["document_id"],
        source.manifest["policy_id"],
        config_id,
        len(source.chunks),
        destination,
        max(token_counts),
    )


def load_semantic_index(
    index_directory: Path | str,
    *,
    expected_chunks: Path | str | None = None,
) -> LoadedSemanticIndex:
    """Load and fully validate one persisted exact index."""
    directory = Path(index_directory)
    _reject_symlink_path(directory, "index path")
    if directory.is_symlink() or not directory.is_dir():
        raise SemanticIndexError("index must be an existing real directory")
    try:
        entries = {entry.name for entry in directory.iterdir()}
    except OSError as error:
        raise SemanticIndexError("index directory could not be read") from error
    if entries != _INDEX_FILES:
        raise SemanticIndexError("index directory has missing or unexpected files")
    paths = {name: directory / name for name in _INDEX_FILES}
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise SemanticIndexError("index artifacts must be ordinary files")

    manifest_bytes = _read_bytes(paths["index_manifest.json"], "index manifest")
    manifest = _parse_json_object(manifest_bytes, "index manifest")
    _validate_manifest(manifest)
    config = manifest["index_configuration"]
    expected_config_id = (
        "index_config_sha256_" + hashlib.sha256(canonical_json(config)).hexdigest()
    )
    if manifest["index_config_id"] != expected_config_id:
        raise SemanticIndexError("index configuration identity mismatch")

    metadata_bytes = _read_bytes(paths["metadata.jsonl"], "index metadata")
    if (
        hashlib.sha256(metadata_bytes).hexdigest()
        != manifest["outputs"]["metadata_sha256"]
    ):
        raise SemanticIndexError("index metadata hash mismatch")
    metadata = tuple(_parse_json_lines(metadata_bytes, "index metadata"))
    _validate_metadata(metadata, manifest)

    if _sha256(paths["vectors.npy"]) != manifest["outputs"]["vectors_sha256"]:
        raise SemanticIndexError("index vector hash mismatch")
    try:
        vectors = np.load(paths["vectors.npy"], allow_pickle=False)
    except (OSError, ValueError) as error:
        raise SemanticIndexError("index vectors could not be loaded safely") from error
    _validate_stored_vectors(vectors, len(metadata), manifest["vectors"]["dimension"])

    if expected_chunks is not None:
        source = _load_chunk_input(Path(expected_chunks))
        identities = manifest["input"]
        expected = {
            "chunk_count": len(source.chunks),
            "chunk_manifest_sha256": source.manifest_sha256,
            "chunks_sha256": source.chunks_sha256,
            "document_id": source.manifest["document_id"],
            "ordered_chunk_ids_sha256": source.ordered_chunk_ids_sha256,
            "policy_id": source.manifest["policy_id"],
        }
        for key, value in expected.items():
            if identities.get(key) != value:
                raise SemanticIndexError("index is stale for the supplied chunk input")
        expected_metadata = tuple(
            _metadata_row(index, chunk) for index, chunk in enumerate(source.chunks)
        )
        if metadata != expected_metadata:
            raise SemanticIndexError(
                "index metadata does not match the supplied chunk input"
            )
    return LoadedSemanticIndex(directory, manifest, vectors, metadata)


def load_validated_chunk_texts(
    chunk_policy_directory: Path | str,
) -> tuple[ValidatedChunkText, ...]:
    """Load fully validated chunk text for runtime citation display."""
    source = _load_chunk_input(Path(chunk_policy_directory))
    return tuple(
        ValidatedChunkText(
            chunk["chunk_id"],
            chunk["document_chunk_index"],
            chunk["document_id"],
            chunk["end_char"],
            chunk["section_chunk_index"],
            chunk["section_id"],
            chunk["source_path"],
            chunk["start_char"],
            chunk["text"],
            chunk["text_sha256"],
        )
        for chunk in source.chunks
    )


def _load_chunk_input(directory: Path) -> _ChunkInput:
    _reject_symlink_path(directory, "chunk input path")
    if directory.is_symlink() or not directory.is_dir():
        raise SemanticIndexError("chunk input must be an existing real directory")
    try:
        entries = {entry.name for entry in directory.iterdir()}
    except OSError as error:
        raise SemanticIndexError("chunk input could not be inspected") from error
    if entries != {"chunk_manifest.json", "chunks.jsonl"}:
        raise SemanticIndexError("chunk input must contain exactly the chunk artifacts")
    manifest_path = directory / "chunk_manifest.json"
    chunks_path = directory / "chunks.jsonl"
    if any(
        path.is_symlink() or not path.is_file() for path in (manifest_path, chunks_path)
    ):
        raise SemanticIndexError("chunk input artifacts must be ordinary files")
    manifest_bytes = _read_bytes(manifest_path, "chunk manifest")
    chunks_bytes = _read_bytes(chunks_path, "chunks")
    manifest = _parse_json_object(manifest_bytes, "chunk manifest")
    chunks = tuple(_parse_json_lines(chunks_bytes, "chunks"))
    _validate_chunk_manifest(manifest, chunks, chunks_bytes)
    _validate_chunks(manifest, chunks)
    ordered_digest = hashlib.sha256(
        canonical_json([chunk["chunk_id"] for chunk in chunks])
    ).hexdigest()
    return _ChunkInput(
        directory,
        manifest,
        chunks,
        hashlib.sha256(manifest_bytes).hexdigest(),
        hashlib.sha256(chunks_bytes).hexdigest(),
        ordered_digest,
    )


def _validate_chunk_manifest(
    manifest: dict[str, Any], chunks: tuple[dict[str, Any], ...], chunks_bytes: bytes
) -> None:
    if manifest.get("schema_version") != CHUNK_MANIFEST_SCHEMA_VERSION:
        raise SemanticIndexError("unsupported chunk manifest schema")
    if manifest.get("chunker_version") != CHUNKER_VERSION:
        raise SemanticIndexError("unsupported chunker version")
    document_id = _require_id(manifest, "document_id", "doc_sha256_")
    source_sha256 = _require_hash(manifest, "source_sha256")
    if document_id != f"doc_sha256_{source_sha256}":
        raise SemanticIndexError("chunk manifest document identity mismatch")
    policy = manifest.get("policy")
    if not isinstance(policy, dict):
        raise SemanticIndexError("chunk manifest policy must be an object")
    expected_policy = (
        "chunk_policy_sha256_" + hashlib.sha256(canonical_json(policy)).hexdigest()
    )
    if manifest.get("policy_id") != expected_policy:
        raise SemanticIndexError("chunk policy identity mismatch")
    if manifest.get("chunk_count") != len(chunks) or not chunks:
        raise SemanticIndexError("chunk count mismatch or empty input")
    if manifest.get("chunks_sha256") != hashlib.sha256(chunks_bytes).hexdigest():
        raise SemanticIndexError("chunk stream hash mismatch")
    for key in ("input_document_sha256", "input_sections_sha256"):
        _require_hash(manifest, key)
    for key in ("target_chars", "max_chars", "overlap_chars"):
        if type(policy.get(key)) is not int:
            raise SemanticIndexError("chunk policy sizes must be integers")
    if (
        policy.get("strategy") != "paragraph_window_v1"
        or policy.get("candidate_filter") != "retrieval_candidate"
    ):
        raise SemanticIndexError("unsupported chunk policy")


def _validate_chunks(
    manifest: dict[str, Any], chunks: tuple[dict[str, Any], ...]
) -> None:
    seen: set[str] = set()
    previous_by_section: dict[str, dict[str, Any]] = {}
    closed_sections: set[str] = set()
    active_section: str | None = None
    prior_spine = -1
    policy = manifest["policy"]
    for expected_index, chunk in enumerate(chunks):
        if chunk.get("schema_version") != CHUNK_SCHEMA_VERSION:
            raise SemanticIndexError("unsupported chunk schema")
        chunk_id = _require_id(chunk, "chunk_id", "chunk_sha256_")
        if chunk_id in seen:
            raise SemanticIndexError("duplicate chunk ID")
        seen.add(chunk_id)
        if chunk.get("document_id") != manifest["document_id"]:
            raise SemanticIndexError("chunk document identity mismatch")
        if chunk.get("policy_id") != manifest["policy_id"]:
            raise SemanticIndexError("chunk policy identity mismatch")
        if chunk.get("document_chunk_index") != expected_index:
            raise SemanticIndexError("chunk document order must be contiguous")
        section_id = _require_id(chunk, "section_id", "sec_sha256_")
        if section_id != active_section:
            if active_section is not None:
                closed_sections.add(active_section)
            if section_id in closed_sections:
                raise SemanticIndexError("section chunks must be contiguous")
            active_section = section_id
        spine_index = _nonnegative_integer(chunk, "spine_index")
        section_index = _nonnegative_integer(chunk, "section_chunk_index")
        start = _nonnegative_integer(chunk, "start_char")
        end = _nonnegative_integer(chunk, "end_char")
        overlap = _nonnegative_integer(chunk, "overlap_left_chars")
        text = chunk.get("text")
        if (
            not isinstance(text, str)
            or not text
            or end <= start
            or len(text) != end - start
        ):
            raise SemanticIndexError("chunk text and coordinates are inconsistent")
        text_sha256 = _require_hash(chunk, "text_sha256")
        _require_hash(chunk, "section_text_sha256")
        _require_hash(chunk, "source_content_sha256")
        if (
            not isinstance(chunk.get("source_path"), str)
            or not chunk["source_path"]
            or Path(chunk["source_path"]).is_absolute()
            or ".." in Path(chunk["source_path"]).parts
        ):
            raise SemanticIndexError("chunk source path must be a nonempty string")
        if chunk.get("role") != "body":
            raise SemanticIndexError("indexed chunks must have the body role")
        if text_sha256 != hashlib.sha256(text.encode("utf-8")).hexdigest():
            raise SemanticIndexError("chunk text hash mismatch")
        if end - start > policy["max_chars"] or overlap > policy["overlap_chars"]:
            raise SemanticIndexError("chunk exceeds policy bounds")
        seed = (
            f"amis:chunk:v1\0{section_id}\0{manifest['policy_id']}\0"
            f"{start}\0{end}\0{text_sha256}"
        )
        if chunk_id != "chunk_sha256_" + hashlib.sha256(seed.encode()).hexdigest():
            raise SemanticIndexError("chunk identity derivation mismatch")
        if spine_index < prior_spine:
            raise SemanticIndexError("chunk spine order is not monotonic")
        prior_spine = spine_index
        previous = previous_by_section.get(section_id)
        if previous is None:
            if section_index != 0 or overlap != 0:
                raise SemanticIndexError("first section chunk metadata is invalid")
        else:
            if section_index != previous["section_chunk_index"] + 1:
                raise SemanticIndexError("section chunk order is not contiguous")
            expected_overlap = max(0, previous["end_char"] - start)
            if (
                overlap != expected_overlap
                or start <= previous["start_char"]
                or end <= previous["end_char"]
            ):
                raise SemanticIndexError("chunk overlap or advancement is invalid")
        previous_by_section[section_id] = chunk


def _index_configuration(identity: dict[str, object]) -> dict[str, object]:
    if not isinstance(identity, dict):
        raise SemanticIndexError("embedder identity must be an object")
    return {
        "document_transform": DOCUMENT_TRANSFORM,
        "embedder": identity,
        "indexer_version": INDEXER_VERSION,
        "model": EMBEDDING_GEMMA.as_dict(),
        "model_spec_id": EMBEDDING_GEMMA.spec_id,
        "query_transform": QUERY_TRANSFORM,
        "ranking_rule": RANKING_RULE,
        "schema_version": INDEX_CONFIG_SCHEMA_VERSION,
        "vectors": {
            "dimension": EMBEDDING_GEMMA.dimension,
            "dtype": "float32",
            "metric": "cosine_via_unit_dot_product",
            "normalization": "explicit_l2",
        },
    }


def _validate_and_normalize_vectors(
    values: NDArray[np.float32], count: int
) -> NDArray[np.float32]:
    array = np.asarray(values)
    expected_shape = (count, EMBEDDING_GEMMA.dimension)
    if array.shape != expected_shape:
        raise SemanticIndexError("embedder output has the wrong shape")
    if array.dtype != np.dtype(np.float32):
        raise SemanticIndexError("embedder output must have dtype float32")
    if not np.isfinite(array).all():
        raise SemanticIndexError("embedder output contains nonfinite values")
    norms = np.linalg.norm(array, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0.0):
        raise SemanticIndexError("embedder output contains a zero or invalid row")
    normalized = np.asarray(array / norms[:, None], dtype=np.float32)
    _validate_stored_vectors(normalized, count, EMBEDDING_GEMMA.dimension)
    return normalized


def _validate_stored_vectors(array: Any, count: int, dimension: int) -> None:
    if not isinstance(array, np.ndarray) or array.shape != (count, dimension):
        raise SemanticIndexError("stored vector matrix has the wrong shape")
    if array.dtype != np.dtype(np.float32):
        raise SemanticIndexError("stored vector matrix must have dtype float32")
    if not np.isfinite(array).all():
        raise SemanticIndexError("stored vector matrix contains nonfinite values")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms == 0.0) or not np.allclose(norms, 1.0, rtol=0.0, atol=1e-5):
        raise SemanticIndexError("stored vector rows must be unit normalized")


def _metadata_row(index: int, chunk: dict[str, Any]) -> dict[str, Any]:
    row = {
        "chunk_id": chunk["chunk_id"],
        "document_chunk_index": chunk["document_chunk_index"],
        "document_id": chunk["document_id"],
        "end_char": chunk["end_char"],
        "row_index": index,
        "schema_version": INDEX_ROW_SCHEMA_VERSION,
        "section_chunk_index": chunk["section_chunk_index"],
        "section_id": chunk["section_id"],
        "section_text_sha256": chunk["section_text_sha256"],
        "source_content_sha256": chunk["source_content_sha256"],
        "source_path": chunk["source_path"],
        "spine_index": chunk["spine_index"],
        "start_char": chunk["start_char"],
        "text_sha256": chunk["text_sha256"],
    }
    return row


def _index_manifest(
    source: _ChunkInput,
    config: dict[str, object],
    config_id: str,
    embedder_identity: dict[str, object],
    vectors: NDArray[np.float32],
    vectors_sha256: str,
    metadata_sha256: str,
    maximum_tokens: int,
    timestamp: datetime,
) -> dict[str, Any]:
    if timestamp.tzinfo is None:
        raise SemanticIndexError("build timestamp must be timezone-aware")
    try:
        amis_version = importlib.metadata.version("amis")
    except importlib.metadata.PackageNotFoundError:
        amis_version = "0.1.0"
    return {
        "amis_version": amis_version,
        "backend": embedder_identity,
        "build": {
            "timestamp_identity": False,
            "timestamp_utc": timestamp.astimezone(UTC)
            .isoformat()
            .replace("+00:00", "Z"),
        },
        "index_config_id": config_id,
        "index_configuration": config,
        "indexer_version": INDEXER_VERSION,
        "input": {
            "chunk_count": len(source.chunks),
            "chunk_manifest_sha256": source.manifest_sha256,
            "chunks_sha256": source.chunks_sha256,
            "document_id": source.manifest["document_id"],
            "input_document_sha256": source.manifest["input_document_sha256"],
            "input_sections_sha256": source.manifest["input_sections_sha256"],
            "ordered_chunk_ids_sha256": source.ordered_chunk_ids_sha256,
            "policy": source.manifest["policy"],
            "policy_id": source.manifest["policy_id"],
            "source_sha256": source.manifest["source_sha256"],
        },
        "model_spec_id": EMBEDDING_GEMMA.spec_id,
        "outputs": {
            "metadata_sha256": metadata_sha256,
            "vectors_sha256": vectors_sha256,
        },
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "ranking_rule": RANKING_RULE,
        "schema_version": INDEX_MANIFEST_SCHEMA_VERSION,
        "tokens": {
            "effective_limit": EMBEDDING_GEMMA.effective_token_limit,
            "hard_limit": EMBEDDING_GEMMA.hard_token_limit,
            "maximum_document_tokens": maximum_tokens,
            "reserve": EMBEDDING_GEMMA.token_reserve,
            "truncated_document_count": 0,
        },
        "vectors": {
            "count": vectors.shape[0],
            "dimension": vectors.shape[1],
            "dtype": "float32",
            "metric": "cosine_via_unit_dot_product",
            "normalization": "explicit_l2",
            "shape": list(vectors.shape),
        },
    }


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != INDEX_MANIFEST_SCHEMA_VERSION:
        raise SemanticIndexError("unsupported semantic index manifest schema")
    if manifest.get("indexer_version") != INDEXER_VERSION:
        raise SemanticIndexError("unsupported semantic indexer version")
    if manifest.get("ranking_rule") != RANKING_RULE:
        raise SemanticIndexError("unsupported semantic ranking rule")
    if not isinstance(manifest.get("index_configuration"), dict):
        raise SemanticIndexError("index configuration is missing")
    if (
        manifest["index_configuration"].get("schema_version")
        != INDEX_CONFIG_SCHEMA_VERSION
    ):
        raise SemanticIndexError("unsupported semantic index configuration")
    configuration = manifest["index_configuration"]
    if (
        configuration.get("model") != EMBEDDING_GEMMA.as_dict()
        or configuration.get("model_spec_id") != EMBEDDING_GEMMA.spec_id
        or configuration.get("document_transform") != DOCUMENT_TRANSFORM
        or configuration.get("query_transform") != QUERY_TRANSFORM
        or configuration.get("ranking_rule") != RANKING_RULE
    ):
        raise SemanticIndexError("semantic index configuration is incompatible")
    if manifest.get("model_spec_id") != EMBEDDING_GEMMA.spec_id:
        raise SemanticIndexError("semantic index model identity mismatch")
    for container in ("input", "outputs", "tokens", "vectors", "build"):
        if not isinstance(manifest.get(container), dict):
            raise SemanticIndexError(f"index manifest {container} must be an object")
    if manifest["build"].get("timestamp_identity") is not False:
        raise SemanticIndexError("build timestamp identity marker is invalid")
    for key in ("metadata_sha256", "vectors_sha256"):
        _require_hash(manifest["outputs"], key)
    if (
        manifest["vectors"].get("dtype") != "float32"
        or manifest["vectors"].get("dimension") != EMBEDDING_GEMMA.dimension
    ):
        raise SemanticIndexError("index vector contract is incompatible")
    count = manifest["input"].get("chunk_count")
    if (
        type(count) is not int
        or count < 1
        or manifest["vectors"].get("count") != count
        or manifest["vectors"].get("shape") != [count, EMBEDDING_GEMMA.dimension]
    ):
        raise SemanticIndexError("index vector count or shape metadata is invalid")
    if (
        manifest["tokens"].get("hard_limit") != EMBEDDING_GEMMA.hard_token_limit
        or manifest["tokens"].get("effective_limit")
        != EMBEDDING_GEMMA.effective_token_limit
        or manifest["tokens"].get("reserve") != EMBEDDING_GEMMA.token_reserve
        or manifest["tokens"].get("truncated_document_count") != 0
    ):
        raise SemanticIndexError("index token contract is incompatible")
    timestamp = manifest["build"].get("timestamp_utc")
    if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
        raise SemanticIndexError("build timestamp is not RFC 3339 UTC")
    try:
        datetime.fromisoformat(timestamp.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise SemanticIndexError("build timestamp is not RFC 3339 UTC") from error


def _validate_metadata(
    metadata: tuple[dict[str, Any], ...], manifest: dict[str, Any]
) -> None:
    if len(metadata) != manifest["input"].get("chunk_count"):
        raise SemanticIndexError("index metadata count mismatch")
    seen: set[str] = set()
    previous_by_section: dict[str, int] = {}
    closed_sections: set[str] = set()
    active_section: str | None = None
    previous_spine = -1
    for expected_index, row in enumerate(metadata):
        if set(row) != _METADATA_KEYS:
            raise SemanticIndexError(
                "index metadata schema contains unexpected fields or passage data"
            )
        if row.get("schema_version") != INDEX_ROW_SCHEMA_VERSION:
            raise SemanticIndexError("unsupported index metadata schema")
        row_index = _nonnegative_integer(row, "row_index")
        document_chunk_index = _nonnegative_integer(row, "document_chunk_index")
        if row_index != expected_index or document_chunk_index != expected_index:
            raise SemanticIndexError("index metadata order is invalid")
        document_id = _require_id(row, "document_id", "doc_sha256_")
        if document_id != manifest["input"].get("document_id"):
            raise SemanticIndexError("index metadata document identity mismatch")
        source_path = row.get("source_path")
        if not _safe_relative_path(source_path):
            raise SemanticIndexError("index metadata source path is invalid")
        chunk_id = _require_id(row, "chunk_id", "chunk_sha256_")
        if chunk_id in seen:
            raise SemanticIndexError("index metadata contains duplicate chunk IDs")
        seen.add(chunk_id)
        for key in ("text_sha256", "section_text_sha256", "source_content_sha256"):
            _require_hash(row, key)
        start = _nonnegative_integer(row, "start_char")
        end = _nonnegative_integer(row, "end_char")
        spine_index = _nonnegative_integer(row, "spine_index")
        section_index = _nonnegative_integer(row, "section_chunk_index")
        if start >= end:
            raise SemanticIndexError("index metadata coordinates are invalid")
        if spine_index < previous_spine:
            raise SemanticIndexError("index metadata spine order is invalid")
        previous_spine = spine_index
        section_id = _require_id(row, "section_id", "sec_sha256_")
        if section_id != active_section:
            if active_section is not None:
                closed_sections.add(active_section)
            if section_id in closed_sections:
                raise SemanticIndexError("index metadata sections are not contiguous")
            active_section = section_id
        previous_section_index = previous_by_section.get(section_id)
        if (
            previous_section_index is None
            and section_index != 0
            or previous_section_index is not None
            and section_index != previous_section_index + 1
        ):
            raise SemanticIndexError("index metadata section order is invalid")
        previous_by_section[section_id] = section_index
        seed = (
            f"amis:chunk:v1\0{section_id}\0{manifest['input'].get('policy_id')}\0"
            f"{start}\0{end}\0{row['text_sha256']}"
        )
        expected_chunk_id = "chunk_sha256_" + hashlib.sha256(seed.encode()).hexdigest()
        if chunk_id != expected_chunk_id:
            raise SemanticIndexError("index metadata chunk identity mismatch")
    digest = hashlib.sha256(
        canonical_json([row["chunk_id"] for row in metadata])
    ).hexdigest()
    if digest != manifest["input"].get("ordered_chunk_ids_sha256"):
        raise SemanticIndexError("index metadata chunk identity digest mismatch")


def _validate_output_isolation(input_directory: Path, output_root: Path) -> None:
    _reject_symlink_path(input_directory, "chunk input path")
    _reject_symlink_path(output_root, "index output path")
    try:
        resolved_input = input_directory.resolve(strict=True)
        resolved_output = output_root.resolve(strict=False)
    except OSError as error:
        raise SemanticIndexError(
            "input or output path could not be resolved"
        ) from error
    if (
        resolved_input == resolved_output
        or resolved_input in resolved_output.parents
        or resolved_output in resolved_input.parents
    ):
        raise SemanticIndexError("index output root must be disjoint from chunk input")


def _make_output_parents(root: Path, parent: Path, created: list[Path]) -> None:
    _reject_symlink_path(parent, "index output path")
    missing: list[Path] = []
    current = parent
    while not current.exists():
        missing.append(current)
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        raise SemanticIndexError("index output parent must be a real directory")
    for directory in reversed(missing):
        directory.mkdir()
        created.append(directory)
        if directory.is_symlink() or not directory.is_dir():
            raise SemanticIndexError("index output parent must be a real directory")
    if root.is_symlink():
        raise SemanticIndexError("index output root must not be a symbolic link")


def _reject_symlink_path(path: Path, label: str) -> None:
    try:
        reject_symlink_components(path)
    except UnsafePathError as error:
        raise SemanticIndexError(
            f"{label} must not traverse a symbolic link"
        ) from error


def _safe_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return bool(path.parts) and not path.is_absolute() and ".." not in path.parts


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise SemanticIndexError(f"{label} could not be read") from error


def _parse_json_object(content: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            content.decode("utf-8"), object_pairs_hook=_without_duplicates
        )
    except (UnicodeError, json.JSONDecodeError, _DuplicateKeyError) as error:
        raise SemanticIndexError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise SemanticIndexError(f"{label} must contain a JSON object")
    return value


def _parse_json_lines(content: bytes, label: str) -> list[dict[str, Any]]:
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise SemanticIndexError(f"{label} is not valid UTF-8") from error
    values: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        if not line:
            raise SemanticIndexError(f"{label} line {number} is empty")
        try:
            value = json.loads(line, object_pairs_hook=_without_duplicates)
        except (json.JSONDecodeError, _DuplicateKeyError) as error:
            raise SemanticIndexError(
                f"{label} line {number} is invalid JSON"
            ) from error
        if not isinstance(value, dict):
            raise SemanticIndexError(f"{label} line {number} must be an object")
        values.append(value)
    return values


def _without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateKeyError(key)
        value[key] = item
    return value


def _require_hash(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or _HASH_RE.fullmatch(item) is None:
        raise SemanticIndexError(f"field {key} must be a lowercase SHA-256")
    return item


def _require_id(value: dict[str, Any], key: str, prefix: str) -> str:
    item = value.get(key)
    if (
        not isinstance(item, str)
        or not item.startswith(prefix)
        or _ID_RE.fullmatch(item) is None
    ):
        raise SemanticIndexError(f"field {key} has an invalid stable ID")
    return item


def _nonnegative_integer(value: dict[str, Any], key: str) -> int:
    item = value.get(key)
    if type(item) is not int or item < 0:
        raise SemanticIndexError(f"field {key} must be a non-negative integer")
    return item


def _json_line(value: object) -> bytes:
    return canonical_json(value) + b"\n"


def _write_synced(path: Path, content: bytes) -> None:
    with path.open("xb") as output_file:
        output_file.write(content)
        output_file.flush()
        os.fsync(output_file.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_empty(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass
