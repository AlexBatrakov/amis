"""Synthetic semantic-index components for public tests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from amis.chunking import ChunkPolicy, chunk_document
from amis.embeddings import EmbeddingError, EmbeddingOutput
from tests.normalized_factory import SyntheticSection, write_normalized_document


class FakeEmbedder:
    """Deterministic, configurable embedding backend with no external I/O."""

    def __init__(
        self,
        *,
        token_counts: tuple[int, ...] | None = None,
        vectors: np.ndarray | None = None,
        output_ids: tuple[str, ...] | None = None,
        preflight_error: str | None = None,
    ) -> None:
        self.token_counts = token_counts
        self.vectors = vectors
        self.output_ids = output_ids
        self.preflight_error = preflight_error
        self.embed_calls = 0

    @property
    def identity(self) -> dict[str, object]:
        return {
            "backend": "deterministic_fake_v1",
            "batch_size": 8,
            "dependencies": {"numpy": np.__version__},
            "device": "cpu",
            "threads": 4,
        }

    def preflight_documents(
        self, item_ids: Sequence[str], exact_texts: Sequence[str]
    ) -> tuple[int, ...]:
        if self.preflight_error:
            raise EmbeddingError(self.preflight_error)
        return self.token_counts or tuple(10 + index for index in range(len(item_ids)))

    def preflight_queries(
        self, item_ids: Sequence[str], exact_texts: Sequence[str]
    ) -> tuple[int, ...]:
        counts = tuple(len(text.split()) + 8 for text in exact_texts)
        if any(count > 1984 for count in counts):
            raise EmbeddingError("query token limit exceeded")
        return counts

    def embed_documents(
        self, item_ids: Sequence[str], exact_texts: Sequence[str]
    ) -> EmbeddingOutput:
        self.embed_calls += 1
        if self.vectors is None:
            vectors = np.zeros((len(item_ids), 768), dtype=np.float32)
            for index in range(len(item_ids)):
                vectors[index, index % 768] = np.float32(index + 1)
        else:
            vectors = self.vectors
        return EmbeddingOutput(self.output_ids or tuple(item_ids), vectors)

    def embed_queries(
        self, item_ids: Sequence[str], exact_texts: Sequence[str]
    ) -> EmbeddingOutput:
        vectors = np.zeros((len(item_ids), 768), dtype=np.float32)
        vectors[:, 0] = 1.0
        return EmbeddingOutput(tuple(item_ids), vectors)


def write_chunk_policy(root: Path, *, section_count: int = 3) -> Path:
    """Create a valid, small chunk-policy directory."""
    sections = [
        SyntheticSection(f"Synthetic semantic passage number {index}.")
        for index in range(section_count)
    ]
    normalized = write_normalized_document(root / "normalized", sections)
    result = chunk_document(
        normalized,
        root / "chunks",
        ChunkPolicy(target_chars=100, max_chars=120, overlap_chars=10),
    )
    return result.output_directory


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
