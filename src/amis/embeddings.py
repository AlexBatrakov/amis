"""Replaceable embedding boundary and the local EmbeddingGemma adapter."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from amis.model_spec import (
    DOCUMENT_PROMPT,
    DOCUMENT_TRANSFORM_VERSION,
    EMBEDDING_GEMMA,
    QUERY_PROMPT,
    QUERY_TRANSFORM_VERSION,
    EmbeddingModelSpec,
)


class EmbeddingError(Exception):
    """Raised when tokenization or embedding violates the adapter contract."""


@dataclass(frozen=True)
class EmbeddingOutput:
    """Ordered model output bound to caller-provided item identifiers."""

    item_ids: tuple[str, ...]
    vectors: NDArray[np.float32]


class Embedder(Protocol):
    """Model-independent boundary consumed by the semantic index builder."""

    @property
    def identity(self) -> dict[str, object]: ...

    def preflight_documents(
        self, item_ids: Sequence[str], exact_texts: Sequence[str]
    ) -> tuple[int, ...]: ...

    def preflight_queries(
        self, item_ids: Sequence[str], exact_texts: Sequence[str]
    ) -> tuple[int, ...]: ...

    def embed_documents(
        self, item_ids: Sequence[str], exact_texts: Sequence[str]
    ) -> EmbeddingOutput: ...

    def embed_queries(
        self, item_ids: Sequence[str], exact_texts: Sequence[str]
    ) -> EmbeddingOutput: ...


class SentenceTransformerEmbedder:
    """Pinned CPU-only Sentence Transformers adapter for EmbeddingGemma."""

    def __init__(
        self,
        snapshot_directory: Path | str,
        *,
        spec: EmbeddingModelSpec = EMBEDDING_GEMMA,
        batch_size: int = 8,
        threads: int = 4,
    ) -> None:
        _validate_runtime_versions()
        if sys.version_info[:2] != (3, 13):
            raise EmbeddingError("semantic indexing requires Python 3.13")
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            from transformers import AutoTokenizer
        except ImportError as error:
            raise EmbeddingError(
                'embedding requires the "semantic-index" optional dependencies'
            ) from error

        self.spec = spec
        self.batch_size = batch_size
        self.threads = threads
        self.snapshot_directory = Path(snapshot_directory)
        _validate_snapshot_prompts(self.snapshot_directory)
        torch.set_num_threads(threads)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.snapshot_directory, local_files_only=True
            )
            self._model = SentenceTransformer(
                str(self.snapshot_directory), local_files_only=True, device="cpu"
            )
        except Exception as error:
            raise EmbeddingError(
                "the verified local model could not be loaded"
            ) from error
        if str(self._model.device) != "cpu":
            raise EmbeddingError("pinned embedding model must run on CPU")
        if int(self._tokenizer.model_max_length) != spec.hard_token_limit:
            raise EmbeddingError(
                "pinned tokenizer hard limit does not match the model spec"
            )

    @property
    def identity(self) -> dict[str, object]:
        return {
            "backend": "sentence_transformers_cpu_v1",
            "batch_size": self.batch_size,
            "dependencies": {
                name: metadata.version(name)
                for name in (
                    "huggingface-hub",
                    "numpy",
                    "safetensors",
                    "sentence-transformers",
                    "tokenizers",
                    "torch",
                    "transformers",
                )
            },
            "device": "cpu",
            "model_spec_id": self.spec.spec_id,
            "threads": self.threads,
        }

    def preflight_documents(
        self, item_ids: Sequence[str], exact_texts: Sequence[str]
    ) -> tuple[int, ...]:
        return self._preflight(item_ids, exact_texts, DOCUMENT_PROMPT)

    def preflight_queries(
        self, item_ids: Sequence[str], exact_texts: Sequence[str]
    ) -> tuple[int, ...]:
        return self._preflight(item_ids, exact_texts, QUERY_PROMPT)

    def embed_documents(
        self, item_ids: Sequence[str], exact_texts: Sequence[str]
    ) -> EmbeddingOutput:
        return self._embed(item_ids, exact_texts, DOCUMENT_PROMPT)

    def embed_queries(
        self, item_ids: Sequence[str], exact_texts: Sequence[str]
    ) -> EmbeddingOutput:
        return self._embed(item_ids, exact_texts, QUERY_PROMPT)

    def _preflight(
        self, item_ids: Sequence[str], exact_texts: Sequence[str], prompt: str
    ) -> tuple[int, ...]:
        _validate_parallel_inputs(item_ids, exact_texts)
        transformed = [prompt + text for text in exact_texts]
        raw_counts = tuple(self._raw_count(text) for text in transformed)
        for item_id, count in zip(item_ids, raw_counts, strict=True):
            if count > self.spec.effective_token_limit:
                raise EmbeddingError(
                    f"token limit exceeded for {item_id}: {count} > "
                    f"{self.spec.effective_token_limit}"
                )
        try:
            encoded = self._model.preprocess(transformed)
            backend_counts = tuple(
                int(row.count_nonzero().item()) for row in encoded["attention_mask"]
            )
        except Exception as error:
            raise EmbeddingError("backend token preflight failed") from error
        if raw_counts != backend_counts:
            raise EmbeddingError("backend token counts do not match untruncated counts")
        return raw_counts

    def _raw_count(self, text: str) -> int:
        encoded = self._tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        return len(encoded["input_ids"])

    def _embed(
        self, item_ids: Sequence[str], exact_texts: Sequence[str], prompt: str
    ) -> EmbeddingOutput:
        _validate_parallel_inputs(item_ids, exact_texts)
        transformed = [prompt + text for text in exact_texts]
        try:
            vectors = self._model.encode(
                transformed,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        except Exception as error:
            raise EmbeddingError("document embedding failed") from error
        return EmbeddingOutput(tuple(item_ids), np.asarray(vectors))


def _validate_parallel_inputs(
    item_ids: Sequence[str], exact_texts: Sequence[str]
) -> None:
    if not item_ids or len(item_ids) != len(exact_texts):
        raise EmbeddingError("embedding inputs must have equal nonzero counts")
    if len(set(item_ids)) != len(item_ids):
        raise EmbeddingError("embedding input identifiers must be unique")


def _validate_snapshot_prompts(snapshot: Path) -> None:
    try:
        config = json.loads(
            (snapshot / "config_sentence_transformers.json").read_text("utf-8")
        )
        prompts = config["prompts"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise EmbeddingError(
            "model prompt configuration could not be validated"
        ) from error
    if (
        prompts.get("document") != DOCUMENT_PROMPT
        or prompts.get("query") != QUERY_PROMPT
    ):
        raise EmbeddingError(
            "model prompt configuration does not match the pinned spec"
        )


def _validate_runtime_versions() -> None:
    requirements = {
        "sentence-transformers": ((5, 5), (5, 6)),
        "torch": ((2, 12), (2, 13)),
        "transformers": ((5, 10), (5, 11)),
    }
    for package, (minimum, maximum) in requirements.items():
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError as error:
            raise EmbeddingError(
                'embedding requires the "semantic-index" optional dependencies'
            ) from error
        numeric = tuple(int(part) for part in version.split(".")[:2])
        if not minimum <= numeric < maximum:
            raise EmbeddingError(f"unsupported {package} version {version}")


DOCUMENT_TRANSFORM = {
    "prompt": DOCUMENT_PROMPT,
    "version": DOCUMENT_TRANSFORM_VERSION,
}
QUERY_TRANSFORM = {"prompt": QUERY_PROMPT, "version": QUERY_TRANSFORM_VERSION}
