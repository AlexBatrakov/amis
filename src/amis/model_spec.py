"""Pinned model configuration for AMIS semantic indexing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

MODEL_SPEC_SCHEMA_VERSION = "amis.embedding_model_spec.v1"
DOCUMENT_TRANSFORM_VERSION = "embeddinggemma-retrieval-document-v1"
QUERY_TRANSFORM_VERSION = "embeddinggemma-retrieval-query-v1"
DOCUMENT_PROMPT = "title: none | text: "
QUERY_PROMPT = "task: search result | query: "


@dataclass(frozen=True)
class ModelFile:
    """One immutable file required by the pinned runtime."""

    name: str
    size: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True)
class EmbeddingModelSpec:
    """Complete semantic identity of the supported embedding model."""

    repository: str
    revision: str
    files: tuple[ModelFile, ...]
    hard_token_limit: int = 2048
    effective_token_limit: int = 1984
    token_reserve: int = 64
    dimension: int = 768

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": {
                "device": "cpu",
                "python": ">=3.13,<3.14",
                "sentence_transformers": ">=5.5,<5.6",
                "torch": ">=2.12,<2.13",
                "transformers": ">=5.10,<5.11",
            },
            "document_transform": {
                "prompt": DOCUMENT_PROMPT,
                "version": DOCUMENT_TRANSFORM_VERSION,
            },
            "gated": True,
            "license": "Gemma Terms of Use",
            "model_files": [item.as_dict() for item in self.files],
            "output": {
                "dimension": self.dimension,
                "dtype": "float32",
                "metric": "cosine_via_unit_dot_product",
                "normalization": "explicit_l2",
            },
            "provenance_url": "https://huggingface.co/google/embeddinggemma-300m",
            "query_transform": {
                "prompt": QUERY_PROMPT,
                "version": QUERY_TRANSFORM_VERSION,
            },
            "repository": self.repository,
            "revision": self.revision,
            "schema_version": MODEL_SPEC_SCHEMA_VERSION,
            "tokens": {
                "effective_limit": self.effective_token_limit,
                "hard_limit": self.hard_token_limit,
                "reserve": self.token_reserve,
            },
        }

    @property
    def spec_id(self) -> str:
        digest = hashlib.sha256(canonical_json(self.as_dict())).hexdigest()
        return f"model_spec_sha256_{digest}"


def canonical_json(value: object) -> bytes:
    """Serialize one identity object canonically."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


EMBEDDING_GEMMA = EmbeddingModelSpec(
    repository="google/embeddinggemma-300m",
    revision="64614b0b8b64f0c6c1e52b07e4e9a4e8fe4d2da2",
    files=(
        ModelFile(
            "1_Pooling/config.json",
            312,
            "35bbd47d7fdf1e378db6130bcc668b09d1aa67a7bbf7c8f89a9c71f4cc8ebcc6",
        ),
        ModelFile(
            "2_Dense/config.json",
            134,
            "0661e5e0b67b8f8408ab31ab5d073a78972fc1dc24a49992a64796557e4f9e53",
        ),
        ModelFile(
            "2_Dense/model.safetensors",
            9437272,
            "c327f2acb00149676ade24a75e11eb6ebbd367f9ee050267ba56829d2979f702",
        ),
        ModelFile(
            "3_Dense/config.json",
            134,
            "8c4575c49353d63fb907878856ba94384635c3b2711fd5b7439e7f71888c66fc",
        ),
        ModelFile(
            "3_Dense/model.safetensors",
            9437272,
            "ffb6cc5162e11e2ce6bc2367e121ee3bbbc4e82e1ee26826bd7573d4948d81b8",
        ),
        ModelFile(
            "added_tokens.json",
            35,
            "50b2f405ba56a26d4913fd772089992252d7f942123cc0a034d96424221ba946",
        ),
        ModelFile(
            "config.json",
            1488,
            "8f863f76e2d9c710cc833dc92efa898c9adfd41031c786507cc6b0e49c2e3e68",
        ),
        ModelFile(
            "config_sentence_transformers.json",
            997,
            "8eadac15526f83d8950aa8d962a7f4f6e3d678bea71689960194561f33a5f64f",
        ),
        ModelFile(
            "model.safetensors",
            1211486072,
            "cbf5a78393b6a033e0b8a63a57549964f7ed5c6fbeb4ba0694214f36123f2fd2",
        ),
        ModelFile(
            "modules.json",
            573,
            "5b5649645fb756dad1a8e2efe7872d3bb32bc00b93c95f276dd17f474eedccdc",
        ),
        ModelFile(
            "sentence_bert_config.json",
            58,
            "5ea26221ce733ace29a3897360e7c6ac8816b2ca0f7306657d69e594fece7325",
        ),
        ModelFile(
            "special_tokens_map.json",
            662,
            "2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397",
        ),
        ModelFile(
            "tokenizer.json",
            33385008,
            "6852f8d561078cc0cebe70ca03c5bfdd0d60a45f9d2e0e1e4cc05b68e9ec329e",
        ),
        ModelFile(
            "tokenizer.model",
            4689074,
            "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
        ),
        ModelFile(
            "tokenizer_config.json",
            1155346,
            "9076840490613047bc9115963ee96b7702018b0d26ba644240bf856efda93118",
        ),
    ),
)
