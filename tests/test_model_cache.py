"""Network-free tests for pinned model acquisition and verification."""

from __future__ import annotations

from pathlib import Path

import pytest

from amis.model_cache import ModelCacheError, acquire_model, verify_model_snapshot
from amis.model_spec import EMBEDDING_GEMMA, EmbeddingModelSpec, ModelFile
from tests.semantic_factory import sha256_bytes


def _spec() -> EmbeddingModelSpec:
    config = b'{"synthetic":true}\n'
    weights = b"synthetic weights"
    return EmbeddingModelSpec(
        repository="synthetic/model",
        revision="1" * 40,
        files=(
            ModelFile("config.json", len(config), sha256_bytes(config)),
            ModelFile("nested/model.bin", len(weights), sha256_bytes(weights)),
        ),
        dimension=3,
    )


def _download(spec: EmbeddingModelSpec, destination: Path) -> None:
    contents = {
        "config.json": b'{"synthetic":true}\n',
        "nested/model.bin": b"synthetic weights",
    }
    for required in spec.files:
        path = destination / required.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents[required.name])


def test_public_model_spec_is_complete_and_excludes_nonruntime_files() -> None:
    names = {item.name for item in EMBEDDING_GEMMA.files}

    assert len(names) == 15
    assert {
        "model.safetensors",
        "tokenizer.json",
        "tokenizer.model",
        "modules.json",
    } <= names
    assert {".gitattributes", "README.md", "generation_config.json"}.isdisjoint(names)
    assert EMBEDDING_GEMMA.repository == "google/embeddinggemma-300m"
    assert EMBEDDING_GEMMA.revision == "64614b0b8b64f0c6c1e52b07e4e9a4e8fe4d2da2"
    assert EMBEDDING_GEMMA.effective_token_limit == 1984
    assert EMBEDDING_GEMMA.dimension == 768


def test_fake_acquisition_verifies_and_is_idempotent(tmp_path: Path) -> None:
    spec = _spec()
    cache = tmp_path / "cache"
    first = acquire_model(cache, spec=spec, downloader=_download)

    def must_not_download(spec: EmbeddingModelSpec, destination: Path) -> None:
        raise AssertionError("valid cached model must not be downloaded again")

    repeated = acquire_model(cache, spec=spec, downloader=must_not_download)

    assert repeated == first
    assert first.snapshot_directory.is_dir()
    assert first.spec.spec_id.startswith("model_spec_sha256_")
    assert not any(
        path.name.startswith(".") for path in first.snapshot_directory.parent.iterdir()
    )


@pytest.mark.parametrize("failure", ["missing", "size", "hash", "unexpected"])
def test_snapshot_validation_rejects_invalid_content(
    tmp_path: Path, failure: str
) -> None:
    spec = _spec()
    snapshot = tmp_path / "snapshot"
    _download(spec, snapshot)
    if failure == "missing":
        (snapshot / "config.json").unlink()
    elif failure == "size":
        (snapshot / "config.json").write_bytes(b"x")
    elif failure == "hash":
        (snapshot / "config.json").write_bytes(b'{"synthetic":fals}\n')
    else:
        (snapshot / "other.bin").write_bytes(b"unexpected")

    with pytest.raises(ModelCacheError):
        verify_model_snapshot(snapshot, spec=spec)


def test_snapshot_file_cannot_escape_cache_root(tmp_path: Path) -> None:
    spec = _spec()
    cache = tmp_path / "cache"
    snapshot = cache / "snapshot"
    _download(spec, snapshot)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b'{"synthetic":true}\n')
    (snapshot / "config.json").unlink()
    (snapshot / "config.json").symlink_to(outside)

    with pytest.raises(ModelCacheError, match="outside"):
        verify_model_snapshot(snapshot, cache_root=cache, spec=spec)


def test_failed_acquisition_cleans_only_its_staging(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    sentinel = cache / "sentinel"
    sentinel.parent.mkdir()
    sentinel.write_text("preserve")

    def fail(spec: EmbeddingModelSpec, destination: Path) -> None:
        (destination / "partial").write_text("partial")
        raise ModelCacheError("synthetic gated access failure")

    with pytest.raises(ModelCacheError, match="gated"):
        acquire_model(cache, spec=_spec(), downloader=fail)

    assert sentinel.read_text() == "preserve"
    assert not any(path.name.startswith(".") for path in cache.rglob("*"))


def test_snapshot_root_must_not_be_a_symlink(tmp_path: Path) -> None:
    spec = _spec()
    real = tmp_path / "real"
    _download(spec, real)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(ModelCacheError, match="symbolic link"):
        verify_model_snapshot(linked, cache_root=tmp_path, spec=spec)


def test_intermediate_cache_and_snapshot_symlinks_are_rejected(
    tmp_path: Path,
) -> None:
    spec = _spec()
    real = tmp_path / "real"
    snapshot = real / "cache" / "snapshot"
    _download(spec, snapshot)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(ModelCacheError, match="symbolic link"):
        verify_model_snapshot(
            linked / "cache" / "snapshot",
            cache_root=linked / "cache",
            spec=spec,
        )
    with pytest.raises(ModelCacheError, match="symbolic link"):
        acquire_model(linked / "new-cache", spec=spec, downloader=_download)
    assert not (real / "new-cache").exists()


def test_required_file_symlinks_contained_by_real_cache_are_allowed(
    tmp_path: Path,
) -> None:
    spec = _spec()
    cache = tmp_path / "cache"
    snapshot = cache / "snapshot"
    blobs = cache / "blobs"
    _download(spec, snapshot)
    for required in spec.files:
        path = snapshot / required.name
        target = blobs / required.name
        target.parent.mkdir(parents=True, exist_ok=True)
        path.replace(target)
        path.symlink_to(target)

    verified = verify_model_snapshot(snapshot, cache_root=cache, spec=spec)

    assert verified.snapshot_directory == snapshot
