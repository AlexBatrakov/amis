"""Explicit acquisition and local verification of the pinned model snapshot."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from amis._path_safety import UnsafePathError, reject_symlink_components
from amis.model_spec import EMBEDDING_GEMMA, EmbeddingModelSpec

MODEL_CACHE_ENV = "AMIS_MODEL_CACHE"
_KNOWN_AUXILIARY_FILES = frozenset(
    {".gitattributes", "README.md", "generation_config.json"}
)


class ModelCacheError(Exception):
    """Raised when the pinned model cannot be acquired or verified safely."""


@dataclass(frozen=True)
class VerifiedModel:
    """A verified local snapshot and its immutable specification."""

    snapshot_directory: Path
    spec: EmbeddingModelSpec


ModelDownloader = Callable[[EmbeddingModelSpec, Path], None]


def default_model_cache() -> Path:
    """Return the configurable user-local model cache root."""
    configured = os.environ.get(MODEL_CACHE_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "amis" / "models"


def snapshot_directory(
    cache_root: Path | str, spec: EmbeddingModelSpec = EMBEDDING_GEMMA
) -> Path:
    """Return the stable snapshot location for a model specification."""
    repository_name = spec.repository.replace("/", "--")
    return Path(cache_root) / repository_name / spec.revision


def verify_model_snapshot(
    snapshot: Path | str,
    *,
    cache_root: Path | str | None = None,
    spec: EmbeddingModelSpec = EMBEDDING_GEMMA,
) -> VerifiedModel:
    """Verify every required local model file without network access."""
    snapshot_path = Path(snapshot)
    _reject_symlink_path(snapshot_path, "model snapshot")
    selected_root = Path(cache_root or snapshot_path)
    _reject_symlink_path(selected_root, "model cache root")
    if snapshot_path.is_symlink() or not snapshot_path.is_dir():
        raise ModelCacheError("model snapshot must be an existing real directory")
    try:
        resolved_snapshot = snapshot_path.resolve(strict=True)
        allowed_root = selected_root.resolve(strict=True)
    except OSError as error:
        raise ModelCacheError("model snapshot path could not be resolved") from error
    if (
        resolved_snapshot != allowed_root
        and allowed_root not in resolved_snapshot.parents
    ):
        raise ModelCacheError("model snapshot must be contained by the cache root")

    expected_names = {item.name for item in spec.files}
    observed_names: set[str] = set()
    try:
        for path in snapshot_path.rglob("*"):
            if path.is_dir():
                continue
            observed_names.add(path.relative_to(snapshot_path).as_posix())
    except OSError as error:
        raise ModelCacheError("model snapshot could not be inspected") from error
    unexpected = observed_names - expected_names - _KNOWN_AUXILIARY_FILES
    if unexpected:
        raise ModelCacheError("model snapshot contains unexpected files")

    for required in spec.files:
        path = snapshot_path / required.name
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise ModelCacheError(
                f"model snapshot is missing {required.name}"
            ) from error
        if resolved != allowed_root and allowed_root not in resolved.parents:
            raise ModelCacheError("model file resolves outside the cache root")
        if not path.is_file():
            raise ModelCacheError(f"model snapshot is missing {required.name}")
        try:
            if path.stat().st_size != required.size:
                raise ModelCacheError(f"model file size mismatch: {required.name}")
            digest = _sha256(path)
        except ModelCacheError:
            raise
        except OSError as error:
            raise ModelCacheError(
                f"model file could not be read: {required.name}"
            ) from error
        if digest != required.sha256:
            raise ModelCacheError(f"model file hash mismatch: {required.name}")
    return VerifiedModel(snapshot_path, spec)


def acquire_model(
    cache_root: Path | str,
    *,
    spec: EmbeddingModelSpec = EMBEDDING_GEMMA,
    downloader: ModelDownloader | None = None,
) -> VerifiedModel:
    """Download, verify, and atomically publish the exact pinned snapshot."""
    root = Path(cache_root)
    destination = snapshot_directory(root, spec)
    _reject_symlink_path(destination, "model cache path")
    if destination.exists() or destination.is_symlink():
        return verify_model_snapshot(destination, cache_root=root, spec=spec)
    if root.is_symlink():
        raise ModelCacheError("model cache root must not be a symbolic link")

    stage: Path | None = None
    root_created = not root.exists()
    repository_root = destination.parent
    repository_created = not repository_root.exists()
    try:
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir() or root.is_symlink():
            raise ModelCacheError("model cache root must be a real directory")
        repository_root.mkdir(exist_ok=True)
        if repository_root.is_symlink() or not repository_root.is_dir():
            raise ModelCacheError("model repository cache must be a real directory")
        stage = Path(
            tempfile.mkdtemp(prefix=f".{spec.revision}.tmp-", dir=repository_root)
        )
        (downloader or _huggingface_downloader)(spec, stage)
        verified = verify_model_snapshot(stage, cache_root=repository_root, spec=spec)
        _sync_tree(stage)
        if destination.exists() or destination.is_symlink():
            raise ModelCacheError("model cache contains conflicting data")
        os.replace(stage, destination)
        stage = None
        _sync_directory(repository_root)
        return VerifiedModel(destination, verified.spec)
    except ModelCacheError:
        raise
    except OSError as error:
        raise ModelCacheError(
            "model snapshot could not be published atomically"
        ) from error
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        if repository_created:
            _remove_empty(repository_root)
        if root_created:
            _remove_empty(root)


def _huggingface_downloader(spec: EmbeddingModelSpec, destination: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    except ImportError as error:
        raise ModelCacheError(
            'model acquisition requires the "semantic-index" optional dependencies'
        ) from error

    download_cache = destination / ".download-cache"
    try:
        for required in spec.files:
            cached = Path(
                hf_hub_download(
                    repo_id=spec.repository,
                    filename=required.name,
                    revision=spec.revision,
                    cache_dir=download_cache,
                )
            )
            target = destination / required.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, target)
        shutil.rmtree(download_cache)
    except GatedRepoError as error:
        raise ModelCacheError(
            "model access is gated; accept the Gemma Terms of Use and authenticate "
            "with Hugging Face before retrying"
        ) from error
    except HfHubHTTPError as error:
        raise ModelCacheError(
            "the exact pinned model revision could not be acquired"
        ) from error
    except OSError as error:
        raise ModelCacheError("downloaded model files could not be staged") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as input_file:
                os.fsync(input_file.fileno())
    for path in sorted(
        (item for item in root.rglob("*") if item.is_dir()), reverse=True
    ):
        _sync_directory(path)
    _sync_directory(root)


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


def _reject_symlink_path(path: Path, label: str) -> None:
    try:
        reject_symlink_components(path)
    except UnsafePathError as error:
        raise ModelCacheError(f"{label} must not traverse a symbolic link") from error
