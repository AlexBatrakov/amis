"""Acquire reviewed public-domain demo corpus sources."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from amis._path_safety import UnsafePathError, reject_symlink_components

PUBLIC_CORPUS_MANIFEST_SCHEMA_VERSION = "amis.public_corpus_manifest.v1"
PUBLIC_CORPUS_REGISTRY_VERSION = "amis.public_corpus_registry.v1"
DEFAULT_PUBLIC_CORPUS_ROOT = Path("data/raw/public-domain")
SOURCE_MANIFEST_NAME = "source_manifest.json"
MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024


class PublicCorpusError(Exception):
    """Raised when public corpus acquisition cannot complete safely."""


@dataclass(frozen=True)
class PublicCorpusSource:
    """One reviewed downloadable public corpus source."""

    corpus_id: str
    title: str
    author: str
    translator: str
    language: str
    original_publication_year: int
    translation_publication_year: int
    provider: str
    catalog_url: str
    source_url: str
    source_format: str
    artifact_name: str
    expected_sha256: str
    expected_size: int
    source_checked_date: str
    license_url: str
    legal_basis: str
    caveats: tuple[str, ...]
    alternate_sources: tuple[dict[str, str], ...]

    def as_registry_record(self) -> dict[str, Any]:
        """Return a passage-free public registry record."""
        return {
            "artifact_name": self.artifact_name,
            "author": self.author,
            "catalog_url": self.catalog_url,
            "caveats": list(self.caveats),
            "corpus_id": self.corpus_id,
            "expected_sha256": self.expected_sha256,
            "expected_size": self.expected_size,
            "language": self.language,
            "legal_basis": self.legal_basis,
            "license_url": self.license_url,
            "original_publication_year": self.original_publication_year,
            "provider": self.provider,
            "registry_version": PUBLIC_CORPUS_REGISTRY_VERSION,
            "source_checked_date": self.source_checked_date,
            "source_format": self.source_format,
            "source_url": self.source_url,
            "title": self.title,
            "translation_publication_year": self.translation_publication_year,
            "translator": self.translator,
            "alternate_sources": list(self.alternate_sources),
        }


@dataclass(frozen=True)
class PublicCorpusAcquisitionResult:
    """Summary of a completed or already-present public corpus acquisition."""

    corpus_id: str
    artifact_path: Path
    manifest_path: Path
    source_url: str
    final_url: str
    sha256: str
    size: int
    already_present: bool


@dataclass(frozen=True)
class _DownloadedSource:
    content: bytes
    final_url: str
    content_type: str


CRIME_AND_PUNISHMENT_GARNETT = PublicCorpusSource(
    corpus_id="crime-and-punishment-garnett",
    title="Crime and Punishment",
    author="Fyodor Dostoevsky",
    translator="Constance Garnett",
    language="English",
    original_publication_year=1866,
    translation_publication_year=1914,
    provider="Project Gutenberg",
    catalog_url="https://www.gutenberg.org/ebooks/2554",
    source_url="https://www.gutenberg.org/ebooks/2554.epub.noimages",
    source_format="epub2-noimages",
    artifact_name="crime-and-punishment-garnett.epub",
    expected_sha256=(
        "45c4d898bf915fd903ecdcc010551e48eed5128bd92940518fc27969e0fe428a"
    ),
    expected_size=651291,
    source_checked_date="2026-07-01",
    license_url="https://www.gutenberg.org/policy/license.html",
    legal_basis=(
        "Project Gutenberg catalog status: public domain in the USA; "
        "Constance Garnett English translation first published in 1914."
    ),
    caveats=(
        "Copyright status is jurisdiction-specific; users outside the United "
        "States should check local law before downloading or using the source.",
        "Project Gutenberg trademark and redistribution terms are separate from "
        "the underlying text's U.S. copyright status.",
        "The downloaded EPUB is a local raw source artifact and is intentionally "
        "not tracked in this repository.",
    ),
    alternate_sources=(
        {
            "provider": "Standard Ebooks",
            "url": (
                "https://standardebooks.org/ebooks/fyodor-dostoevsky/"
                "crime-and-punishment/constance-garnett"
            ),
            "format_note": "downloadable EPUB is EPUB 3 and needs later loader support",
            "legal_note": (
                "source text/artwork believed public domain in the United States; "
                "Standard Ebooks contributions dedicated via CC0"
            ),
        },
        {
            "provider": "Standard Ebooks source repository",
            "url": (
                "https://github.com/standardebooks/"
                "fyodor-dostoevsky_crime-and-punishment_constance-garnett"
            ),
            "format_note": "source repository is an EPUB 3 source folder",
            "legal_note": (
                "source text/artwork believed public domain in the United States; "
                "contributor work dedicated via CC0"
            ),
        },
    ),
)

PUBLIC_CORPUS_REGISTRY = {
    CRIME_AND_PUNISHMENT_GARNETT.corpus_id: CRIME_AND_PUNISHMENT_GARNETT,
}


def acquire_public_corpus(
    corpus_id: str,
    output_root: Path | str = DEFAULT_PUBLIC_CORPUS_ROOT,
    *,
    timestamp: datetime | None = None,
) -> PublicCorpusAcquisitionResult:
    """Download and stage one reviewed public corpus source."""
    source = _source_for(corpus_id)
    root = Path(output_root)
    corpus_directory = root / source.corpus_id
    artifact_path = corpus_directory / source.artifact_name
    manifest_path = corpus_directory / SOURCE_MANIFEST_NAME
    _reject_unsafe_path(root)
    _reject_unsafe_path(corpus_directory)

    if corpus_directory.exists():
        return _existing_result(source, corpus_directory, artifact_path, manifest_path)

    downloaded = _download_url(source.source_url)
    digest = hashlib.sha256(downloaded.content).hexdigest()
    size = len(downloaded.content)
    _verify_download(source, digest, size)

    stage: Path | None = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        _reject_unsafe_path(root)
        stage = Path(tempfile.mkdtemp(prefix=f".{source.corpus_id}.tmp-", dir=root))
        stage_artifact = stage / source.artifact_name
        stage_manifest = stage / SOURCE_MANIFEST_NAME
        _write_synced(stage_artifact, downloaded.content)
        manifest = _local_manifest(
            source,
            downloaded,
            digest,
            size,
            timestamp or datetime.now(UTC),
        )
        _write_synced(stage_manifest, _json_bytes(manifest))
        _sync_directory(stage)
        if corpus_directory.exists() or corpus_directory.is_symlink():
            raise PublicCorpusError("corpus output already contains conflicting data")
        os.replace(stage, corpus_directory)
        stage = None
        _sync_directory(root)
    except PublicCorpusError:
        raise
    except OSError as error:
        raise PublicCorpusError(
            "public corpus could not be published atomically"
        ) from error
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage, ignore_errors=True)

    return PublicCorpusAcquisitionResult(
        corpus_id=source.corpus_id,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        source_url=source.source_url,
        final_url=downloaded.final_url,
        sha256=digest,
        size=size,
        already_present=False,
    )


def public_corpus_record(corpus_id: str) -> dict[str, Any]:
    """Return passage-free registry metadata for one supported corpus."""
    return _source_for(corpus_id).as_registry_record()


def _source_for(corpus_id: str) -> PublicCorpusSource:
    try:
        return PUBLIC_CORPUS_REGISTRY[corpus_id]
    except KeyError as error:
        raise PublicCorpusError(f"unsupported public corpus ID: {corpus_id}") from error


def _reject_unsafe_path(path: Path) -> None:
    try:
        reject_symlink_components(path)
    except UnsafePathError as error:
        raise PublicCorpusError(str(error)) from error


def _existing_result(
    source: PublicCorpusSource,
    corpus_directory: Path,
    artifact_path: Path,
    manifest_path: Path,
) -> PublicCorpusAcquisitionResult:
    if corpus_directory.is_symlink() or not corpus_directory.is_dir():
        raise PublicCorpusError("corpus output already contains conflicting data")
    _reject_unsafe_path(artifact_path)
    _reject_unsafe_path(manifest_path)
    if not artifact_path.is_file() or not manifest_path.is_file():
        raise PublicCorpusError("corpus output already contains conflicting data")
    if {path.name for path in corpus_directory.iterdir()} != {
        source.artifact_name,
        SOURCE_MANIFEST_NAME,
    }:
        raise PublicCorpusError("corpus output already contains conflicting data")
    try:
        artifact = artifact_path.read_bytes()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicCorpusError(
            "corpus output already contains conflicting data"
        ) from error

    digest = hashlib.sha256(artifact).hexdigest()
    size = len(artifact)
    _verify_download(source, digest, size)
    if not isinstance(manifest, dict):
        raise PublicCorpusError("corpus output already contains conflicting data")
    expected_manifest_values = {
        "artifact_name": source.artifact_name,
        "byte_size": size,
        "corpus_id": source.corpus_id,
        "provider": source.provider,
        "schema_version": PUBLIC_CORPUS_MANIFEST_SCHEMA_VERSION,
        "sha256": digest,
        "source_url": source.source_url,
    }
    for key, expected in expected_manifest_values.items():
        if manifest.get(key) != expected:
            raise PublicCorpusError("corpus output already contains conflicting data")

    final_url = manifest.get("final_url")
    if not isinstance(final_url, str) or not final_url:
        raise PublicCorpusError("corpus output already contains conflicting data")
    return PublicCorpusAcquisitionResult(
        corpus_id=source.corpus_id,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        source_url=source.source_url,
        final_url=final_url,
        sha256=digest,
        size=size,
        already_present=True,
    )


def _download_url(url: str) -> _DownloadedSource:
    if not url.startswith("https://"):
        raise PublicCorpusError("public corpus source URL must use HTTPS")
    request = Request(url, headers={"User-Agent": "AMIS public corpus acquisition"})
    try:
        with urlopen(request, timeout=30) as response:
            chunks: list[bytes] = []
            size = 0
            while chunk := response.read(1024 * 1024):
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise PublicCorpusError("public corpus download exceeds size limit")
            final_url = response.geturl()
            if not final_url.startswith("https://"):
                raise PublicCorpusError("public corpus final URL must use HTTPS")
            content_type = response.headers.get("content-type", "")
    except PublicCorpusError:
        raise
    except (OSError, URLError) as error:
        raise PublicCorpusError(
            "public corpus source could not be downloaded"
        ) from error
    return _DownloadedSource(
        content=b"".join(chunks),
        final_url=final_url,
        content_type=content_type,
    )


def _verify_download(source: PublicCorpusSource, digest: str, size: int) -> None:
    if digest != source.expected_sha256:
        raise PublicCorpusError("public corpus source SHA-256 did not match")
    if size != source.expected_size:
        raise PublicCorpusError("public corpus source byte size did not match")


def _local_manifest(
    source: PublicCorpusSource,
    downloaded: _DownloadedSource,
    digest: str,
    size: int,
    timestamp: datetime,
) -> dict[str, Any]:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return {
        "schema_version": PUBLIC_CORPUS_MANIFEST_SCHEMA_VERSION,
        "registry_version": PUBLIC_CORPUS_REGISTRY_VERSION,
        "corpus_id": source.corpus_id,
        "title": source.title,
        "author": source.author,
        "translator": source.translator,
        "language": source.language,
        "provider": source.provider,
        "catalog_url": source.catalog_url,
        "source_url": source.source_url,
        "final_url": downloaded.final_url,
        "content_type": downloaded.content_type,
        "source_format": source.source_format,
        "artifact_name": source.artifact_name,
        "byte_size": size,
        "sha256": digest,
        "expected_sha256": source.expected_sha256,
        "retrieved_at": timestamp.astimezone(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "source_checked_date": source.source_checked_date,
        "legal_basis": source.legal_basis,
        "license_url": source.license_url,
        "caveats": list(source.caveats),
    }


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_synced(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as output_file:
            output_file.write(content)
            output_file.flush()
            os.fsync(output_file.fileno())
    except OSError as error:
        raise PublicCorpusError("public corpus output could not be written") from error


def _sync_directory(path: Path) -> None:
    try:
        directory_fd = os.open(path, os.O_RDONLY)
    except OSError as error:
        raise PublicCorpusError(
            "public corpus output directory could not be synced"
        ) from error
    try:
        os.fsync(directory_fd)
    except OSError as error:
        raise PublicCorpusError(
            "public corpus output directory could not be synced"
        ) from error
    finally:
        os.close(directory_fd)
