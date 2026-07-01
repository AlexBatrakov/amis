"""Public behavior tests for public-domain corpus acquisition."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pytest import MonkeyPatch

import amis.public_corpus as public_corpus
from amis.public_corpus import (
    PublicCorpusError,
    PublicCorpusSource,
    acquire_public_corpus,
    public_corpus_record,
)


def test_registry_exposes_passage_free_source_metadata() -> None:
    record = public_corpus_record("crime-and-punishment-garnett")

    assert record["title"] == "Crime and Punishment"
    assert record["author"] == "Fyodor Dostoevsky"
    assert record["translator"] == "Constance Garnett"
    assert record["provider"] == "Project Gutenberg"
    assert record["source_format"] == "epub2-noimages"
    assert record["source_url"] == "https://www.gutenberg.org/ebooks/2554.epub.noimages"
    assert record["expected_sha256"] == (
        "45c4d898bf915fd903ecdcc010551e48eed5128bd92940518fc27969e0fe428a"
    )
    assert "public domain in the USA" in record["legal_basis"]
    assert all("Raskolnikov" not in json.dumps(value) for value in record.values())


def test_acquire_public_corpus_writes_artifact_and_manifest(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    content = b"synthetic public-domain epub bytes"
    source = _synthetic_source(content)
    monkeypatch.setattr(
        public_corpus, "PUBLIC_CORPUS_REGISTRY", {source.corpus_id: source}
    )
    monkeypatch.setattr(
        public_corpus,
        "_download_url",
        lambda url: public_corpus._DownloadedSource(
            content=content,
            final_url="https://example.test/final.epub",
            content_type="application/epub+zip",
        ),
    )

    result = acquire_public_corpus(
        source.corpus_id,
        tmp_path / "raw",
        timestamp=datetime(2026, 7, 1, 10, 45, 0, tzinfo=UTC),
    )

    assert result.already_present is False
    assert result.size == len(content)
    assert result.sha256 == hashlib.sha256(content).hexdigest()
    assert result.artifact_path.read_bytes() == content

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest == {
        "artifact_name": source.artifact_name,
        "author": source.author,
        "byte_size": len(content),
        "catalog_url": source.catalog_url,
        "caveats": list(source.caveats),
        "content_type": "application/epub+zip",
        "corpus_id": source.corpus_id,
        "expected_sha256": hashlib.sha256(content).hexdigest(),
        "final_url": "https://example.test/final.epub",
        "language": source.language,
        "legal_basis": source.legal_basis,
        "license_url": source.license_url,
        "provider": source.provider,
        "registry_version": "amis.public_corpus_registry.v1",
        "retrieved_at": "2026-07-01T10:45:00Z",
        "schema_version": "amis.public_corpus_manifest.v1",
        "sha256": hashlib.sha256(content).hexdigest(),
        "source_checked_date": source.source_checked_date,
        "source_format": source.source_format,
        "source_url": source.source_url,
        "title": source.title,
        "translator": source.translator,
    }


def test_repeated_acquire_validates_existing_output_without_network(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    content = b"synthetic public-domain epub bytes"
    source = _synthetic_source(content)
    monkeypatch.setattr(
        public_corpus, "PUBLIC_CORPUS_REGISTRY", {source.corpus_id: source}
    )
    monkeypatch.setattr(
        public_corpus,
        "_download_url",
        lambda url: public_corpus._DownloadedSource(
            content=content,
            final_url="https://example.test/final.epub",
            content_type="application/epub+zip",
        ),
    )
    first = acquire_public_corpus(source.corpus_id, tmp_path / "raw")

    def fail_download(url: str) -> public_corpus._DownloadedSource:
        raise AssertionError("existing valid corpus must not be downloaded again")

    monkeypatch.setattr(public_corpus, "_download_url", fail_download)

    second = acquire_public_corpus(source.corpus_id, tmp_path / "raw")

    assert second.already_present is True
    assert second.artifact_path == first.artifact_path
    assert second.manifest_path == first.manifest_path
    assert second.sha256 == first.sha256


def test_hash_mismatch_leaves_no_corpus_directory(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    source = _synthetic_source(b"expected bytes")
    monkeypatch.setattr(
        public_corpus, "PUBLIC_CORPUS_REGISTRY", {source.corpus_id: source}
    )
    monkeypatch.setattr(
        public_corpus,
        "_download_url",
        lambda url: public_corpus._DownloadedSource(
            content=b"different bytes",
            final_url="https://example.test/final.epub",
            content_type="application/epub+zip",
        ),
    )

    with pytest.raises(PublicCorpusError, match="SHA-256"):
        acquire_public_corpus(source.corpus_id, tmp_path / "raw")

    assert not (tmp_path / "raw" / source.corpus_id).exists()


def test_conflicting_existing_output_is_not_modified(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    content = b"synthetic public-domain epub bytes"
    source = _synthetic_source(content)
    output_root = tmp_path / "raw"
    corpus_directory = output_root / source.corpus_id
    corpus_directory.mkdir(parents=True)
    conflict = corpus_directory / source.artifact_name
    conflict.write_bytes(b"conflict")
    monkeypatch.setattr(
        public_corpus, "PUBLIC_CORPUS_REGISTRY", {source.corpus_id: source}
    )

    with pytest.raises(PublicCorpusError, match="conflicting data"):
        acquire_public_corpus(source.corpus_id, output_root)

    assert conflict.read_bytes() == b"conflict"


def test_existing_output_with_extra_files_is_rejected(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    content = b"synthetic public-domain epub bytes"
    source = _synthetic_source(content)
    monkeypatch.setattr(
        public_corpus, "PUBLIC_CORPUS_REGISTRY", {source.corpus_id: source}
    )
    monkeypatch.setattr(
        public_corpus,
        "_download_url",
        lambda url: public_corpus._DownloadedSource(
            content=content,
            final_url="https://example.test/final.epub",
            content_type="application/epub+zip",
        ),
    )
    result = acquire_public_corpus(source.corpus_id, tmp_path / "raw")
    (result.artifact_path.parent / "extra.txt").write_text("unexpected\n")

    with pytest.raises(PublicCorpusError, match="conflicting data"):
        acquire_public_corpus(source.corpus_id, tmp_path / "raw")


def test_rejects_unsupported_corpus_id(tmp_path: Path) -> None:
    with pytest.raises(PublicCorpusError, match="unsupported public corpus ID"):
        acquire_public_corpus("unknown-corpus", tmp_path / "raw")


def test_rejects_symlink_output_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(PublicCorpusError, match="symbolic link"):
        acquire_public_corpus("crime-and-punishment-garnett", link)


def _synthetic_source(content: bytes) -> PublicCorpusSource:
    return PublicCorpusSource(
        corpus_id="synthetic-public-book",
        title="Synthetic Public Book",
        author="Example Author",
        translator="Example Translator",
        language="English",
        original_publication_year=1900,
        translation_publication_year=1901,
        provider="Example Public Source",
        catalog_url="https://example.test/catalog",
        source_url="https://example.test/source.epub",
        source_format="epub2",
        artifact_name="synthetic-public-book.epub",
        expected_sha256=hashlib.sha256(content).hexdigest(),
        expected_size=len(content),
        source_checked_date="2026-07-01",
        license_url="https://example.test/license",
        legal_basis="Synthetic public fixture for tests.",
        caveats=("Synthetic fixture caveat.",),
        alternate_sources=(),
    )
