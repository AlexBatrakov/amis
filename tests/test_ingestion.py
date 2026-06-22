"""Public behavior tests for deterministic EPUB 2 ingestion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pytest import CaptureFixture, MonkeyPatch

import amis.ingestion as ingestion
from amis.cli import main
from amis.ingestion import IngestionError, ingest_epub
from tests.epub_factory import build_epub


def test_ingestion_writes_stable_ordered_records(tmp_path: Path) -> None:
    source = build_epub(tmp_path / "fixture.epub")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first = ingest_epub(source, first_root)
    repeated = ingest_epub(source, first_root)
    second = ingest_epub(source, second_root)

    expected_document_id = f"doc_sha256_{source_hash}"
    assert first.document_id == expected_document_id
    assert repeated == first
    assert second.document_id == first.document_id
    assert first.section_count == 5

    first_document = first.output_directory / "document.json"
    first_sections = first.output_directory / "sections.jsonl"
    second_document = second.output_directory / "document.json"
    second_sections = second.output_directory / "sections.jsonl"
    assert first_document.read_bytes() == second_document.read_bytes()
    assert first_sections.read_bytes() == second_sections.read_bytes()
    assert first_document.read_bytes().endswith(b"\n")
    assert first_sections.read_bytes().endswith(b"\n")

    document = json.loads(first_document.read_text())
    sections = _read_sections(first_sections)
    assert document["schema_version"] == "amis.document.v1"
    assert document["source_basename"] == "fixture.epub"
    assert document["source_sha256"] == source_hash
    assert document["section_count"] == 5
    assert document["loader_version"] == "amis.epub.v1"
    assert [section["spine_index"] for section in sections] == list(range(5))
    assert [section["source_path"] for section in sections] == [
        "OPS/Text/front.xhtml",
        "OPS/Text/part one.xhtml",
        "OPS/Text/body-two.xhtml",
        "OPS/Text/notes.xhtml",
        "OPS/Text/copyright.xhtml",
    ]
    assert [section["role"] for section in sections] == [
        "frontmatter",
        "body",
        "body",
        "notes",
        "backmatter",
    ]
    assert [section["retrieval_candidate"] for section in sections] == [
        False,
        True,
        True,
        False,
        False,
    ]
    assert sections[-1]["linear"] is False
    assert "nonlinear_spine_item" in sections[-1]["warnings"]
    assert "navigation_incomplete" in _document_warning_codes(document)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash


def test_generated_fixture_is_byte_deterministic(tmp_path: Path) -> None:
    first = build_epub(tmp_path / "first.epub")
    second = build_epub(tmp_path / "second.epub")

    assert first.read_bytes() == second.read_bytes()


def test_normalization_scene_breaks_titles_and_note_targets(tmp_path: Path) -> None:
    source = build_epub(tmp_path / "fixture.epub")
    result = ingest_epub(source, tmp_path / "output")
    sections = _read_sections(result.output_directory / "sections.jsonl")
    body_one = sections[1]
    body_two = sections[2]
    notes = sections[3]

    assert body_one["text"] == (
        "First Synthetic Part\n\n"
        "Clockwork ravens crossed paper skies.\n\n"
        "A nested sentence\ncontinues on another line.\n\n"
        "After the marker, 1 gears hummed. Outside"
    )
    assert body_one["scene_break_offsets"] == [
        body_one["text"].index("\n\nAfter the marker")
    ]
    assert body_one["note_targets"] == ["OPS/Text/notes.xhtml#note-1"]
    assert "external_link_ignored" in body_one["warnings"]
    assert body_one["heading_count"] == 1
    assert body_one["paragraph_count"] == 4
    assert body_one["link_count"] == 2
    assert body_one["title_origin"] == "heading"

    assert body_two["title"] == "Second Synthetic Part"
    assert body_two["title_origin"] == "navigation"
    assert "scene_break_ambiguous" in body_two["warnings"]
    assert notes["note_targets"] == ["OPS/Text/part one.xhtml#ref-1"]
    assert (
        body_one["text_sha256"] == hashlib.sha256(body_one["text"].encode()).hexdigest()
    )


def test_legacy_name_anchor_is_a_valid_note_fragment(tmp_path: Path) -> None:
    source = build_epub(tmp_path / "legacy-name.epub", "legacy_name_target")

    result = ingest_epub(source, tmp_path / "output")
    sections = _read_sections(result.output_directory / "sections.jsonl")

    assert sections[1]["note_targets"] == ["OPS/Text/notes.xhtml#note-1"]
    assert "internal_link_target_missing" not in sections[1]["warnings"]


@pytest.mark.parametrize(
    ("anomaly", "fragment", "expect_warning"),
    [
        ("nonspine_id_target", "aux-note", False),
        ("nonspine_name_target", "aux-note", False),
        ("nonspine_missing_fragment", "missing-note", True),
    ],
)
def test_nonspine_content_fragments_are_validated_without_emitting_sections(
    tmp_path: Path, anomaly: str, fragment: str, expect_warning: bool
) -> None:
    source = build_epub(tmp_path / f"{anomaly}.epub", anomaly)

    result = ingest_epub(source, tmp_path / "output")
    sections = _read_sections(result.output_directory / "sections.jsonl")
    body_one = sections[1]

    assert len(sections) == 5
    assert "OPS/Text/auxiliary.xhtml" not in {
        section["source_path"] for section in sections
    }
    assert body_one["note_targets"] == [f"OPS/Text/auxiliary.xhtml#{fragment}"]
    assert ("internal_link_target_missing" in body_one["warnings"]) is expect_warning


def test_section_ids_follow_the_accepted_seed(tmp_path: Path) -> None:
    source = build_epub(tmp_path / "fixture.epub")
    result = ingest_epub(source, tmp_path / "output")
    document = json.loads((result.output_directory / "document.json").read_text())
    sections = _read_sections(result.output_directory / "sections.jsonl")

    for section in sections:
        seed = (
            f"amis:section:v1\0{document['document_id']}\0"
            f"{document['package_path']}\0{section['source_path']}\0"
            f"{section['spine_index']}"
        )
        expected = "sec_sha256_" + hashlib.sha256(seed.encode()).hexdigest()
        assert section["section_id"] == expected
        assert section["document_id"] == document["document_id"]


def test_cli_ingests_one_explicit_source(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    source = build_epub(tmp_path / "fixture.epub")

    assert main(["ingest", str(source), "--output", str(tmp_path / "out")]) == 0

    captured = capsys.readouterr()
    assert "with 5 ordered sections" in captured.out
    assert captured.err == ""


@pytest.mark.parametrize(
    ("anomaly", "warning_code"),
    [
        ("navigation_missing", "navigation_missing"),
        ("navigation_empty", "navigation_empty"),
        ("navigation_malformed", "navigation_missing"),
        ("missing_internal_link", "internal_link_target_missing"),
        ("unknown_role", "section_role_unknown"),
    ],
)
def test_recoverable_anomalies_are_retained_with_warnings(
    tmp_path: Path, anomaly: str, warning_code: str
) -> None:
    source = build_epub(tmp_path / f"{anomaly}.epub", anomaly)

    result = ingest_epub(source, tmp_path / "output")

    document = json.loads((result.output_directory / "document.json").read_text())
    sections = _read_sections(result.output_directory / "sections.jsonl")
    assert len(sections) == 5
    assert warning_code in _document_warning_codes(document)
    if anomaly == "unknown_role":
        assert sections[-1]["role"] == "unknown"
        assert sections[-1]["retrieval_candidate"] is False
        assert warning_code in sections[-1]["warnings"]


@pytest.mark.parametrize(
    ("anomaly", "message"),
    [
        ("non_zip", "valid ZIP"),
        ("misplaced_mimetype", "first archive member"),
        ("compressed_mimetype", "without compression"),
        ("invalid_mimetype", "value is invalid"),
        ("missing_container", "container document is missing"),
        ("malformed_container", "container document is malformed XML"),
        ("zero_rootfiles", "exactly one package rootfile"),
        ("multiple_rootfiles", "exactly one package rootfile"),
        ("unsupported_rootfile_media", "media type is unsupported"),
        ("missing_package", "package document is missing"),
        ("malformed_package", "package document is malformed XML"),
        ("unsupported_version", "unsupported EPUB package version"),
        ("missing_manifest", "manifest is missing or malformed"),
        ("empty_manifest", "manifest is empty"),
        ("missing_spine", "spine is missing or malformed"),
        ("empty_spine", "spine is empty"),
        ("unresolved_spine", "unresolved idref"),
        ("missing_spine_resource", "spine resource is missing"),
        ("unsupported_spine_media", "media type is unsupported"),
        ("malformed_content", "spine content is malformed XML"),
        ("doctype_content", "unsupported declarations"),
        ("missing_body", "exactly one body"),
        ("encryption", "encrypted or DRM-protected"),
        ("unsafe_member", "unsafe member path"),
        ("duplicate_member", "duplicate member names"),
        ("bad_crc", "CRC check"),
    ],
)
def test_structural_failures_leave_no_output(
    tmp_path: Path, anomaly: str, message: str
) -> None:
    source = build_epub(tmp_path / f"{anomaly}.epub", anomaly)
    output_root = tmp_path / "output"

    with pytest.raises(IngestionError, match=message):
        ingest_epub(source, output_root)

    assert not output_root.exists() or list(output_root.iterdir()) == []


def test_member_and_archive_size_limits(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    member_source = build_epub(tmp_path / "member.epub", "oversize_member")
    monkeypatch.setattr(ingestion, "MAX_MEMBER_SIZE", 100)
    with pytest.raises(IngestionError, match="member exceeds"):
        ingest_epub(member_source, tmp_path / "member-output")

    archive_source = build_epub(tmp_path / "archive.epub")
    monkeypatch.setattr(ingestion, "MAX_MEMBER_SIZE", 64 * 1024 * 1024)
    monkeypatch.setattr(ingestion, "MAX_ARCHIVE_SIZE", 100)
    with pytest.raises(IngestionError, match="archive exceeds"):
        ingest_epub(archive_source, tmp_path / "archive-output")


def test_publish_failure_removes_staging_directory(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    source = build_epub(tmp_path / "fixture.epub")
    output_root = tmp_path / "output"

    def fail_replace(source_path: Path, destination_path: Path) -> None:
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(ingestion.os, "replace", fail_replace)
    with pytest.raises(IngestionError, match="published atomically"):
        ingest_epub(source, output_root)

    assert list(output_root.iterdir()) == []


def test_conflicting_existing_output_is_not_modified(tmp_path: Path) -> None:
    source = build_epub(tmp_path / "fixture.epub")
    output_root = tmp_path / "output"
    result = ingest_epub(source, output_root)
    document_path = result.output_directory / "document.json"
    document_path.write_text("synthetic conflict\n")

    with pytest.raises(IngestionError, match="conflicting data"):
        ingest_epub(source, output_root)

    assert document_path.read_text() == "synthetic conflict\n"
    assert not any(path.name.startswith(".") for path in output_root.iterdir())


def test_cli_reports_ingestion_failure(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    source = build_epub(tmp_path / "invalid.epub", "non_zip")

    assert main(["ingest", str(source), "--output", str(tmp_path / "out")]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("amis ingest: error:")


def _read_sections(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _document_warning_codes(document: dict[str, object]) -> list[str]:
    return [warning["code"] for warning in document["warnings"]]
