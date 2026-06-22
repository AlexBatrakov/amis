"""Public behavior tests for deterministic normalized-text chunking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pytest import CaptureFixture, MonkeyPatch

import amis.chunking as chunking
from amis.chunking import ChunkingError, ChunkPolicy, chunk_document
from amis.cli import main
from amis.ingestion import ingest_epub
from tests.epub_factory import build_epub
from tests.normalized_factory import (
    SyntheticSection,
    read_records,
    rewrite_document,
    rewrite_sections,
    write_normalized_document,
)


def test_policy_identity_uses_canonical_complete_policy() -> None:
    policy = ChunkPolicy()
    canonical = (
        b'{"candidate_filter":"retrieval_candidate","max_chars":4000,'
        b'"overlap_chars":400,"strategy":"paragraph_window_v1",'
        b'"target_chars":3000}'
    )

    assert policy.as_dict() == json.loads(canonical)
    assert policy.policy_id == (
        "chunk_policy_sha256_" + hashlib.sha256(canonical).hexdigest()
    )


@pytest.mark.parametrize(
    "policy",
    [
        ChunkPolicy(target_chars=10, max_chars=10, overlap_chars=0),
        ChunkPolicy(target_chars=10, max_chars=12, overlap_chars=9),
    ],
)
def test_valid_policy_edges(policy: ChunkPolicy) -> None:
    assert policy.policy_id.startswith("chunk_policy_sha256_")


@pytest.mark.parametrize(
    "values",
    [
        (0, 1, 0),
        (10, 9, 0),
        (10, 10, 10),
        (10, 12, -1),
        (True, 12, 0),
    ],
)
def test_invalid_policy_is_rejected(values: tuple[object, object, object]) -> None:
    with pytest.raises(ChunkingError, match="policy"):
        ChunkPolicy(
            target_chars=values[0],  # type: ignore[arg-type]
            max_chars=values[1],  # type: ignore[arg-type]
            overlap_chars=values[2],  # type: ignore[arg-type]
        )


def test_generated_epub_ingestion_to_chunking_is_exact_and_ordered(
    tmp_path: Path,
) -> None:
    source = build_epub(tmp_path / "fixture.epub")
    ingested = ingest_epub(source, tmp_path / "normalized")
    policy = ChunkPolicy(target_chars=50, max_chars=75, overlap_chars=10)

    result = chunk_document(ingested.output_directory, tmp_path / "chunks", policy)

    chunks_path = result.output_directory / "chunks.jsonl"
    manifest_path = result.output_directory / "chunk_manifest.json"
    chunks = read_records(chunks_path)
    manifest = json.loads(manifest_path.read_text())
    sections = read_records(ingested.output_directory / "sections.jsonl")
    candidates = [section for section in sections if section["retrieval_candidate"]]
    sections_by_id = {section["section_id"]: section for section in candidates}

    assert manifest["schema_version"] == "amis.chunk_manifest.v1"
    assert manifest["chunker_version"] == "amis.chunker.v1"
    assert manifest["total_section_count"] == 5
    assert manifest["eligible_section_count"] == 2
    assert manifest["skipped_section_count"] == 3
    assert manifest["chunk_count"] == len(chunks) == result.chunk_count
    assert manifest["policy"] == policy.as_dict()
    assert manifest["policy_id"] == policy.policy_id == result.policy_id
    assert (
        manifest["chunks_sha256"]
        == hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    )
    assert (
        manifest["input_document_sha256"]
        == hashlib.sha256(
            (ingested.output_directory / "document.json").read_bytes()
        ).hexdigest()
    )
    assert (
        manifest["input_sections_sha256"]
        == hashlib.sha256(
            (ingested.output_directory / "sections.jsonl").read_bytes()
        ).hexdigest()
    )
    assert [chunk["document_chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )
    assert [chunk["spine_index"] for chunk in chunks] == sorted(
        chunk["spine_index"] for chunk in chunks
    )

    for chunk in chunks:
        section = sections_by_id[chunk["section_id"]]
        expected_text = section["text"][chunk["start_char"] : chunk["end_char"]]
        assert chunk["schema_version"] == "amis.chunk.v1"
        assert chunk["text"] == expected_text
        assert (
            chunk["text_sha256"] == hashlib.sha256(expected_text.encode()).hexdigest()
        )
        assert chunk["section_text_sha256"] == section["text_sha256"]
        assert chunk["end_char"] - chunk["start_char"] <= policy.max_chars
        seed = (
            f"amis:chunk:v1\0{chunk['section_id']}\0{policy.policy_id}\0"
            f"{chunk['start_char']}\0{chunk['end_char']}\0{chunk['text_sha256']}"
        )
        assert chunk["chunk_id"] == (
            "chunk_sha256_" + hashlib.sha256(seed.encode()).hexdigest()
        )
    _assert_complete_coverage(candidates, chunks)
    _assert_bounded_overlap(chunks, policy.overlap_chars)


@pytest.mark.parametrize(
    ("text", "scene_offsets", "max_chars", "expected_end", "warning_kind"),
    [
        (
            "A" * 25 + "\n\n" + "B" * 25 + "\n\n" + "C" * 70,
            (52,),
            60,
            52,
            None,
        ),
        ("A" * 44 + "\n\n" + "B" * 80, (), 50, 44, None),
        ("A" * 34 + ". " + "B" * 80, (), 40, 35, "sentence"),
        ("A" * 28 + " " + "B" * 80, (), 40, 28, "whitespace"),
        ("A" * 100, (), 40, 40, "hard"),
    ],
)
def test_boundary_priority_and_oversize_fallbacks(
    tmp_path: Path,
    text: str,
    scene_offsets: tuple[int, ...],
    max_chars: int,
    expected_end: int,
    warning_kind: str | None,
) -> None:
    input_directory = write_normalized_document(
        tmp_path / "input",
        [SyntheticSection(text, scene_break_offsets=scene_offsets)],
    )
    policy = ChunkPolicy(target_chars=30, max_chars=max_chars, overlap_chars=5)

    result = chunk_document(input_directory, tmp_path / "output", policy)

    chunks = read_records(result.output_directory / "chunks.jsonl")
    manifest = json.loads((result.output_directory / "chunk_manifest.json").read_text())
    assert chunks[0]["start_char"] == 0
    assert chunks[0]["end_char"] == expected_end
    if warning_kind is not None:
        assert manifest["warnings"][0]["code"] == "boundary_fallback_used"
        assert manifest["warnings"][0]["context"]["counts"][warning_kind] >= 1
    _assert_bounded_overlap(chunks, policy.overlap_chars)


def test_high_overlap_never_reuses_an_already_covered_end(tmp_path: Path) -> None:
    text = "A" * 14 + ". " + "B" + "\n\n" + "C" * 100
    input_directory = write_normalized_document(
        tmp_path / "input", [SyntheticSection(text)]
    )
    policy = ChunkPolicy(target_chars=30, max_chars=51, overlap_chars=28)

    result = chunk_document(input_directory, tmp_path / "output", policy)

    chunks = read_records(result.output_directory / "chunks.jsonl")
    assert [(chunk["start_char"], chunk["end_char"]) for chunk in chunks[:2]] == [
        (0, 17),
        (16, 67),
    ]
    _assert_bounded_overlap(chunks, policy.overlap_chars)


def test_oversized_source_unit_advances_beyond_the_preceding_end(
    tmp_path: Path,
) -> None:
    text = "A" * 44 + "\n\n" + "B" * 80
    input_directory = write_normalized_document(
        tmp_path / "input", [SyntheticSection(text)]
    )
    policy = ChunkPolicy(target_chars=30, max_chars=50, overlap_chars=5)

    result = chunk_document(input_directory, tmp_path / "output", policy)

    chunks = read_records(result.output_directory / "chunks.jsonl")
    assert [
        (chunk["start_char"], chunk["end_char"], chunk["overlap_left_chars"])
        for chunk in chunks
    ] == [(0, 44, 0), (39, 89, 5), (84, 126, 5)]
    _assert_bounded_overlap(chunks, policy.overlap_chars)


def test_scene_boundary_prevents_cross_scene_overlap(tmp_path: Path) -> None:
    text = "A" * 25 + "\n\n" + "B" * 25 + "\n\n" + "C" * 70
    scene_offset = 52
    input_directory = write_normalized_document(
        tmp_path / "input",
        [SyntheticSection(text, scene_break_offsets=(scene_offset,))],
    )

    result = chunk_document(
        input_directory,
        tmp_path / "output",
        ChunkPolicy(target_chars=45, max_chars=60, overlap_chars=15),
    )

    chunks = read_records(result.output_directory / "chunks.jsonl")
    assert chunks[0]["end_char"] == scene_offset
    assert chunks[1]["start_char"] == scene_offset + 2
    assert chunks[1]["overlap_left_chars"] == 0


def test_overlap_prefers_a_source_unit_start_within_the_cap(tmp_path: Path) -> None:
    text = "A" * 10 + "\n\n" + "B" * 10 + "\n\n" + "C" * 35
    input_directory = write_normalized_document(
        tmp_path / "input", [SyntheticSection(text)]
    )
    policy = ChunkPolicy(target_chars=18, max_chars=25, overlap_chars=14)

    result = chunk_document(input_directory, tmp_path / "output", policy)

    chunks = read_records(result.output_directory / "chunks.jsonl")
    assert chunks[0]["start_char"] == 0
    assert chunks[0]["end_char"] == 22
    assert chunks[1]["start_char"] == 12
    assert chunks[1]["overlap_left_chars"] == 10
    _assert_bounded_overlap(chunks, policy.overlap_chars)


def test_hard_fallback_makes_progress_with_exact_overlap(tmp_path: Path) -> None:
    text = "x" * 45
    input_directory = write_normalized_document(
        tmp_path / "input", [SyntheticSection(text)]
    )
    policy = ChunkPolicy(target_chars=15, max_chars=20, overlap_chars=5)

    result = chunk_document(input_directory, tmp_path / "output", policy)

    chunks = read_records(result.output_directory / "chunks.jsonl")
    assert [
        (chunk["start_char"], chunk["end_char"], chunk["overlap_left_chars"])
        for chunk in chunks
    ] == [(0, 20, 0), (15, 35, 5), (30, 45, 5)]


def test_small_and_multiple_sections_never_cross_boundaries(tmp_path: Path) -> None:
    sections = [
        SyntheticSection("First small synthetic section."),
        SyntheticSection("Second small synthetic section.", title="Second"),
        SyntheticSection("Skipped synthetic notes.", role="notes"),
    ]
    input_directory = write_normalized_document(tmp_path / "input", sections)

    result = chunk_document(
        input_directory,
        tmp_path / "output",
        ChunkPolicy(target_chars=30, max_chars=40, overlap_chars=5),
    )

    chunks = read_records(result.output_directory / "chunks.jsonl")
    assert len(chunks) == 2
    assert [chunk["spine_index"] for chunk in chunks] == [0, 1]
    assert [chunk["section_chunk_index"] for chunk in chunks] == [0, 0]
    assert [chunk["overlap_left_chars"] for chunk in chunks] == [0, 0]
    assert "title" not in chunks[0]
    assert chunks[1]["title"] == "Second"


def test_repeated_and_independent_runs_are_byte_deterministic(tmp_path: Path) -> None:
    text = ("Synthetic paragraph words. " * 8 + "\n\n") * 5
    input_directory = write_normalized_document(
        tmp_path / "input", [SyntheticSection(text)]
    )
    policy = ChunkPolicy(target_chars=80, max_chars=100, overlap_chars=20)

    first = chunk_document(input_directory, tmp_path / "first", policy)
    repeated = chunk_document(input_directory, tmp_path / "first", policy)
    second = chunk_document(input_directory, tmp_path / "second", policy)

    assert repeated == first
    for name in ("chunk_manifest.json", "chunks.jsonl"):
        assert first.output_directory.joinpath(name).read_bytes() == (
            second.output_directory.joinpath(name).read_bytes()
        )


def test_alternate_policies_have_distinct_coexisting_namespaces(
    tmp_path: Path,
) -> None:
    input_directory = write_normalized_document(
        tmp_path / "input", [SyntheticSection("word " * 100)]
    )
    output_root = tmp_path / "output"

    first = chunk_document(
        input_directory,
        output_root,
        ChunkPolicy(target_chars=40, max_chars=50, overlap_chars=10),
    )
    second = chunk_document(
        input_directory,
        output_root,
        ChunkPolicy(target_chars=45, max_chars=55, overlap_chars=10),
    )

    assert first.policy_id != second.policy_id
    assert first.output_directory.is_dir()
    assert second.output_directory.is_dir()
    assert first.output_directory.parent == second.output_directory.parent


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("document_schema", "document schema"),
        ("section_schema", "section schema"),
        ("document_id", "document ID"),
        ("section_document_id", "section document ID"),
        ("section_count", "section count"),
        ("spine_order", "spine indices"),
        ("duplicate_section", "section ID"),
        ("text_hash", "text SHA-256"),
        ("scene_offset", "scene offsets"),
        ("empty_candidate", "retrieval-candidate"),
        ("document_warning_type", "document warnings"),
        ("identifier_type", "document identifier"),
        ("title_consistency", "title and title origin"),
        ("integer_type", "spine_index"),
    ],
)
def test_inconsistent_normalized_records_are_rejected_before_output(
    tmp_path: Path, case: str, message: str
) -> None:
    input_directory = write_normalized_document(
        tmp_path / "input",
        [SyntheticSection("First candidate."), SyntheticSection("Second candidate.")],
    )
    document = json.loads((input_directory / "document.json").read_text())
    sections = read_records(input_directory / "sections.jsonl")

    if case == "document_schema":
        document["schema_version"] = "amis.document.v2"
    elif case == "section_schema":
        sections[0]["schema_version"] = "amis.section.v2"
    elif case == "document_id":
        document["document_id"] = "doc_sha256_" + "0" * 64
    elif case == "section_document_id":
        sections[0]["document_id"] = "doc_sha256_" + "0" * 64
    elif case == "section_count":
        document["section_count"] = 3
    elif case == "spine_order":
        sections[1]["spine_index"] = 0
    elif case == "duplicate_section":
        sections[1]["section_id"] = sections[0]["section_id"]
    elif case == "text_hash":
        sections[0]["text_sha256"] = "0" * 64
    elif case == "scene_offset":
        sections[0]["scene_break_offsets"] = [2, 2]
    elif case == "empty_candidate":
        sections[0]["text"] = ""
        sections[0]["text_sha256"] = hashlib.sha256(b"").hexdigest()
        sections[0]["retrieval_candidate"] = True
    elif case == "document_warning_type":
        document["warnings"] = ["not a warning object"]
    elif case == "identifier_type":
        document["identifiers"] = ["not an identifier object"]
    elif case == "title_consistency":
        sections[0]["title_origin"] = "heading"
    elif case == "integer_type":
        sections[0]["spine_index"] = False

    rewrite_document(input_directory, document)
    rewrite_sections(input_directory, sections)
    with pytest.raises(ChunkingError, match=message):
        chunk_document(input_directory, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("document.json", b"{not-json}\n", "document.json"),
        ("document.json", b'{"schema_version":"a","schema_version":"b"}\n', "JSON"),
        ("sections.jsonl", b"{not-json}\n", "sections.jsonl"),
        ("sections.jsonl", b"[]\n", "JSON object"),
        ("sections.jsonl", b"\xff", "UTF-8"),
    ],
)
def test_malformed_json_inputs_are_rejected(
    tmp_path: Path, filename: str, content: bytes, message: str
) -> None:
    input_directory = write_normalized_document(tmp_path / "input")
    input_directory.joinpath(filename).write_bytes(content)

    with pytest.raises(ChunkingError, match=message):
        chunk_document(input_directory, tmp_path / "output")


def test_document_without_candidates_is_a_hard_failure(tmp_path: Path) -> None:
    input_directory = write_normalized_document(
        tmp_path / "input",
        [SyntheticSection("Synthetic front matter.", role="frontmatter")],
    )

    with pytest.raises(ChunkingError, match="no retrieval-candidate"):
        chunk_document(input_directory, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("relation", ["equal", "ancestor", "descendant"])
def test_output_root_must_be_disjoint_from_input(tmp_path: Path, relation: str) -> None:
    input_directory = write_normalized_document(tmp_path / "normalized" / "document")
    if relation == "equal":
        output_root = input_directory
    elif relation == "ancestor":
        output_root = input_directory.parent
    else:
        output_root = input_directory / "chunks"

    with pytest.raises(ChunkingError, match="disjoint"):
        chunk_document(input_directory, output_root)
    assert not (input_directory / "chunks").exists()


def test_symlinked_document_output_cannot_escape_the_output_root(
    tmp_path: Path,
) -> None:
    input_directory = write_normalized_document(tmp_path / "input")
    document_id = json.loads((input_directory / "document.json").read_text())[
        "document_id"
    ]
    output_root = tmp_path / "output"
    output_root.mkdir()
    (output_root / document_id).symlink_to(input_directory, target_is_directory=True)
    original_entries = {path.name for path in input_directory.iterdir()}

    with pytest.raises(ChunkingError, match="symbolic link"):
        chunk_document(input_directory, output_root)

    assert {path.name for path in input_directory.iterdir()} == original_entries


def test_symlinked_policy_directory_is_never_idempotent(tmp_path: Path) -> None:
    input_directory = write_normalized_document(tmp_path / "input")
    trusted = chunk_document(input_directory, tmp_path / "trusted")
    output_root = tmp_path / "output"
    document_root = output_root / trusted.document_id
    document_root.mkdir(parents=True)
    destination = document_root / trusted.policy_id
    destination.symlink_to(trusted.output_directory, target_is_directory=True)
    original_target = destination.readlink()

    with pytest.raises(ChunkingError, match="conflicting data"):
        chunk_document(input_directory, output_root)

    assert destination.is_symlink()
    assert destination.readlink() == original_target
    assert not any(path.name.startswith(".") for path in document_root.iterdir())


@pytest.mark.parametrize("linked_name", ["chunk_manifest.json", "chunks.jsonl"])
def test_symlinked_final_file_is_never_idempotent(
    tmp_path: Path, linked_name: str
) -> None:
    input_directory = write_normalized_document(tmp_path / "input")
    trusted = chunk_document(input_directory, tmp_path / "trusted")
    output_root = tmp_path / "output"
    destination = output_root / trusted.document_id / trusted.policy_id
    destination.mkdir(parents=True)
    for name in ("chunk_manifest.json", "chunks.jsonl"):
        path = destination / name
        trusted_path = trusted.output_directory / name
        if name == linked_name:
            path.symlink_to(trusted_path)
        else:
            path.write_bytes(trusted_path.read_bytes())
    linked_path = destination / linked_name
    original_target = linked_path.readlink()

    with pytest.raises(ChunkingError, match="conflicting data"):
        chunk_document(input_directory, output_root)

    assert linked_path.is_symlink()
    assert linked_path.readlink() == original_target
    assert not any(path.name.startswith(".") for path in destination.parent.iterdir())


def test_conflicting_existing_output_is_preserved(tmp_path: Path) -> None:
    input_directory = write_normalized_document(tmp_path / "input")
    result = chunk_document(input_directory, tmp_path / "output")
    manifest_path = result.output_directory / "chunk_manifest.json"
    manifest_path.write_text("synthetic conflict\n")

    with pytest.raises(ChunkingError, match="conflicting data"):
        chunk_document(input_directory, tmp_path / "output")

    assert manifest_path.read_text() == "synthetic conflict\n"
    assert not any(
        path.name.startswith(".") for path in result.output_directory.parent.iterdir()
    )


def test_publish_failure_removes_staging_and_final_artifacts(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    input_directory = write_normalized_document(tmp_path / "input")
    output_root = tmp_path / "output"

    def fail_replace(source_path: Path, destination_path: Path) -> None:
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(chunking.os, "replace", fail_replace)
    with pytest.raises(ChunkingError, match="published atomically"):
        chunk_document(input_directory, output_root)

    assert not output_root.exists() or list(output_root.rglob("*")) == []


def test_cli_chunks_one_normalized_document(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    input_directory = write_normalized_document(tmp_path / "input")

    assert (
        main(
            [
                "chunk",
                str(input_directory),
                "--output",
                str(tmp_path / "output"),
                "--target-chars",
                "30",
                "--max-chars",
                "40",
                "--overlap-chars",
                "5",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert (
        "with" in captured.out and "chunks under chunk_policy_sha256_" in captured.out
    )
    assert captured.err == ""


def test_cli_reports_chunking_failure(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    input_directory = write_normalized_document(tmp_path / "input")

    assert (
        main(
            [
                "chunk",
                str(input_directory),
                "--output",
                str(tmp_path / "output"),
                "--target-chars",
                "10",
                "--max-chars",
                "9",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("amis chunk: error:")


def _assert_complete_coverage(
    sections: list[dict[str, object]], chunks: list[dict[str, object]]
) -> None:
    for section in sections:
        covered: set[int] = set()
        for chunk in chunks:
            if chunk["section_id"] == section["section_id"]:
                covered.update(range(chunk["start_char"], chunk["end_char"]))
        assert all(
            character.isspace() or index in covered
            for index, character in enumerate(section["text"])
        )


def _assert_bounded_overlap(
    chunks: list[dict[str, object]], overlap_limit: int
) -> None:
    previous_by_section: dict[str, dict[str, object]] = {}
    for chunk in chunks:
        previous = previous_by_section.get(chunk["section_id"])
        expected = (
            0
            if previous is None
            else max(0, previous["end_char"] - chunk["start_char"])
        )
        assert chunk["overlap_left_chars"] == expected
        assert 0 <= expected <= overlap_limit
        if previous is not None:
            assert chunk["start_char"] > previous["start_char"]
            assert chunk["end_char"] > previous["end_char"]
        previous_by_section[chunk["section_id"]] = chunk
