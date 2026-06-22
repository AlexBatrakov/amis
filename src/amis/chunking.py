"""Deterministic chunking for one normalized AMIS document."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from amis.ingestion import DOCUMENT_SCHEMA_VERSION, SECTION_SCHEMA_VERSION

CHUNK_SCHEMA_VERSION = "amis.chunk.v1"
CHUNK_MANIFEST_SCHEMA_VERSION = "amis.chunk_manifest.v1"
CHUNKER_VERSION = "amis.chunker.v1"
CHUNK_STRATEGY = "paragraph_window_v1"

DEFAULT_TARGET_CHARS = 3000
DEFAULT_MAX_CHARS = 4000
DEFAULT_OVERLAP_CHARS = 400

_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_PARAGRAPH_SEPARATOR_RE = re.compile(r"\n{2,}")
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_PUNCTUATION = frozenset(".!?")
_SENTENCE_CLOSERS = frozenset(
    "\"')]}\N{RIGHT SINGLE QUOTATION MARK}"
    "\N{RIGHT DOUBLE QUOTATION MARK}"
    "\N{RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK}"
)
_ROLES = frozenset(
    {"frontmatter", "body", "navigation", "notes", "backmatter", "unknown"}
)
_TITLE_ORIGINS = frozenset({"heading", "navigation", "none"})


class ChunkingError(Exception):
    """Raised when normalized input cannot be chunked under the contract."""


class _DuplicateKeyError(ValueError):
    pass


@dataclass(frozen=True)
class ChunkPolicy:
    """Versioned character-window policy."""

    target_chars: int = DEFAULT_TARGET_CHARS
    max_chars: int = DEFAULT_MAX_CHARS
    overlap_chars: int = DEFAULT_OVERLAP_CHARS

    def __post_init__(self) -> None:
        values = (self.target_chars, self.max_chars, self.overlap_chars)
        if any(type(value) is not int for value in values):
            raise ChunkingError("chunk policy sizes must be integers")
        if not (self.max_chars >= self.target_chars > self.overlap_chars >= 0):
            raise ChunkingError(
                "chunk policy must satisfy max_chars >= target_chars "
                "> overlap_chars >= 0"
            )

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical policy object."""
        return {
            "candidate_filter": "retrieval_candidate",
            "max_chars": self.max_chars,
            "overlap_chars": self.overlap_chars,
            "strategy": CHUNK_STRATEGY,
            "target_chars": self.target_chars,
        }

    @property
    def policy_id(self) -> str:
        """Return the stable identity of this complete policy."""
        digest = hashlib.sha256(_compact_json(self.as_dict()).encode()).hexdigest()
        return f"chunk_policy_sha256_{digest}"


@dataclass(frozen=True)
class ChunkingResult:
    """Summary of one completed chunking run."""

    document_id: str
    policy_id: str
    chunk_count: int
    output_directory: Path


@dataclass(frozen=True)
class _ValidatedInput:
    document: dict[str, Any]
    sections: list[dict[str, Any]]
    document_sha256: str
    sections_sha256: str


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    overlap_left: int
    end_kind: str


def chunk_document(
    input_document_directory: Path | str,
    output_root: Path | str,
    policy: ChunkPolicy | None = None,
) -> ChunkingResult:
    """Validate and chunk one P003 document directory, then publish atomically."""
    selected_policy = policy or ChunkPolicy()
    input_directory = Path(input_document_directory)
    output_directory = Path(output_root)
    _validate_output_isolation(input_directory, output_directory)
    validated = _load_normalized_input(input_directory)

    policy_id = selected_policy.policy_id
    chunk_records: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    eligible_count = 0

    for section in validated.sections:
        if not section["retrieval_candidate"]:
            continue
        eligible_count += 1
        spans = _chunk_text(
            section["text"], section["scene_break_offsets"], selected_policy
        )
        fallback_counts = Counter(
            span.end_kind
            for span in spans
            if span.end_kind in {"sentence", "whitespace", "hard"}
        )
        if fallback_counts:
            warnings.append(
                {
                    "code": "boundary_fallback_used",
                    "context": {
                        "counts": {
                            name: fallback_counts[name]
                            for name in ("sentence", "whitespace", "hard")
                            if fallback_counts[name]
                        },
                        "section_id": section["section_id"],
                    },
                }
            )
        for section_chunk_index, span in enumerate(spans):
            chunk_records.append(
                _chunk_record(
                    section=section,
                    policy_id=policy_id,
                    span=span,
                    document_chunk_index=len(chunk_records),
                    section_chunk_index=section_chunk_index,
                )
            )

    if eligible_count == 0:
        raise ChunkingError("document has no retrieval-candidate sections")

    chunks_bytes = b"".join(_json_bytes(record) for record in chunk_records)
    manifest = {
        "schema_version": CHUNK_MANIFEST_SCHEMA_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "document_id": validated.document["document_id"],
        "source_sha256": validated.document["source_sha256"],
        "input_document_schema_version": validated.document["schema_version"],
        "input_section_schema_version": SECTION_SCHEMA_VERSION,
        "input_document_sha256": validated.document_sha256,
        "input_sections_sha256": validated.sections_sha256,
        "policy": selected_policy.as_dict(),
        "policy_id": policy_id,
        "total_section_count": len(validated.sections),
        "eligible_section_count": eligible_count,
        "skipped_section_count": len(validated.sections) - eligible_count,
        "chunk_count": len(chunk_records),
        "chunks_sha256": hashlib.sha256(chunks_bytes).hexdigest(),
        "warnings": warnings,
    }
    manifest_bytes = _json_bytes(manifest)
    destination = _publish(
        output_directory,
        validated.document["document_id"],
        policy_id,
        manifest_bytes,
        chunks_bytes,
    )
    return ChunkingResult(
        document_id=validated.document["document_id"],
        policy_id=policy_id,
        chunk_count=len(chunk_records),
        output_directory=destination,
    )


def _load_normalized_input(input_directory: Path) -> _ValidatedInput:
    if not input_directory.is_dir():
        raise ChunkingError("input document directory must exist")
    document_bytes = _read_input_file(input_directory / "document.json")
    sections_bytes = _read_input_file(input_directory / "sections.jsonl")
    document = _parse_json_object(document_bytes, "document.json")
    sections = _parse_json_lines(sections_bytes)
    _validate_document(document)
    _validate_sections(document, sections)
    return _ValidatedInput(
        document=document,
        sections=sections,
        document_sha256=hashlib.sha256(document_bytes).hexdigest(),
        sections_sha256=hashlib.sha256(sections_bytes).hexdigest(),
    )


def _read_input_file(path: Path) -> bytes:
    try:
        if not path.is_file():
            raise ChunkingError(f"input is missing {path.name}")
        return path.read_bytes()
    except ChunkingError:
        raise
    except OSError as error:
        raise ChunkingError(f"input {path.name} could not be read") from error


def _parse_json_object(content: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            content.decode("utf-8"), object_pairs_hook=_object_without_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateKeyError) as error:
        raise ChunkingError(f"input {name} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ChunkingError(f"input {name} must contain a JSON object")
    return value


def _parse_json_lines(content: bytes) -> list[dict[str, Any]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ChunkingError("input sections.jsonl is not valid UTF-8") from error
    if not text:
        return []
    lines = text.splitlines()
    sections: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ChunkingError(f"input sections.jsonl line {line_number} is empty")
        try:
            value = json.loads(line, object_pairs_hook=_object_without_duplicates)
        except (json.JSONDecodeError, _DuplicateKeyError) as error:
            raise ChunkingError(
                f"input sections.jsonl line {line_number} is not valid JSON"
            ) from error
        if not isinstance(value, dict):
            raise ChunkingError(
                f"input sections.jsonl line {line_number} must be a JSON object"
            )
        sections.append(value)
    return sections


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateKeyError(key)
        value[key] = item
    return value


def _validate_document(document: dict[str, Any]) -> None:
    _require_string(document, "schema_version")
    if document["schema_version"] != DOCUMENT_SCHEMA_VERSION:
        raise ChunkingError("unsupported document schema version")
    document_id = _require_string(document, "document_id")
    source_sha256 = _require_hash(document, "source_sha256")
    if document_id != f"doc_sha256_{source_sha256}":
        raise ChunkingError("document ID does not match source SHA-256")
    _require_string(document, "source_basename")
    _require_nonnegative_integer(document, "source_size")
    _require_string(document, "epub_version")
    _require_string(document, "package_path", nonempty=True)
    _require_nonnegative_integer(document, "section_count")
    _require_string(document, "loader_version")
    identifiers = _require_list(document, "identifiers")
    for identifier in identifiers:
        _validate_identifier(identifier)
    _validate_warning_objects(_require_list(document, "warnings"), "document")
    package_identifier = document.get("package_unique_identifier")
    if package_identifier is not None:
        _validate_identifier(package_identifier)
    for key in ("title", "language", "publisher", "date"):
        if key in document and not isinstance(document[key], str):
            raise ChunkingError(f"document field {key} has invalid type")
    if "creators" in document and not _is_string_list(document["creators"]):
        raise ChunkingError("document field creators has invalid type")


def _validate_sections(
    document: dict[str, Any], sections: list[dict[str, Any]]
) -> None:
    if document["section_count"] != len(sections):
        raise ChunkingError("document section count does not match sections.jsonl")
    seen_ids: set[str] = set()
    for expected_index, section in enumerate(sections):
        _validate_section(document, section, expected_index)
        section_id = section["section_id"]
        if section_id in seen_ids:
            raise ChunkingError("sections contain a duplicate section ID")
        seen_ids.add(section_id)


def _validate_section(
    document: dict[str, Any], section: dict[str, Any], expected_index: int
) -> None:
    schema_version = _require_string(section, "schema_version")
    if schema_version != SECTION_SCHEMA_VERSION:
        raise ChunkingError("unsupported section schema version")
    section_id = _require_string(section, "section_id")
    document_id = _require_string(section, "document_id")
    if document_id != document["document_id"]:
        raise ChunkingError("section document ID does not match document")
    spine_index = _require_nonnegative_integer(section, "spine_index")
    if spine_index != expected_index:
        raise ChunkingError("section spine indices must be contiguous and ordered")
    _require_string(section, "source_href", nonempty=True)
    source_path = _require_string(section, "source_path", nonempty=True)
    _require_hash(section, "source_content_sha256")
    linear = _require_boolean(section, "linear")
    role = _require_string(section, "role")
    if role not in _ROLES:
        raise ChunkingError("section role is invalid")
    retrieval_candidate = _require_boolean(section, "retrieval_candidate")
    title_origin = _require_string(section, "title_origin")
    if title_origin not in _TITLE_ORIGINS:
        raise ChunkingError("section title origin is invalid")
    if "title" in section and (
        not isinstance(section["title"], str) or not section["title"]
    ):
        raise ChunkingError("section title has invalid type")
    if (title_origin == "none") != ("title" not in section):
        raise ChunkingError("section title and title origin are inconsistent")
    text = _require_string(section, "text")
    text_sha256 = _require_hash(section, "text_sha256")
    if text_sha256 != hashlib.sha256(text.encode()).hexdigest():
        raise ChunkingError("section text SHA-256 does not match text")
    expected_candidate = bool(text) and linear and role == "body"
    if retrieval_candidate != expected_candidate:
        raise ChunkingError("section retrieval-candidate invariant is inconsistent")

    scene_offsets = _require_list(section, "scene_break_offsets")
    previous_offset = -1
    for offset in scene_offsets:
        if type(offset) is not int or not 0 <= offset <= len(text):
            raise ChunkingError("section scene offsets must be in range")
        if offset <= previous_offset:
            raise ChunkingError("section scene offsets must be ordered and unique")
        previous_offset = offset

    _require_list(section, "note_targets")
    _require_list(section, "warnings")
    if not _is_string_list(section["note_targets"]):
        raise ChunkingError("section note targets have invalid type")
    if not _is_string_list(section["warnings"]):
        raise ChunkingError("section warnings have invalid type")
    for key in ("paragraph_count", "heading_count", "image_count", "link_count"):
        _require_nonnegative_integer(section, key)

    seed = (
        f"amis:section:v1\0{document_id}\0{document['package_path']}\0"
        f"{source_path}\0{spine_index}"
    )
    expected_id = "sec_sha256_" + hashlib.sha256(seed.encode()).hexdigest()
    if section_id != expected_id:
        raise ChunkingError("section ID does not match the accepted derivation")


def _chunk_text(
    text: str, scene_offsets: list[int], policy: ChunkPolicy
) -> list[_Span]:
    paragraph_ends, paragraph_starts = _paragraph_boundaries(text)
    sentence_ends, sentence_starts = _sentence_boundaries(text)
    whitespace_ends, whitespace_starts = _whitespace_boundaries(text)
    scene_starts = {_advance_whitespace(text, offset) for offset in scene_offsets}
    spans: list[_Span] = []
    start = 0
    covered_end = 0
    overlap_left = 0

    while start < len(text):
        minimum_end = max(start, covered_end)
        remaining = len(text) - start
        if remaining <= policy.max_chars:
            end = len(text)
            end_kind = "terminal"
        else:
            hard_end = start + policy.max_chars
            target_end = start + policy.target_chars
            end, end_kind = _select_end(
                target_end,
                hard_end,
                minimum_end,
                scene_offsets,
                paragraph_ends,
                sentence_ends,
                whitespace_ends,
            )
        if end <= minimum_end or end - start > policy.max_chars:
            raise ChunkingError("chunk boundary selection did not make progress")
        spans.append(
            _Span(
                start=start,
                end=end,
                overlap_left=overlap_left,
                end_kind=end_kind,
            )
        )
        covered_end = end
        if end == len(text):
            break

        overlap_floor = max(end - policy.overlap_chars, start + 1)
        prior_scenes = [offset for offset in scene_offsets if start < offset <= end]
        if prior_scenes:
            overlap_floor = max(overlap_floor, prior_scenes[-1])
        next_start = _select_overlap_start(
            text,
            overlap_floor,
            end,
            scene_starts,
            paragraph_starts,
            sentence_starts,
            whitespace_starts,
        )
        if next_start <= start:
            raise ChunkingError("chunk overlap selection did not make progress")
        overlap_left = max(0, end - next_start)
        if overlap_left > policy.overlap_chars:
            raise ChunkingError("chunk overlap exceeds the configured limit")
        start = next_start
    return spans


def _select_end(
    target_end: int,
    hard_end: int,
    minimum_end: int,
    scene_ends: list[int],
    paragraph_ends: list[int],
    sentence_ends: list[int],
    whitespace_ends: list[int],
) -> tuple[int, str]:
    groups = (
        ("scene", scene_ends),
        ("paragraph", paragraph_ends),
        ("sentence", sentence_ends),
        ("whitespace", whitespace_ends),
    )
    for kind, positions in groups:
        candidates = [
            position for position in positions if minimum_end < position <= hard_end
        ]
        if candidates:
            selected = min(
                candidates,
                key=lambda position: (
                    abs(position - target_end),
                    position > target_end,
                    position,
                ),
            )
            return selected, kind
    return hard_end, "hard"


def _select_overlap_start(
    text: str,
    overlap_floor: int,
    previous_end: int,
    scene_starts: set[int],
    paragraph_starts: list[int],
    sentence_starts: list[int],
    whitespace_starts: list[int],
) -> int:
    if overlap_floor >= previous_end:
        return _advance_whitespace(text, overlap_floor)
    for positions in (
        scene_starts,
        paragraph_starts,
        sentence_starts,
        whitespace_starts,
    ):
        candidates = [
            position
            for position in positions
            if overlap_floor <= position < previous_end
        ]
        if candidates:
            return min(candidates)
    return _advance_whitespace(text, overlap_floor)


def _paragraph_boundaries(text: str) -> tuple[list[int], list[int]]:
    matches = list(_PARAGRAPH_SEPARATOR_RE.finditer(text))
    return [match.start() for match in matches], [match.end() for match in matches]


def _sentence_boundaries(text: str) -> tuple[list[int], list[int]]:
    ends: list[int] = []
    starts: list[int] = []
    for index, character in enumerate(text):
        if character not in _SENTENCE_PUNCTUATION:
            continue
        boundary = index + 1
        while boundary < len(text) and text[boundary] in _SENTENCE_CLOSERS:
            boundary += 1
        if boundary < len(text) and not text[boundary].isspace():
            continue
        ends.append(boundary)
        start = _advance_whitespace(text, boundary)
        if start < len(text):
            starts.append(start)
    return ends, starts


def _whitespace_boundaries(text: str) -> tuple[list[int], list[int]]:
    matches = list(_WHITESPACE_RE.finditer(text))
    return [match.start() for match in matches], [match.end() for match in matches]


def _advance_whitespace(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _chunk_record(
    *,
    section: dict[str, Any],
    policy_id: str,
    span: _Span,
    document_chunk_index: int,
    section_chunk_index: int,
) -> dict[str, Any]:
    text = section["text"][span.start : span.end]
    text_sha256 = hashlib.sha256(text.encode()).hexdigest()
    seed = (
        f"amis:chunk:v1\0{section['section_id']}\0{policy_id}\0"
        f"{span.start}\0{span.end}\0{text_sha256}"
    )
    record = {
        "schema_version": CHUNK_SCHEMA_VERSION,
        "chunk_id": "chunk_sha256_" + hashlib.sha256(seed.encode()).hexdigest(),
        "policy_id": policy_id,
        "document_id": section["document_id"],
        "section_id": section["section_id"],
        "document_chunk_index": document_chunk_index,
        "section_chunk_index": section_chunk_index,
        "spine_index": section["spine_index"],
        "start_char": span.start,
        "end_char": span.end,
        "overlap_left_chars": span.overlap_left,
        "text": text,
        "text_sha256": text_sha256,
        "section_text_sha256": section["text_sha256"],
        "source_path": section["source_path"],
        "source_content_sha256": section["source_content_sha256"],
        "role": section["role"],
    }
    if "title" in section:
        record["title"] = section["title"]
    return record


def _validate_output_isolation(input_directory: Path, output_root: Path) -> None:
    try:
        resolved_input = input_directory.resolve(strict=True)
    except OSError as error:
        raise ChunkingError("input document directory must exist") from error
    if not resolved_input.is_dir():
        raise ChunkingError("input document directory must exist")
    try:
        resolved_output = output_root.resolve(strict=False)
    except OSError as error:
        raise ChunkingError("output root path could not be resolved") from error
    if (
        resolved_input == resolved_output
        or resolved_input in resolved_output.parents
        or resolved_output in resolved_input.parents
    ):
        raise ChunkingError("output root must be disjoint from the input directory")


def _publish(
    output_root: Path,
    document_id: str,
    policy_id: str,
    manifest_bytes: bytes,
    chunks_bytes: bytes,
) -> Path:
    output_root_created = not output_root.exists()
    document_root = output_root / document_id
    document_root_created = not document_root.exists()
    stage_path: Path | None = None
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        if not output_root.is_dir():
            raise ChunkingError("output root must be a directory")
        if document_root.is_symlink():
            raise ChunkingError("document output root must not be a symbolic link")
        document_root.mkdir(exist_ok=True)
        if not document_root.is_dir():
            raise ChunkingError("document output root must be a directory")

        destination = document_root / policy_id
        stage_path = Path(
            tempfile.mkdtemp(prefix=f".{policy_id}.tmp-", dir=document_root)
        )
        _write_synced(stage_path / "chunk_manifest.json", manifest_bytes)
        _write_synced(stage_path / "chunks.jsonl", chunks_bytes)
        _sync_directory(stage_path)

        if destination.is_symlink() or destination.exists():
            if _matches_existing(destination, manifest_bytes, chunks_bytes):
                return destination
            raise ChunkingError("chunk output already contains conflicting data")
        os.replace(stage_path, destination)
        stage_path = None
        _sync_directory(document_root)
        return destination
    except ChunkingError:
        raise
    except OSError as error:
        raise ChunkingError("chunk output could not be published atomically") from error
    finally:
        if stage_path is not None and stage_path.exists():
            shutil.rmtree(stage_path, ignore_errors=True)
        if document_root_created:
            _remove_empty_directory(document_root)
        if output_root_created:
            _remove_empty_directory(output_root)


def _write_synced(path: Path, content: bytes) -> None:
    with path.open("xb") as output_file:
        output_file.write(content)
        output_file.flush()
        os.fsync(output_file.fileno())


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_empty_directory(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass


def _matches_existing(
    destination: Path, manifest_bytes: bytes, chunks_bytes: bytes
) -> bool:
    if destination.is_symlink() or not destination.is_dir():
        return False
    try:
        entries = {entry.name for entry in destination.iterdir()}
        manifest_path = destination / "chunk_manifest.json"
        chunks_path = destination / "chunks.jsonl"
        return (
            entries == {"chunk_manifest.json", "chunks.jsonl"}
            and not manifest_path.is_symlink()
            and manifest_path.is_file()
            and not chunks_path.is_symlink()
            and chunks_path.is_file()
            and manifest_path.read_bytes() == manifest_bytes
            and chunks_path.read_bytes() == chunks_bytes
        )
    except OSError:
        return False


def _require_string(value: dict[str, Any], key: str, *, nonempty: bool = False) -> str:
    item = value.get(key)
    if not isinstance(item, str) or (nonempty and not item):
        raise ChunkingError(f"field {key} must be a string")
    return item


def _require_hash(value: dict[str, Any], key: str) -> str:
    item = _require_string(value, key)
    if _HASH_RE.fullmatch(item) is None:
        raise ChunkingError(f"field {key} must be a lowercase SHA-256")
    return item


def _require_nonnegative_integer(value: dict[str, Any], key: str) -> int:
    item = value.get(key)
    if type(item) is not int or item < 0:
        raise ChunkingError(f"field {key} must be a non-negative integer")
    return item


def _require_boolean(value: dict[str, Any], key: str) -> bool:
    item = value.get(key)
    if type(item) is not bool:
        raise ChunkingError(f"field {key} must be a boolean")
    return item


def _require_list(value: dict[str, Any], key: str) -> list[Any]:
    item = value.get(key)
    if not isinstance(item, list):
        raise ChunkingError(f"field {key} must be a list")
    return item


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _validate_identifier(value: Any) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("value"), str):
        raise ChunkingError("document identifier has invalid type")
    for key in ("element_id", "scheme"):
        if value.get(key) is not None and not isinstance(value[key], str):
            raise ChunkingError("document identifier has invalid type")


def _validate_warning_objects(values: list[Any], owner: str) -> None:
    for value in values:
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("code"), str)
            or not isinstance(value.get("context"), dict)
        ):
            raise ChunkingError(f"{owner} warnings have invalid type")


def _compact_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_bytes(value: dict[str, Any]) -> bytes:
    return f"{_compact_json(value)}\n".encode()
