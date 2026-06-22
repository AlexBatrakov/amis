"""Synthetic normalized AMIS records for public chunking tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SyntheticSection:
    """One synthetic section used to build accepted P003-shaped input."""

    text: str
    role: str = "body"
    linear: bool = True
    scene_break_offsets: tuple[int, ...] = ()
    title: str | None = None


def write_normalized_document(
    directory: Path,
    sections: list[SyntheticSection] | None = None,
) -> Path:
    """Write deterministic, entirely synthetic P003 records."""
    source_sha256 = hashlib.sha256(b"public synthetic normalized source").hexdigest()
    document_id = f"doc_sha256_{source_sha256}"
    package_path = "OPS/package.opf"
    selected_sections = sections or [
        SyntheticSection(
            "Brass birds crossed a painted morning.\n\n"
            "Paper leaves answered with a harmless refrain.",
            title="Synthetic Opening",
        ),
        SyntheticSection("A public fixture notice.", role="backmatter", linear=False),
    ]
    records = [
        _section_record(
            value,
            document_id=document_id,
            package_path=package_path,
            spine_index=index,
        )
        for index, value in enumerate(selected_sections)
    ]
    document = {
        "schema_version": "amis.document.v1",
        "document_id": document_id,
        "source_basename": "synthetic.epub",
        "source_sha256": source_sha256,
        "source_size": 1234,
        "epub_version": "2.0",
        "package_path": package_path,
        "package_unique_identifier": {
            "element_id": "book-id",
            "scheme": "UUID",
            "value": "urn:uuid:00000000-0000-4000-8000-000000000004",
        },
        "identifiers": [],
        "section_count": len(records),
        "warnings": [],
        "loader_version": "amis.epub.v1",
        "title": "A Wholly Synthetic Volume",
        "creators": ["Fixture Author"],
        "language": "en",
    }
    directory.mkdir(parents=True)
    directory.joinpath("document.json").write_bytes(_json_bytes(document))
    directory.joinpath("sections.jsonl").write_bytes(
        b"".join(_json_bytes(record) for record in records)
    )
    return directory


def read_records(path: Path) -> list[dict[str, object]]:
    """Read JSONL test output."""
    return [json.loads(line) for line in path.read_text().splitlines()]


def rewrite_document(directory: Path, document: dict[str, object]) -> None:
    """Replace the synthetic document record canonically."""
    directory.joinpath("document.json").write_bytes(_json_bytes(document))


def rewrite_sections(directory: Path, sections: list[dict[str, object]]) -> None:
    """Replace synthetic section records canonically."""
    directory.joinpath("sections.jsonl").write_bytes(
        b"".join(_json_bytes(section) for section in sections)
    )


def _section_record(
    value: SyntheticSection,
    *,
    document_id: str,
    package_path: str,
    spine_index: int,
) -> dict[str, object]:
    source_path = f"OPS/Text/section-{spine_index}.xhtml"
    seed = (
        f"amis:section:v1\0{document_id}\0{package_path}\0{source_path}\0{spine_index}"
    )
    record: dict[str, object] = {
        "schema_version": "amis.section.v1",
        "section_id": "sec_sha256_" + hashlib.sha256(seed.encode()).hexdigest(),
        "document_id": document_id,
        "spine_index": spine_index,
        "source_href": f"Text/section-{spine_index}.xhtml",
        "source_path": source_path,
        "source_content_sha256": hashlib.sha256(
            f"synthetic source {spine_index}".encode()
        ).hexdigest(),
        "linear": value.linear,
        "role": value.role,
        "retrieval_candidate": bool(value.text)
        and value.linear
        and value.role == "body",
        "title_origin": "heading" if value.title else "none",
        "text": value.text,
        "text_sha256": hashlib.sha256(value.text.encode()).hexdigest(),
        "scene_break_offsets": list(value.scene_break_offsets),
        "note_targets": [],
        "paragraph_count": value.text.count("\n\n") + bool(value.text),
        "heading_count": int(value.title is not None),
        "image_count": 0,
        "link_count": 0,
        "warnings": [],
    }
    if value.title:
        record["title"] = value.title
    return record


def _json_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
