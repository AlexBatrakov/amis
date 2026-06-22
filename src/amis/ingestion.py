"""Deterministic one-file EPUB 2 ingestion."""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import shutil
import tempfile
import unicodedata
import zipfile
import zlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

DOCUMENT_SCHEMA_VERSION = "amis.document.v1"
SECTION_SCHEMA_VERSION = "amis.section.v1"
LOADER_VERSION = "amis.epub.v1"
MAX_MEMBER_SIZE = 64 * 1024 * 1024
MAX_ARCHIVE_SIZE = 256 * 1024 * 1024

_BLOCK_TAGS = {"blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "li", "p"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_IGNORED_TEXT_TAGS = {"head", "img", "script", "style", "svg", "title"}
_BR_MARKER = "\N{SYMBOL FOR NEWLINE}"
_SPACE_RE = re.compile(r"\s+")
_HINT_SEPARATOR_RE = re.compile(r"[^a-z0-9]+")


class IngestionError(Exception):
    """Raised when a source cannot be ingested without violating the contract."""


@dataclass(frozen=True)
class IngestionResult:
    """Summary of one completed ingestion."""

    document_id: str
    section_count: int
    output_directory: Path


@dataclass(frozen=True)
class _ManifestItem:
    item_id: str
    href: str
    media_type: str
    source_path: str
    properties: tuple[str, ...]


@dataclass
class _Link:
    target: str
    note_reference: bool


@dataclass
class _ParsedSection:
    record: dict[str, Any]
    fragment_identifiers: set[str]
    note_element_ids: set[str]
    links: list[_Link]
    warning_events: list[dict[str, Any]] = field(default_factory=list)

    def warn(self, code: str, **context: Any) -> None:
        if code not in self.record["warnings"]:
            self.record["warnings"].append(code)
        self.warning_events.append({"code": code, "context": context})


@dataclass(frozen=True)
class _LoadedEpub:
    document: dict[str, Any]
    sections: list[dict[str, Any]]


def ingest_epub(source: Path | str, output_root: Path | str) -> IngestionResult:
    """Load one EPUB and atomically publish its normalized records."""
    loaded = _load_epub(Path(source))
    document_bytes = _json_bytes(loaded.document)
    section_bytes = b"".join(_json_bytes(section) for section in loaded.sections)
    destination = _publish(
        Path(output_root),
        loaded.document["document_id"],
        document_bytes,
        section_bytes,
    )
    return IngestionResult(
        document_id=loaded.document["document_id"],
        section_count=len(loaded.sections),
        output_directory=destination,
    )


def _load_epub(source: Path) -> _LoadedEpub:
    if not source.is_file():
        raise IngestionError("source must be an existing local file")

    source_sha256, source_size = _hash_file(source)
    try:
        archive = zipfile.ZipFile(source)
    except (OSError, zipfile.BadZipFile) as error:
        raise IngestionError("source is not a valid ZIP archive") from error

    with archive:
        _validate_archive(archive)
        package_path = _package_path(archive)
        package_root = _parse_member_xml(archive, package_path, "package document")
        package_version = package_root.get("version", "").strip()
        if package_version != "2.0":
            raise IngestionError("unsupported EPUB package version")

        document_warnings: list[dict[str, Any]] = []
        metadata = _read_metadata(package_root, document_warnings)
        manifest = _read_manifest(
            archive, package_root, package_path, document_warnings
        )
        spine, ncx_id = _read_spine(archive, package_root, manifest)
        guide_roles = _read_guide(package_root, package_path)
        navigation_titles, navigation_roles = _read_navigation(
            archive,
            manifest,
            ncx_id,
            {item.source_path for item, _ in spine},
            document_warnings,
        )

        document_id = f"doc_sha256_{source_sha256}"
        parsed_sections: list[_ParsedSection] = []
        for spine_index, (item, linear) in enumerate(spine):
            content = archive.read(item.source_path)
            root = _parse_xml_bytes(content, "spine content")
            body = _single_body(root)
            role = _classify_role(
                item,
                root,
                body,
                guide_roles.get(item.source_path),
                navigation_roles.get(item.source_path),
            )
            section = _parse_section(
                content=content,
                root=root,
                body=body,
                item=item,
                role=role,
                linear=linear,
                spine_index=spine_index,
                document_id=document_id,
                package_path=package_path,
                navigation_title=navigation_titles.get(item.source_path),
            )
            parsed_sections.append(section)

        fragment_index, content_resource_paths = _build_fragment_index(
            archive, manifest, parsed_sections
        )
        _resolve_links(
            parsed_sections,
            set(archive.namelist()),
            fragment_index,
            content_resource_paths,
        )
        for section in parsed_sections:
            for warning in section.warning_events:
                context = {
                    "source_path": section.record["source_path"],
                    "spine_index": section.record["spine_index"],
                    **warning["context"],
                }
                document_warnings.append({"code": warning["code"], "context": context})

    document: dict[str, Any] = {
        "schema_version": DOCUMENT_SCHEMA_VERSION,
        "document_id": document_id,
        "source_basename": source.name,
        "source_sha256": source_sha256,
        "source_size": source_size,
        "epub_version": package_version,
        "package_path": package_path,
        "package_unique_identifier": metadata["package_unique_identifier"],
        "identifiers": metadata["identifiers"],
        "section_count": len(parsed_sections),
        "warnings": document_warnings,
        "loader_version": LOADER_VERSION,
    }
    for key in ("title", "creators", "language", "publisher", "date"):
        if metadata[key]:
            document[key] = metadata[key]

    return _LoadedEpub(
        document=document,
        sections=[section.record for section in parsed_sections],
    )


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as source_file:
            while chunk := source_file.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError as error:
        raise IngestionError("source file could not be read") from error
    return digest.hexdigest(), size


def _validate_archive(archive: zipfile.ZipFile) -> None:
    members = archive.infolist()
    if not members:
        raise IngestionError("EPUB archive is empty")

    names = [member.filename for member in members]
    if len(names) != len(set(names)):
        raise IngestionError("EPUB archive contains duplicate member names")
    for name in names:
        if not _safe_member_name(name):
            raise IngestionError("EPUB archive contains an unsafe member path")

    total_size = 0
    for member in members:
        if member.flag_bits & 0x1:
            raise IngestionError("encrypted EPUB content is unsupported")
        if member.file_size > MAX_MEMBER_SIZE:
            raise IngestionError("EPUB member exceeds the size limit")
        total_size += member.file_size
    if total_size > MAX_ARCHIVE_SIZE:
        raise IngestionError("EPUB archive exceeds the uncompressed size limit")

    try:
        bad_member = archive.testzip()
    except (OSError, RuntimeError, zipfile.BadZipFile, zlib.error) as error:
        raise IngestionError("EPUB archive failed its CRC check") from error
    if bad_member is not None:
        raise IngestionError("EPUB archive failed its CRC check")

    mimetype = members[0]
    if mimetype.filename != "mimetype":
        raise IngestionError("EPUB mimetype must be the first archive member")
    if mimetype.compress_type != zipfile.ZIP_STORED:
        raise IngestionError("EPUB mimetype must be stored without compression")
    if archive.read(mimetype) != b"application/epub+zip":
        raise IngestionError("EPUB mimetype value is invalid")

    lower_names = {name.casefold() for name in names}
    if {"meta-inf/encryption.xml", "meta-inf/rights.xml"} & lower_names:
        raise IngestionError("encrypted or DRM-protected EPUB content is unsupported")


def _safe_member_name(name: str) -> bool:
    if not name or "\\" in name or "\x00" in name:
        return False
    path = PurePosixPath(name)
    first_part = path.parts[0] if path.parts else ""
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and not first_part.endswith(":")
    )


def _package_path(archive: zipfile.ZipFile) -> str:
    container_path = "META-INF/container.xml"
    if container_path not in archive.namelist():
        raise IngestionError("EPUB container document is missing")
    root = _parse_member_xml(archive, container_path, "container document")
    rootfiles = [
        element for element in root.iter() if _local_name(element.tag) == "rootfile"
    ]
    if len(rootfiles) != 1:
        raise IngestionError("EPUB container must declare exactly one package rootfile")
    rootfile = rootfiles[0]
    if rootfile.get("media-type") != "application/oebps-package+xml":
        raise IngestionError("EPUB package rootfile media type is unsupported")
    full_path = rootfile.get("full-path", "")
    package_path, _ = _resolve_href("", full_path)
    if package_path not in archive.namelist():
        raise IngestionError("EPUB package document is missing")
    return package_path


def _parse_member_xml(
    archive: zipfile.ZipFile, path: str, description: str
) -> ElementTree.Element:
    try:
        content = archive.read(path)
    except KeyError as error:
        raise IngestionError(f"{description} is missing") from error
    return _parse_xml_bytes(content, description)


def _parse_xml_bytes(content: bytes, description: str) -> ElementTree.Element:
    upper_content = content.upper()
    if b"<!DOCTYPE" in upper_content or b"<!ENTITY" in upper_content:
        raise IngestionError(f"{description} contains unsupported declarations")
    try:
        return ElementTree.fromstring(content)
    except ElementTree.ParseError as error:
        raise IngestionError(f"{description} is malformed XML") from error


def _read_metadata(
    package_root: ElementTree.Element, warnings: list[dict[str, Any]]
) -> dict[str, Any]:
    metadata_elements = _direct_children(package_root, "metadata")
    if len(metadata_elements) != 1:
        raise IngestionError("EPUB package metadata is missing or malformed")
    metadata_root = metadata_elements[0]

    values: dict[str, list[str]] = {
        name: [] for name in ("title", "creator", "language", "publisher", "date")
    }
    identifiers: list[dict[str, Any]] = []
    selected_identifier_id = package_root.get("unique-identifier")
    selected_identifier: dict[str, Any] | None = None

    for element in metadata_root:
        name = _local_name(element.tag)
        value = _normalize_metadata("".join(element.itertext()))
        if not value:
            continue
        if name in values:
            values[name].append(value)
        if name == "identifier":
            identifier = {
                "value": value,
                "element_id": element.get("id"),
                "scheme": _attribute_by_local_name(element, "scheme"),
            }
            identifiers.append(identifier)
            if selected_identifier_id and element.get("id") == selected_identifier_id:
                selected_identifier = identifier.copy()

    for field_name in ("title", "creator", "language", "publisher", "date"):
        if not values[field_name]:
            warnings.append(
                {
                    "code": "metadata_missing_optional",
                    "context": {"field": field_name},
                }
            )
    for field_name in ("title", "language", "publisher", "date"):
        if len(values[field_name]) > 1:
            warnings.append(
                {
                    "code": "metadata_multiple_values",
                    "context": {
                        "count": len(values[field_name]),
                        "field": field_name,
                    },
                }
            )
    if selected_identifier is None:
        warnings.append(
            {
                "code": "metadata_missing_optional",
                "context": {"field": "package_unique_identifier"},
            }
        )

    return {
        "title": _first(values["title"]),
        "creators": values["creator"],
        "language": _first(values["language"]),
        "publisher": _first(values["publisher"]),
        "date": _first(values["date"]),
        "identifiers": identifiers,
        "package_unique_identifier": selected_identifier,
    }


def _read_manifest(
    archive: zipfile.ZipFile,
    package_root: ElementTree.Element,
    package_path: str,
    warnings: list[dict[str, Any]],
) -> dict[str, _ManifestItem]:
    manifests = _direct_children(package_root, "manifest")
    if len(manifests) != 1:
        raise IngestionError("EPUB package manifest is missing or malformed")
    item_elements = _direct_children(manifests[0], "item")
    if not item_elements:
        raise IngestionError("EPUB package manifest is empty")

    manifest: dict[str, _ManifestItem] = {}
    for element in item_elements:
        item_id = element.get("id", "").strip()
        href = element.get("href", "").strip()
        media_type = element.get("media-type", "").strip()
        if not item_id or not href or not media_type or item_id in manifest:
            raise IngestionError("EPUB package manifest is malformed")
        source_path, _ = _resolve_href(package_path, href)
        item = _ManifestItem(
            item_id=item_id,
            href=href,
            media_type=media_type,
            source_path=source_path,
            properties=tuple(element.get("properties", "").split()),
        )
        manifest[item_id] = item
        if source_path not in archive.namelist() and media_type not in {
            "application/xhtml+xml",
            "text/html",
        }:
            warnings.append(
                {
                    "code": "unsupported_manifest_resource_ignored",
                    "context": {"manifest_id": item_id, "reason": "missing"},
                }
            )
        elif not _known_manifest_media_type(media_type):
            warnings.append(
                {
                    "code": "unsupported_manifest_resource_ignored",
                    "context": {
                        "manifest_id": item_id,
                        "media_type": media_type,
                    },
                }
            )
    return manifest


def _known_manifest_media_type(media_type: str) -> bool:
    return (
        media_type
        in {
            "application/font-sfnt",
            "application/vnd.ms-opentype",
            "application/x-dtbncx+xml",
            "application/xhtml+xml",
            "text/css",
            "text/html",
        }
        or media_type.startswith("audio/")
        or media_type.startswith("font/")
        or media_type.startswith("image/")
        or media_type.startswith("video/")
    )


def _read_spine(
    archive: zipfile.ZipFile,
    package_root: ElementTree.Element,
    manifest: dict[str, _ManifestItem],
) -> tuple[list[tuple[_ManifestItem, bool]], str | None]:
    spines = _direct_children(package_root, "spine")
    if len(spines) != 1:
        raise IngestionError("EPUB package spine is missing or malformed")
    itemrefs = _direct_children(spines[0], "itemref")
    if not itemrefs:
        raise IngestionError("EPUB package spine is empty")

    spine: list[tuple[_ManifestItem, bool]] = []
    for itemref in itemrefs:
        idref = itemref.get("idref", "").strip()
        if not idref or idref not in manifest:
            raise IngestionError("EPUB spine contains an unresolved idref")
        item = manifest[idref]
        if item.media_type not in {"application/xhtml+xml", "text/html"}:
            raise IngestionError("EPUB spine resource media type is unsupported")
        if item.source_path not in archive.namelist():
            raise IngestionError("EPUB spine resource is missing")
        linear = itemref.get("linear", "yes").casefold() != "no"
        spine.append((item, linear))
    return spine, spines[0].get("toc")


def _read_guide(package_root: ElementTree.Element, package_path: str) -> dict[str, str]:
    guide_roles: dict[str, str] = {}
    guides = _direct_children(package_root, "guide")
    if len(guides) > 1:
        raise IngestionError("EPUB package guide is malformed")
    if not guides:
        return guide_roles
    for reference in _direct_children(guides[0], "reference"):
        href = reference.get("href", "")
        role = _guide_role(reference.get("type", ""))
        if not href or role is None:
            continue
        try:
            target_path, _ = _resolve_href(package_path, href)
        except IngestionError:
            continue
        guide_roles.setdefault(target_path, role)
    return guide_roles


def _guide_role(guide_type: str) -> str | None:
    normalized = _normalize_hint(guide_type)
    if normalized in {"cover", "dedication", "frontmatter", "title_page", "titlepage"}:
        return "frontmatter"
    if normalized in {"contents", "navigation", "toc"}:
        return "navigation"
    if normalized in {"footnotes", "notes"}:
        return "notes"
    if normalized in {"copyright", "rear", "backmatter"}:
        return "backmatter"
    if normalized in {"body", "start", "text"}:
        return "body"
    return None


def _read_navigation(
    archive: zipfile.ZipFile,
    manifest: dict[str, _ManifestItem],
    ncx_id: str | None,
    spine_paths: set[str],
    warnings: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    ncx_item = manifest.get(ncx_id) if ncx_id else None
    if ncx_item is None:
        ncx_item = next(
            (
                item
                for item in manifest.values()
                if item.media_type == "application/x-dtbncx+xml"
            ),
            None,
        )
    if ncx_item is None or ncx_item.source_path not in archive.namelist():
        warnings.append({"code": "navigation_missing", "context": {}})
        return {}, {}

    try:
        root = _parse_member_xml(archive, ncx_item.source_path, "NCX navigation")
    except IngestionError:
        warnings.append(
            {"code": "navigation_missing", "context": {"reason": "malformed"}}
        )
        return {}, {}

    titles: dict[str, str] = {}
    roles: dict[str, str] = {}
    nav_points = [
        element for element in root.iter() if _local_name(element.tag) == "navPoint"
    ]
    for nav_point in nav_points:
        content_element = next(
            (element for element in nav_point if _local_name(element.tag) == "content"),
            None,
        )
        if content_element is None or not content_element.get("src"):
            continue
        try:
            target_path, _ = _resolve_href(
                ncx_item.source_path, content_element.get("src", "")
            )
        except IngestionError:
            continue
        label = _navigation_label(nav_point)
        if label:
            titles.setdefault(target_path, label)
        role = _role_from_hints([nav_point.get("class", ""), nav_point.get("id", "")])
        if role and role != "body":
            roles.setdefault(target_path, role)

    if not titles and not nav_points:
        warnings.append({"code": "navigation_empty", "context": {}})
    covered_paths = set(titles) & spine_paths
    if covered_paths != spine_paths:
        warnings.append(
            {
                "code": "navigation_incomplete",
                "context": {"missing_count": len(spine_paths - covered_paths)},
            }
        )
    return titles, roles


def _navigation_label(nav_point: ElementTree.Element) -> str | None:
    for element in nav_point.iter():
        if _local_name(element.tag) == "navLabel":
            value = _normalize_metadata("".join(element.itertext()))
            if value:
                return value
    return None


def _classify_role(
    item: _ManifestItem,
    root: ElementTree.Element,
    body: ElementTree.Element,
    guide_role: str | None,
    navigation_role: str | None,
) -> str:
    manifest_role = _role_from_hints(item.properties)
    if manifest_role:
        return manifest_role
    if guide_role:
        return guide_role
    if navigation_role:
        return navigation_role

    hints = [item.source_path, item.item_id]
    for element in (root, body, *list(body)[:3]):
        hints.extend(
            [
                element.get("id", ""),
                element.get("class", ""),
                _attribute_by_local_name(element, "type") or "",
            ]
        )
    return _role_from_hints(hints) or "unknown"


def _role_from_hints(hints: Iterable[str]) -> str | None:
    normalized = {_normalize_hint(hint) for hint in hints if hint}
    role_tokens = (
        ("navigation", {"contents", "nav", "navigation", "toc"}),
        ("notes", {"endnote", "endnotes", "footnote", "footnotes", "note", "notes"}),
        ("backmatter", {"about_author", "about_publisher", "backmatter", "copyright"}),
        (
            "frontmatter",
            {
                "advertising",
                "cover",
                "dedication",
                "frontmatter",
                "praise",
                "prelim",
                "prelims",
                "review",
                "title",
                "title_page",
                "titlepage",
            },
        ),
        ("body", {"body", "chapter", "main_text"}),
    )
    for role, tokens in role_tokens:
        if any(
            _hint_contains(value, token) for value in normalized for token in tokens
        ):
            return role
    return None


def _hint_contains(value: str, token: str) -> bool:
    return (
        value == token
        or value.startswith(f"{token}_")
        or value.endswith(f"_{token}")
        or f"_{token}_" in value
    )


def _normalize_hint(value: str) -> str:
    return _HINT_SEPARATOR_RE.sub("_", value.casefold()).strip("_")


def _parse_section(
    *,
    content: bytes,
    root: ElementTree.Element,
    body: ElementTree.Element,
    item: _ManifestItem,
    role: str,
    linear: bool,
    spine_index: int,
    document_id: str,
    package_path: str,
    navigation_title: str | None,
) -> _ParsedSection:
    section_seed = (
        f"amis:section:v1\0{document_id}\0{package_path}\0"
        f"{item.source_path}\0{spine_index}"
    )
    section_id = "sec_sha256_" + hashlib.sha256(section_seed.encode()).hexdigest()
    text, scene_offsets, ambiguous_blocks = _extract_text(body)

    heading_title = _first_heading(body)
    title = heading_title or navigation_title
    title_origin = (
        "heading" if heading_title else "navigation" if navigation_title else "none"
    )
    record: dict[str, Any] = {
        "schema_version": SECTION_SCHEMA_VERSION,
        "section_id": section_id,
        "document_id": document_id,
        "spine_index": spine_index,
        "source_href": item.href,
        "source_path": item.source_path,
        "source_content_sha256": hashlib.sha256(content).hexdigest(),
        "linear": linear,
        "role": role,
        "retrieval_candidate": bool(text) and linear and role == "body",
        "title_origin": title_origin,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "scene_break_offsets": scene_offsets,
        "note_targets": [],
        "paragraph_count": sum(
            1 for element in body.iter() if _local_name(element.tag) == "p"
        ),
        "heading_count": sum(
            1 for element in body.iter() if _local_name(element.tag) in _HEADING_TAGS
        ),
        "image_count": sum(
            1 for element in body.iter() if _local_name(element.tag) == "img"
        ),
        "link_count": sum(
            1 for element in body.iter() if _local_name(element.tag) == "a"
        ),
        "warnings": [],
    }
    if title:
        record["title"] = title

    section = _ParsedSection(
        record=record,
        fragment_identifiers=_fragment_identifiers(root),
        note_element_ids=_note_element_ids(body),
        links=_read_links(body, item.source_path),
    )
    if role == "unknown":
        section.warn("section_role_unknown")
    if not linear:
        section.warn("nonlinear_spine_item")
    if not text:
        section.warn("empty_section")
    for block_index in ambiguous_blocks:
        section.warn("scene_break_ambiguous", block_index=block_index)
    for link in section.links:
        if link.target.startswith("external:"):
            section.warn("external_link_ignored", target_scheme=link.target[9:])
    return section


def _single_body(root: ElementTree.Element) -> ElementTree.Element:
    bodies = [element for element in root.iter() if _local_name(element.tag) == "body"]
    if len(bodies) != 1:
        raise IngestionError("spine content must contain exactly one body")
    return bodies[0]


def _extract_text(body: ElementTree.Element) -> tuple[str, list[int], list[int]]:
    blocks = _collect_blocks(body)
    segments: list[str | None] = []
    ambiguous: list[int] = []
    for block_index, block in enumerate(blocks):
        value = _normalize_inline(_flatten_inline(block))
        if _is_scene_break(block, value):
            segments.append(None)
            continue
        if _is_ambiguous_scene_break(block, value):
            ambiguous.append(block_index)
        if value:
            segments.append(value)

    output = ""
    offsets: list[int] = []
    pending_break = False
    for segment in segments:
        if segment is None:
            pending_break = True
            continue
        if output:
            if pending_break and (not offsets or offsets[-1] != len(output)):
                offsets.append(len(output))
            output += "\n\n" + segment
        else:
            if pending_break and (not offsets or offsets[-1] != 0):
                offsets.append(0)
            output = segment
        pending_break = False
    if pending_break and (not offsets or offsets[-1] != len(output)):
        offsets.append(len(output))
    return output, offsets, ambiguous


def _collect_blocks(container: ElementTree.Element) -> list[ElementTree.Element]:
    blocks: list[ElementTree.Element] = []
    for child in container:
        name = _local_name(child.tag)
        if name in _IGNORED_TEXT_TAGS:
            continue
        if name in _BLOCK_TAGS:
            blocks.append(child)
            continue
        nested_blocks = _collect_blocks(child)
        if nested_blocks:
            blocks.extend(nested_blocks)
        elif _normalize_inline(_flatten_inline(child)):
            blocks.append(child)
    if not blocks and _normalize_inline(_flatten_inline(container)):
        blocks.append(container)
    return blocks


def _flatten_inline(element: ElementTree.Element) -> str:
    pieces: list[str] = [element.text or ""]
    for child in element:
        name = _local_name(child.tag)
        if name == "br":
            pieces.append(_BR_MARKER)
        elif name not in _IGNORED_TEXT_TAGS:
            if name in _BLOCK_TAGS:
                pieces.append(_BR_MARKER)
            pieces.append(_flatten_inline(child))
            if name in _BLOCK_TAGS:
                pieces.append(_BR_MARKER)
        pieces.append(child.tail or "")
    return "".join(pieces)


def _normalize_inline(value: str) -> str:
    value = value.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for piece in value.split(_BR_MARKER):
        normalized = _SPACE_RE.sub(" ", piece).strip()
        if normalized:
            lines.append(normalized)
        elif lines and lines[-1] != "":
            lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines).strip()


def _is_scene_break(element: ElementTree.Element, text: str) -> bool:
    classes = element.get("class", "").casefold()
    if "scene" in classes or "break" in classes:
        return True
    compact = "".join(text.split())
    if (
        compact
        and len(compact) <= 12
        and all(
            unicodedata.category(character)[0] in {"P", "S"} for character in compact
        )
    ):
        return True
    images = [child for child in element.iter() if _local_name(child.tag) == "img"]
    if images and not text:
        hints = [element.get("class", ""), element.get("id", "")]
        for image in images:
            hints.extend(
                [image.get("class", ""), image.get("id", ""), image.get("src", "")]
            )
        return any(
            token in hint.casefold()
            for hint in hints
            for token in ("break", "ornament", "scene", "separator")
        )
    return False


def _is_ambiguous_scene_break(element: ElementTree.Element, text: str) -> bool:
    if not text or len(text) > 40:
        return False
    presentation = f"{element.get('class', '')} {element.get('style', '')}".casefold()
    return "center" in presentation or "centre" in presentation


def _first_heading(body: ElementTree.Element) -> str | None:
    for element in body.iter():
        if _local_name(element.tag) in _HEADING_TAGS:
            value = _normalize_inline(_flatten_inline(element))
            if value:
                return value
    return None


def _fragment_identifiers(root: ElementTree.Element) -> set[str]:
    identifiers: set[str] = set()
    for element in root.iter():
        identifiers.update(_element_fragment_identifiers(element))
    return identifiers


def _element_fragment_identifiers(element: ElementTree.Element) -> set[str]:
    identifiers = {
        element_id
        for element_id in (_attribute_by_local_name(element, "id"),)
        if element_id
    }
    if _local_name(element.tag) == "a":
        anchor_name = _attribute_by_local_name(element, "name")
        if anchor_name:
            identifiers.add(anchor_name)
    return identifiers


def _note_element_ids(body: ElementTree.Element) -> set[str]:
    note_ids: set[str] = set()
    for element in body.iter():
        if _has_note_hint(element):
            note_ids.update(_element_fragment_identifiers(element))
    return note_ids


def _has_note_hint(element: ElementTree.Element) -> bool:
    hints = " ".join(
        (
            element.get("class", ""),
            element.get("id", ""),
            element.get("name", ""),
            _attribute_by_local_name(element, "type") or "",
        )
    ).casefold()
    return "note" in hints


def _read_links(body: ElementTree.Element, source_path: str) -> list[_Link]:
    parent_map = {child: parent for parent in body.iter() for child in parent}
    links: list[_Link] = []
    for element in body.iter():
        if _local_name(element.tag) != "a" or not element.get("href"):
            continue
        href = element.get("href", "")
        try:
            split = urlsplit(href)
        except ValueError:
            links.append(_Link(target="", note_reference=False))
            continue
        note_reference = _has_note_hint(element) or any(
            _local_name(ancestor.tag) == "sup" or _has_note_hint(ancestor)
            for ancestor in _ancestors(element, parent_map)
        )
        if split.scheme or split.netloc:
            links.append(
                _Link(
                    target=f"external:{split.scheme.casefold()}", note_reference=False
                )
            )
            continue
        try:
            target_path, fragment = _resolve_href(source_path, href)
        except IngestionError:
            target_path, fragment = "", ""
        target = target_path + (f"#{fragment}" if fragment else "")
        links.append(_Link(target=target, note_reference=note_reference))
    return links


def _ancestors(
    element: ElementTree.Element,
    parent_map: dict[ElementTree.Element, ElementTree.Element],
) -> Iterable[ElementTree.Element]:
    current = parent_map.get(element)
    while current is not None:
        yield current
        current = parent_map.get(current)


def _build_fragment_index(
    archive: zipfile.ZipFile,
    manifest: dict[str, _ManifestItem],
    sections: list[_ParsedSection],
) -> tuple[dict[str, set[str]], set[str]]:
    fragment_index: dict[str, set[str]] = {}
    for section in sections:
        source_path = section.record["source_path"]
        fragment_index.setdefault(source_path, set()).update(
            section.fragment_identifiers
        )

    content_resource_paths = {
        item.source_path
        for item in manifest.values()
        if item.media_type in {"application/xhtml+xml", "text/html"}
    }
    archive_members = set(archive.namelist())
    for section in sections:
        for link in section.links:
            if link.target.startswith("external:"):
                continue
            target_path, separator, _ = link.target.partition("#")
            if (
                not separator
                or target_path in fragment_index
                or target_path not in content_resource_paths
                or target_path not in archive_members
            ):
                continue
            try:
                root = _parse_xml_bytes(
                    archive.read(target_path), "linked content resource"
                )
            except IngestionError:
                fragment_index[target_path] = set()
            else:
                fragment_index[target_path] = _fragment_identifiers(root)
    return fragment_index, content_resource_paths


def _resolve_links(
    sections: list[_ParsedSection],
    archive_members: set[str],
    fragment_index: dict[str, set[str]],
    content_resource_paths: set[str],
) -> None:
    sections_by_path = {section.record["source_path"]: section for section in sections}
    for section in sections:
        note_targets: list[str] = []
        for link in section.links:
            if link.target.startswith("external:"):
                continue
            if not link.target:
                section.warn("internal_link_target_missing", target="")
                continue
            target_path, separator, fragment = link.target.partition("#")
            target_section = sections_by_path.get(target_path)
            target_exists = target_path in archive_members
            fragment_exists = (
                not fragment
                or target_path not in content_resource_paths
                or fragment in fragment_index.get(target_path, set())
            )
            if not target_exists or not fragment_exists:
                section.warn("internal_link_target_missing", target=link.target)
            target_is_note = bool(
                target_section
                and (
                    target_section.record["role"] == "notes"
                    or fragment in target_section.note_element_ids
                )
            )
            source_is_note = section.record["role"] == "notes"
            if link.note_reference or target_is_note or source_is_note:
                canonical_target = target_path + (f"#{fragment}" if separator else "")
                if canonical_target not in note_targets:
                    note_targets.append(canonical_target)
        section.record["note_targets"] = note_targets


def _resolve_href(base_path: str, href: str) -> tuple[str, str]:
    try:
        split = urlsplit(href)
    except ValueError as error:
        raise IngestionError("archive resource path is malformed") from error
    if split.scheme or split.netloc:
        raise IngestionError("external archive resource references are unsupported")
    decoded_path = unquote(split.path)
    if "\\" in decoded_path or "\x00" in decoded_path:
        raise IngestionError("archive resource path is unsafe")
    if decoded_path:
        base_directory = posixpath.dirname(base_path)
        resolved = posixpath.normpath(posixpath.join(base_directory, decoded_path))
    else:
        resolved = base_path
    path = PurePosixPath(resolved)
    first_part = path.parts[0] if path.parts else ""
    if (
        not resolved
        or resolved == "."
        or path.is_absolute()
        or ".." in path.parts
        or first_part.endswith(":")
    ):
        raise IngestionError("archive resource path is unsafe")
    return resolved, unquote(split.fragment)


def _publish(
    output_root: Path,
    document_id: str,
    document_bytes: bytes,
    section_bytes: bytes,
) -> Path:
    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise IngestionError("output root could not be created") from error
    if not output_root.is_dir():
        raise IngestionError("output root must be a directory")

    destination = output_root / document_id
    stage_path: Path | None = None
    try:
        stage_path = Path(
            tempfile.mkdtemp(prefix=f".{document_id}.tmp-", dir=output_root)
        )
        _write_synced(stage_path / "document.json", document_bytes)
        _write_synced(stage_path / "sections.jsonl", section_bytes)
        _sync_directory(stage_path)

        if destination.exists():
            if _matches_existing(destination, document_bytes, section_bytes):
                return destination
            raise IngestionError("output directory already contains conflicting data")
        os.replace(stage_path, destination)
        stage_path = None
        _sync_directory(output_root)
        return destination
    except IngestionError:
        raise
    except OSError as error:
        raise IngestionError(
            "normalized output could not be published atomically"
        ) from error
    finally:
        if stage_path is not None and stage_path.exists():
            shutil.rmtree(stage_path, ignore_errors=True)


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


def _matches_existing(
    destination: Path, document_bytes: bytes, section_bytes: bytes
) -> bool:
    if not destination.is_dir():
        return False
    try:
        entries = {entry.name for entry in destination.iterdir()}
        return entries == {"document.json", "sections.jsonl"} and (
            destination.joinpath("document.json").read_bytes() == document_bytes
            and destination.joinpath("sections.jsonl").read_bytes() == section_bytes
        )
    except OSError:
        return False


def _json_bytes(value: dict[str, Any]) -> bytes:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{serialized}\n".encode()


def _normalize_metadata(value: str) -> str:
    return _SPACE_RE.sub(" ", value.replace("\u00a0", " ")).strip()


def _local_name(name: str) -> str:
    return name.rsplit("}", 1)[-1]


def _attribute_by_local_name(element: ElementTree.Element, name: str) -> str | None:
    return next(
        (value for key, value in element.attrib.items() if _local_name(key) == name),
        None,
    )


def _direct_children(
    element: ElementTree.Element, name: str
) -> list[ElementTree.Element]:
    return [child for child in element if _local_name(child.tag) == name]


def _first(values: list[str]) -> str | None:
    return values[0] if values else None
