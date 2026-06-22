"""Generated, synthetic EPUB 2 fixtures for public tests."""

from __future__ import annotations

import struct
import warnings
import zipfile
from pathlib import Path

MIMETYPE = b"application/epub+zip"
PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDAT\x08\xd7c\xf8\xcf\xc0\xf0\x1f\x00"
    b"\x05\x00\x01\xff\x89\x99=\x1d\x00\x00\x00\x00IEND"
    b"\xaeB`\x82"
)


def build_epub(path: Path, anomaly: str | None = None) -> Path:
    """Build a tiny legal EPUB, optionally with one structural anomaly."""
    if anomaly == "non_zip":
        path.write_bytes(b"This is a synthetic non-ZIP fixture.")
        return path

    container = _container_xml(anomaly)
    package = _package_xml(anomaly)
    copyright_path = (
        "OPS/Text/mystery.xhtml"
        if anomaly == "unknown_role"
        else "OPS/Text/copyright.xhtml"
    )
    if anomaly == "navigation_empty":
        navigation = _empty_ncx_xml()
    elif anomaly == "navigation_malformed":
        navigation = b"<ncx>"
    else:
        navigation = _ncx_xml()
    note_target = _note_target(anomaly)
    members: list[tuple[str, bytes, int]] = [
        ("META-INF/container.xml", container, zipfile.ZIP_DEFLATED),
        ("OPS/package.opf", package, zipfile.ZIP_DEFLATED),
        ("OPS/Navigation/toc.ncx", navigation, zipfile.ZIP_DEFLATED),
        ("OPS/Text/front.xhtml", _front_xhtml(), zipfile.ZIP_DEFLATED),
        (
            "OPS/Text/part one.xhtml",
            _body_one_xhtml(note_target),
            zipfile.ZIP_DEFLATED,
        ),
        ("OPS/Text/body-two.xhtml", _body_two_xhtml(), zipfile.ZIP_DEFLATED),
        (
            "OPS/Text/notes.xhtml",
            _notes_xhtml(legacy_name=anomaly == "legacy_name_target"),
            zipfile.ZIP_DEFLATED,
        ),
        (
            copyright_path,
            _copyright_xhtml(generic=anomaly == "unknown_role"),
            zipfile.ZIP_DEFLATED,
        ),
        (
            "OPS/Styles/book.css",
            b".italic { font-style: italic; }\n",
            zipfile.ZIP_DEFLATED,
        ),
        ("OPS/Images/separator.png", PNG_1X1, zipfile.ZIP_DEFLATED),
    ]
    if anomaly in {
        "nonspine_id_target",
        "nonspine_missing_fragment",
        "nonspine_name_target",
    }:
        members.append(
            (
                "OPS/Text/auxiliary.xhtml",
                _auxiliary_xhtml(anomaly),
                zipfile.ZIP_DEFLATED,
            )
        )

    if anomaly == "missing_container":
        members = [
            member for member in members if member[0] != "META-INF/container.xml"
        ]
    elif anomaly == "malformed_container":
        members[0] = (members[0][0], b"<container>", members[0][2])
    elif anomaly == "missing_package":
        members = [member for member in members if member[0] != "OPS/package.opf"]
    elif anomaly == "malformed_package":
        members[1] = (members[1][0], b"<package>", members[1][2])
    elif anomaly == "malformed_content":
        members[5] = (members[5][0], b"<html><body>", members[5][2])
    elif anomaly == "doctype_content":
        members[5] = (
            members[5][0],
            _body_two_xhtml().replace(
                b'<html xmlns="http://www.w3.org/1999/xhtml">',
                b'<!DOCTYPE html><html xmlns="http://www.w3.org/1999/xhtml">',
            ),
            members[5][2],
        )
    elif anomaly == "missing_body":
        members[5] = (
            members[5][0],
            b'<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><head/></html>',
            members[5][2],
        )
    elif anomaly == "missing_spine_resource":
        members = [
            member for member in members if member[0] != "OPS/Text/body-two.xhtml"
        ]
    elif anomaly == "encryption":
        members.append(
            ("META-INF/encryption.xml", b"<encryption/>", zipfile.ZIP_DEFLATED)
        )
    elif anomaly == "unsafe_member":
        members.append(("../escape.xhtml", b"synthetic", zipfile.ZIP_DEFLATED))
    elif anomaly == "oversize_member":
        members.append(("OPS/large.bin", b"x" * 128, zipfile.ZIP_STORED))
    elif anomaly == "bad_crc":
        members.append(("OPS/crc.bin", b"synthetic-crc-data", zipfile.ZIP_STORED))
    elif anomaly == "navigation_missing":
        members = [
            member for member in members if member[0] != "OPS/Navigation/toc.ncx"
        ]

    mimetype_compression = (
        zipfile.ZIP_DEFLATED if anomaly == "compressed_mimetype" else zipfile.ZIP_STORED
    )
    mimetype_value = b"invalid/type" if anomaly == "invalid_mimetype" else MIMETYPE

    with zipfile.ZipFile(path, "w") as archive:
        if anomaly == "misplaced_mimetype":
            _write_member(archive, "placeholder", b"synthetic", zipfile.ZIP_STORED)
        _write_member(archive, "mimetype", mimetype_value, mimetype_compression)
        for name, content, compression in members:
            _write_member(archive, name, content, compression)
        if anomaly == "duplicate_member":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                _write_member(
                    archive,
                    "OPS/Text/front.xhtml",
                    _front_xhtml(),
                    zipfile.ZIP_DEFLATED,
                )

    if anomaly == "bad_crc":
        _corrupt_stored_member(path, "OPS/crc.bin")
    return path


def _write_member(
    archive: zipfile.ZipFile, name: str, content: bytes, compression: int
) -> None:
    member = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
    member.compress_type = compression
    archive.writestr(member, content)


def _container_xml(anomaly: str | None) -> bytes:
    media_type = (
        "application/xml"
        if anomaly == "unsupported_rootfile_media"
        else "application/oebps-package+xml"
    )
    rootfile = (
        ""
        if anomaly == "zero_rootfiles"
        else f'<rootfile full-path="OPS/package.opf" media-type="{media_type}"/>'
    )
    extra = ""
    if anomaly == "multiple_rootfiles":
        extra = (
            '<rootfile full-path="OPS/other.opf" '
            'media-type="application/oebps-package+xml"/>'
        )
    return f"""<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    {rootfile}
    {extra}
  </rootfiles>
</container>
""".encode()


def _package_xml(anomaly: str | None) -> bytes:
    version = "3.0" if anomaly == "unsupported_version" else "2.0"
    manifest_open = "" if anomaly == "missing_manifest" else "<manifest>"
    manifest_close = "" if anomaly == "missing_manifest" else "</manifest>"
    manifest_items = "" if anomaly == "missing_manifest" else _manifest_items(anomaly)
    if anomaly == "missing_spine":
        spine = ""
    elif anomaly == "empty_spine":
        spine = '<spine toc="ncx"/>'
    else:
        spine = _spine(anomaly)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="{version}" unique-identifier="book-id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:opf="http://www.idpf.org/2007/opf">
    <dc:identifier id="book-id" opf:scheme="UUID">urn:uuid:00000000-0000-4000-8000-000000000003</dc:identifier>
    <dc:title>The Clockwork Orchard</dc:title>
    <dc:creator>Example Author</dc:creator>
    <dc:language>en</dc:language>
    <dc:publisher>Public Fixture Press</dc:publisher>
    <dc:date>2026-01-01</dc:date>
  </metadata>
  {manifest_open}
    {manifest_items}
  {manifest_close}
  {spine}
  <guide>
    <reference type="title-page" title="Front" href="Text/front.xhtml"/>
    <reference type="text" title="Start" href="Text/part%20one.xhtml"/>
  </guide>
</package>
""".encode()


def _manifest_items(anomaly: str | None) -> str:
    if anomaly == "empty_manifest":
        return ""
    body_media = (
        "text/css" if anomaly == "unsupported_spine_media" else "application/xhtml+xml"
    )
    copyright_id = "mystery" if anomaly == "unknown_role" else "copyright"
    copyright_href = (
        "Text/mystery.xhtml" if anomaly == "unknown_role" else "Text/copyright.xhtml"
    )
    auxiliary_item = ""
    if anomaly in {
        "nonspine_id_target",
        "nonspine_missing_fragment",
        "nonspine_name_target",
    }:
        auxiliary_item = (
            '<item id="auxiliary" href="Text/auxiliary.xhtml" '
            'media-type="application/xhtml+xml"/>'
        )
    return f"""
    <item id="ncx" href="Navigation/toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="front" href="Text/front.xhtml" media-type="application/xhtml+xml"/>
    <item id="body-one" href="Text/part%20one.xhtml" media-type="{body_media}"/>
    <item id="body-two" href="Text/body-two.xhtml" media-type="application/xhtml+xml"/>
    <item id="notes" href="Text/notes.xhtml" media-type="application/xhtml+xml"/>
    <item id="{copyright_id}" href="{copyright_href}" media-type="text/html"/>
    <item id="css" href="Styles/book.css" media-type="text/css"/>
    <item id="separator" href="Images/separator.png" media-type="image/png"/>
    {auxiliary_item}
    """


def _spine(anomaly: str | None) -> str:
    first_idref = "missing" if anomaly == "unresolved_spine" else "front"
    copyright_idref = "mystery" if anomaly == "unknown_role" else "copyright"
    return f"""<spine toc="ncx">
    <itemref idref="{first_idref}"/>
    <itemref idref="body-one"/>
    <itemref idref="body-two"/>
    <itemref idref="notes"/>
    <itemref idref="{copyright_idref}" linear="no"/>
  </spine>"""


def _ncx_xml() -> bytes:
    return b"""<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head><meta name="dtb:uid" content="urn:uuid:00000000-0000-4000-8000-000000000003"/></head>
  <docTitle><text>The Clockwork Orchard</text></docTitle>
  <navMap>
    <navPoint id="part-one" playOrder="1">
      <navLabel><text>First Synthetic Part</text></navLabel>
      <content src="../Text/part%20one.xhtml"/>
    </navPoint>
    <navPoint id="body-two" playOrder="2">
      <navLabel><text>Second Synthetic Part</text></navLabel>
      <content src="../Text/body-two.xhtml"/>
    </navPoint>
    <navPoint id="copyright-point" class="chapter" playOrder="3">
      <navLabel><text>Synthetic Notice</text></navLabel>
      <content src="../Text/copyright.xhtml"/>
    </navPoint>
  </navMap>
</ncx>
"""


def _empty_ncx_xml() -> bytes:
    return b"""<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <navMap/>
</ncx>
"""


def _front_xhtml() -> bytes:
    return b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Front</title></head>
<body class="title-page"><h1>The Clockwork Orchard</h1><p>Entirely synthetic fixture text.</p>
<p class="empty"> </p><p class="separator"><img src="../Images/separator.png" alt="ignored words"/></p></body></html>
"""


def _note_target(anomaly: str | None) -> str:
    if anomaly == "missing_internal_link":
        return "missing.xhtml#note-1"
    if anomaly == "nonspine_missing_fragment":
        return "auxiliary.xhtml#missing-note"
    if anomaly in {"nonspine_id_target", "nonspine_name_target"}:
        return "auxiliary.xhtml#aux-note"
    return "notes.xhtml#note-1"


def _body_one_xhtml(note_target: str = "notes.xhtml#note-1") -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Part</title></head>
<body class="main-text"><h1>First Synthetic Part</h1>
<p id="opening">Clockwork <em>ravens</em> crossed <span class="italic">paper skies</span>.</p>
<div><p>A nested sentence<br/>continues&#160;on another line.</p></div>
<p class="scene-break">* * *</p>
<p>After the marker, <a id="ref-1" class="noteref" href="{note_target}"><sup>1</sup></a> gears hummed. <a href="https://example.invalid/">Outside</a></p>
</body></html>
""".encode()


def _body_two_xhtml() -> bytes:
    return b"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Second</title></head>
<body class="body"><p id="middle">Copper leaves answered in invented whispers.</p>
<p class="centered">A Small Inscription</p></body></html>
"""


def _notes_xhtml(*, legacy_name: bool = False) -> bytes:
    if legacy_name:
        note_block = (
            '<p class="footnote"><a name="note-1" class="footnote" '
            'href="part%20one.xhtml#ref-1">1</a> '
            "No quotation comes from any published book.</p>"
        )
    else:
        note_block = (
            '<p id="note-1" class="footnote">'
            '<a href="part%20one.xhtml#ref-1">1</a> '
            "No quotation comes from any published book.</p>"
        )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Notes</title></head>
<body class="notes"><h2>Synthetic Notes</h2>{note_block}</body></html>
""".encode()


def _auxiliary_xhtml(anomaly: str) -> bytes:
    if anomaly == "nonspine_name_target":
        anchor = '<a name="aux-note" class="footnote">1</a>'
    elif anomaly == "nonspine_missing_fragment":
        anchor = '<span id="different-note" class="footnote">1</span>'
    else:
        anchor = '<span id="aux-note" class="footnote">1</span>'
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Auxiliary</title></head>
<body><p>{anchor} Synthetic auxiliary note text.</p></body></html>
""".encode()


def _copyright_xhtml(*, generic: bool = False) -> bytes:
    body_class = "ordinary" if generic else "copyright"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Notice</title></head>
<body class="{body_class}"><p>This synthetic fixture is dedicated to public testing.</p></body></html>
""".encode()


def _corrupt_stored_member(path: Path, member_name: str) -> None:
    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo(member_name)
        header_offset = member.header_offset
    with path.open("r+b") as fixture:
        fixture.seek(header_offset)
        header = fixture.read(30)
        filename_length, extra_length = struct.unpack_from("<HH", header, 26)
        data_offset = header_offset + 30 + filename_length + extra_length
        fixture.seek(data_offset)
        original = fixture.read(1)
        fixture.seek(data_offset)
        fixture.write(bytes([original[0] ^ 0xFF]))
