"""Shared lexical path checks for security-sensitive local artifacts."""

from __future__ import annotations

from pathlib import Path


class UnsafePathError(ValueError):
    """Raised when an existing path component is a symbolic link."""


def reject_symlink_components(path: Path | str) -> None:
    """Reject symbolic links in every existing component of ``path``.

    The walk is lexical so resolving or normalizing the path cannot hide an
    alias, including one followed by ``..``.
    """
    selected = Path(path)
    absolute = selected if selected.is_absolute() else Path.cwd() / selected
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        if part == ".":
            continue
        if part == "..":
            current = current.parent
            continue
        current /= part
        try:
            if current.is_symlink():
                raise UnsafePathError("path must not traverse a symbolic link")
        except OSError as error:
            raise UnsafePathError("path components could not be inspected") from error
