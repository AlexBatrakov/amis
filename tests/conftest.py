"""Suite-wide safety fixtures."""

from __future__ import annotations

import socket

import pytest
from pytest import MonkeyPatch


@pytest.fixture(autouse=True)
def deny_network(monkeypatch: MonkeyPatch) -> None:
    """Make accidental network access fail in every in-process public test."""

    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("public tests must not access the network")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
