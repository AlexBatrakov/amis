"""Smoke tests for the AMIS package and CLI."""

import subprocess
import sys

from pytest import CaptureFixture

import amis
from amis.cli import READY_MESSAGE, main


def test_package_is_importable() -> None:
    assert amis.__doc__ == "Ankh-Morpork Intelligence System."


def test_cli_main(capsys: CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert capsys.readouterr().out.strip() == READY_MESSAGE


def test_module_entry_point() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "amis"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == READY_MESSAGE
