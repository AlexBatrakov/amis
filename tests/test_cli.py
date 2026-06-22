"""Smoke tests for the AMIS package and CLI."""

import subprocess
import sys
from pathlib import Path

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


def test_semantic_help_does_not_import_optional_runtime() -> None:
    script = """
import sys
from amis.cli import main
try:
    main(['index', 'build', '--help'])
except SystemExit as error:
    assert error.code == 0
assert 'torch' not in sys.modules
assert 'sentence_transformers' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_index_build_reports_missing_local_model(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "index",
                "build",
                str(tmp_path / "chunks"),
                "--output",
                str(tmp_path / "indexes"),
                "--cache-root",
                str(tmp_path / "cache"),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("amis index: error:")
