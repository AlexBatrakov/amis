"""Smoke tests for the AMIS package and CLI."""

import subprocess
import sys
from pathlib import Path

from pytest import CaptureFixture, MonkeyPatch

import amis
import amis.cli as cli
import amis.embeddings as embeddings
from amis.cli import READY_MESSAGE, main
from amis.model_cache import VerifiedModel
from amis.model_spec import EMBEDDING_GEMMA
from amis.semantic_index import build_semantic_index
from tests.semantic_factory import (
    FakeEmbedder,
    write_chunk_policy,
    write_chunk_policy_from_texts,
)


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


def test_search_help_does_not_import_optional_runtime() -> None:
    script = """
import sys
from amis.cli import main
try:
    main(['search', '--help'])
except SystemExit as error:
    assert error.code == 0
assert 'torch' not in sys.modules
assert 'sentence_transformers' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_lexical_search_help_does_not_import_optional_runtime() -> None:
    script = """
import sys
from amis.cli import main
try:
    main(['lexical-search', '--help'])
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


def test_search_cli_prints_synthetic_citations(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: MonkeyPatch
) -> None:
    chunks = write_chunk_policy(tmp_path / "input")
    result = build_semantic_index(chunks, tmp_path / "indexes", FakeEmbedder())
    fake_query_embedder = FakeEmbedder()
    snapshot = tmp_path / "cache" / "snapshot"

    monkeypatch.setattr(
        cli,
        "verify_model_snapshot",
        lambda selected, *, cache_root: VerifiedModel(snapshot, EMBEDDING_GEMMA),
    )
    monkeypatch.setattr(
        embeddings,
        "SentenceTransformerEmbedder",
        lambda selected: fake_query_embedder,
    )

    assert (
        main(
            [
                "search",
                "synthetic query",
                "--index",
                str(result.output_directory),
                "--chunks",
                str(chunks),
                "--top-k",
                "1",
                "--excerpt-chars",
                "32",
                "--cache-root",
                str(tmp_path / "cache"),
                "--model-snapshot",
                str(snapshot),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Rank 1 | score 1.000000" in captured.out
    assert "chunk_id: chunk_sha256_" in captured.out
    assert "section_id: sec_sha256_" in captured.out
    assert "excerpt: Synthetic semantic" in captured.out


def test_search_cli_empty_query_fails_before_model_verification(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: MonkeyPatch
) -> None:
    def verify_should_not_run(*args: object, **kwargs: object) -> VerifiedModel:
        raise AssertionError("model verification must not run")

    monkeypatch.setattr(cli, "verify_model_snapshot", verify_should_not_run)

    assert (
        main(
            [
                "search",
                "   ",
                "--index",
                str(tmp_path / "index"),
                "--chunks",
                str(tmp_path / "chunks"),
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("amis search: error:")
    assert "empty" in captured.err


def test_lexical_search_cli_prints_synthetic_citations(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    chunks = write_chunk_policy_from_texts(
        tmp_path / "input",
        [
            "Needle alpha alpha beta in a synthetic notice.",
            "Gamma delta in another synthetic notice.",
        ],
    )

    assert (
        main(
            [
                "lexical-search",
                "alpha beta",
                "--chunks",
                str(chunks),
                "--top-k",
                "1",
                "--excerpt-chars",
                "32",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Rank 1 | lexical_score " in captured.out
    assert "chunk_id: chunk_sha256_" in captured.out
    assert "section_id: sec_sha256_" in captured.out
    assert "excerpt: Needle alpha alpha beta" in captured.out


def test_lexical_search_cli_empty_query_fails_before_chunk_loading(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "lexical-search",
                "   ",
                "--chunks",
                str(tmp_path / "missing"),
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("amis lexical-search: error:")
    assert "empty" in captured.err
