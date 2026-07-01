"""Smoke tests for the AMIS package and CLI."""

import hashlib
import subprocess
import sys
from pathlib import Path

from pytest import CaptureFixture, MonkeyPatch

import amis
import amis.cli as cli
import amis.embeddings as embeddings
import amis.public_corpus as public_corpus
from amis.cli import READY_MESSAGE, main
from amis.model_cache import VerifiedModel
from amis.model_spec import EMBEDDING_GEMMA
from amis.public_corpus import PublicCorpusSource
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


def test_hybrid_search_help_does_not_import_optional_runtime() -> None:
    script = """
import sys
from amis.cli import main
try:
    main(['hybrid-search', '--help'])
except SystemExit as error:
    assert error.code == 0
assert 'torch' not in sys.modules
assert 'sentence_transformers' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_corpus_acquire_help_does_not_import_optional_runtime() -> None:
    script = """
import sys
from amis.cli import main
try:
    main(['corpus', 'acquire', '--help'])
except SystemExit as error:
    assert error.code == 0
assert 'torch' not in sys.modules
assert 'sentence_transformers' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_corpus_acquire_cli_prints_passage_free_provenance(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: MonkeyPatch
) -> None:
    content = b"synthetic public-domain epub bytes"
    source = _synthetic_public_corpus_source(content)
    monkeypatch.setattr(
        public_corpus, "PUBLIC_CORPUS_REGISTRY", {source.corpus_id: source}
    )
    monkeypatch.setattr(
        public_corpus,
        "_download_url",
        lambda url: public_corpus._DownloadedSource(
            content=content,
            final_url="https://example.test/final.epub",
            content_type="application/epub+zip",
        ),
    )

    assert (
        main(
            [
                "corpus",
                "acquire",
                source.corpus_id,
                "--output",
                str(tmp_path / "raw"),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert f"Acquired public corpus {source.corpus_id}." in captured.out
    assert "source_url: https://example.test/source.epub" in captured.out
    assert "final_url: https://example.test/final.epub" in captured.out
    assert "byte_size: 34" in captured.out
    assert f"sha256: {hashlib.sha256(content).hexdigest()}" in captured.out
    assert "synthetic public-domain epub bytes" not in captured.out


def test_corpus_acquire_cli_reports_unsupported_id(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "corpus",
                "acquire",
                "unknown-corpus",
                "--output",
                str(tmp_path / "raw"),
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("amis corpus: error:")
    assert "unsupported public corpus ID" in captured.err


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


def test_hybrid_search_cli_prints_synthetic_citations(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: MonkeyPatch
) -> None:
    chunks = write_chunk_policy_from_texts(
        tmp_path / "input",
        [
            "Alpha beta appears in the first synthetic notice.",
            "Gamma appears in the second synthetic notice.",
            "Delta appears in the third synthetic notice.",
        ],
    )
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
                "hybrid-search",
                "gamma",
                "--index",
                str(result.output_directory),
                "--chunks",
                str(chunks),
                "--top-k",
                "1",
                "--candidate-k",
                "3",
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
    assert "Rank 1 | fused_rrf_score " in captured.out
    assert "source_membership: both" in captured.out
    assert "vector_rank: 1 | vector_cosine_score 1.000000" in captured.out
    assert "lexical_rank: 2 | lexical_bm25_score " in captured.out
    assert "chunk_id: chunk_sha256_" in captured.out
    assert "section_id: sec_sha256_" in captured.out
    assert "excerpt: Alpha beta appears" in captured.out


def test_hybrid_search_cli_empty_query_fails_before_model_verification(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: MonkeyPatch
) -> None:
    def verify_should_not_run(*args: object, **kwargs: object) -> VerifiedModel:
        raise AssertionError("model verification must not run")

    monkeypatch.setattr(cli, "verify_model_snapshot", verify_should_not_run)

    assert (
        main(
            [
                "hybrid-search",
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
    assert captured.err.startswith("amis hybrid-search: error:")
    assert "empty" in captured.err


def test_hybrid_search_cli_candidate_validation_precedes_model_verification(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: MonkeyPatch
) -> None:
    def verify_should_not_run(*args: object, **kwargs: object) -> VerifiedModel:
        raise AssertionError("model verification must not run")

    monkeypatch.setattr(cli, "verify_model_snapshot", verify_should_not_run)

    assert (
        main(
            [
                "hybrid-search",
                "alpha",
                "--index",
                str(tmp_path / "index"),
                "--chunks",
                str(tmp_path / "chunks"),
                "--top-k",
                "2",
                "--candidate-k",
                "1",
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("amis hybrid-search: error:")
    assert "candidate-k" in captured.err


def _synthetic_public_corpus_source(content: bytes) -> PublicCorpusSource:
    return PublicCorpusSource(
        corpus_id="synthetic-public-book",
        title="Synthetic Public Book",
        author="Example Author",
        translator="Example Translator",
        language="English",
        original_publication_year=1900,
        translation_publication_year=1901,
        provider="Example Public Source",
        catalog_url="https://example.test/catalog",
        source_url="https://example.test/source.epub",
        source_format="epub2",
        artifact_name="synthetic-public-book.epub",
        expected_sha256=hashlib.sha256(content).hexdigest(),
        expected_size=len(content),
        source_checked_date="2026-07-01",
        license_url="https://example.test/license",
        legal_basis="Synthetic public fixture for tests.",
        caveats=("Synthetic fixture caveat.",),
        alternate_sources=(),
    )
