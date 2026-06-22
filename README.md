# AMIS

AMIS — Ankh-Morpork Intelligence System — is a local-first retrieval and
question-answering project for a literary corpus. It currently provides a
deterministic one-book EPUB 2 ingestion path; chunking and retrieval are not
implemented yet.

Requires Python 3.13 (`>=3.13,<3.14`).

## Quick Start

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Smoke-check the module entry point:

```bash
python -m amis
```

Normalize one local EPUB 2 source:

```bash
amis ingest data/raw/books/example.epub --output data/processed
```

The command writes one document record and ordered section records:

```text
data/processed/
  doc_sha256_<source-sha256>/
    document.json
    sections.jsonl
```

The source path must name one explicit EPUB file. Directory discovery, EPUB 3,
chunking, and indexing are not supported by this command.

## Development Checks

```bash
pytest
ruff check .
ruff format --check .
```

Keep copyrighted source files and generated output in local paths that are
ignored by Git. See the [`data` policy](data/README.md) and
[`CONTRIBUTING.md`](CONTRIBUTING.md).
