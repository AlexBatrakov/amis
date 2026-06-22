# AMIS

AMIS — Ankh-Morpork Intelligence System — is a local-first retrieval and
question-answering project for a literary corpus. It currently provides
deterministic one-book EPUB 2 ingestion and model-independent character
chunking. Retrieval is not implemented yet.

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
and indexing are not supported by this command.

Chunk one normalized document into exact, citable section spans:

```bash
amis chunk data/processed/doc_sha256_<source-sha256> \
  --output data/processed/chunks
```

The default policy uses a 3,000-character target, a 4,000-character hard
maximum, and up to 400 characters of source overlap. Each distinct policy has
a stable identifier and a separate output namespace:

```text
data/processed/chunks/
  doc_sha256_<source-sha256>/
    chunk_policy_sha256_<policy-sha256>/
      chunk_manifest.json
      chunks.jsonl
```

Use `--target-chars`, `--max-chars`, and `--overlap-chars` to select another
versioned character policy. The chunker reads only the normalized records,
never crosses section boundaries, and does not use a model tokenizer. The
input document directory and chunk output root must be disjoint.

## Development Checks

```bash
pytest
ruff check .
ruff format --check .
```

Keep copyrighted source files and generated output in local paths that are
ignored by Git. See the [`data` policy](data/README.md) and
[`CONTRIBUTING.md`](CONTRIBUTING.md).
