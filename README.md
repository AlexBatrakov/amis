# AMIS

AMIS — Ankh-Morpork Intelligence System — is a local-first retrieval and
question-answering project for a literary corpus. It currently provides an
installable Python foundation and command-line smoke entry point; ingestion and
retrieval are not implemented yet.

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

## Development Checks

```bash
pytest
ruff check .
ruff format --check .
```

Copyrighted source text and generated data remain local. See the
[`data` policy](data/README.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).
