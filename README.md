# AMIS

AMIS — Ankh-Morpork Intelligence System — is a local-first retrieval and
question-answering project for a literary corpus. It provides deterministic
one-book EPUB 2 ingestion, model-independent chunking, and a validated local
semantic index with exact cosine ranking.

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

Acquire the optional public-domain demo corpus:

```bash
amis corpus acquire crime-and-punishment-garnett \
  --output data/raw/public-domain
```

This fetches the reviewed Project Gutenberg EPUB 2 source for Constance
Garnett's English translation of `Crime and Punishment`, verifies its SHA-256,
and writes a passage-free local provenance manifest. The downloaded book remains
ignored local data. See
[`docs/public-domain-corpus.md`](docs/public-domain-corpus.md) for source,
format, copyright-jurisdiction, and Project Gutenberg trademark caveats.

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

Search one chunk-policy directory with deterministic BM25 lexical retrieval:

```bash
amis lexical-search "your query" \
  --chunks data/processed/chunks/doc_sha256_<source-sha256>/\
chunk_policy_sha256_<policy-sha256> \
  --top-k 5 \
  --excerpt-chars 320
```

Lexical search requires no embedding model, vector index, model cache, or
network access. It is useful for exact names, titles, quote fragments, rare
words, and spelling-sensitive terms. Results display BM25 lexical scores, which
are not comparable to vector cosine scores, plus bounded runtime excerpts.

Build the optional semantic-index runtime, explicitly acquire the pinned gated
model, and index one chunk-policy directory:

```bash
python -m pip install -e ".[semantic-index]"
amis model acquire
amis index build data/processed/chunks/doc_sha256_<source-sha256>/\
chunk_policy_sha256_<policy-sha256> --output data/indexes
```

Model acquisition is the only network-capable model action. Index builds verify
local model files and do not download anything. See
[`docs/semantic-index.md`](docs/semantic-index.md) for terms, cache overrides,
offline behavior, artifact layout, and the supported dependency lock.

Search one validated local index and display ranked citations:

```bash
amis search "your query" \
  --index data/indexes/doc_sha256_<source-sha256>/\
chunk_policy_sha256_<policy-sha256>/index_config_sha256_<config-sha256> \
  --chunks data/processed/chunks/doc_sha256_<source-sha256>/\
chunk_policy_sha256_<policy-sha256> \
  --top-k 5 \
  --excerpt-chars 320
```

Search verifies the local model snapshot, validates the supplied index and
matching chunks together, and prints bounded runtime excerpts. It does not
download models, generate answers, or persist query text.

Fuse semantic/vector and BM25 lexical candidates with deterministic Reciprocal
Rank Fusion:

```bash
amis hybrid-search "your query" \
  --index data/indexes/doc_sha256_<source-sha256>/\
chunk_policy_sha256_<policy-sha256>/index_config_sha256_<config-sha256> \
  --chunks data/processed/chunks/doc_sha256_<source-sha256>/\
chunk_policy_sha256_<policy-sha256> \
  --top-k 5 \
  --candidate-k 20 \
  --excerpt-chars 320
```

Hybrid search requires the local embedding model because vector retrieval is
active. It does not add cosine and BM25 scores together; it fuses source ranks
with RRF and displays fused ranking scores separately from vector cosine and
lexical BM25 scores. Fused scores are not calibrated relevance probabilities.
Hybrid search does not rerank with a separate model or generate answers.

## Development Checks

```bash
pytest
ruff check .
ruff format --check .
```

Keep copyrighted source files and generated output in local paths that are
ignored by Git. See the [`data` policy](data/README.md) and
[`CONTRIBUTING.md`](CONTRIBUTING.md).
