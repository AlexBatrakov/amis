# Semantic Index Setup

AMIS uses the gated `google/embeddinggemma-300m` model at one immutable revision
for local semantic indexing. Model weights are not included with AMIS. Access is
subject to the Gemma Terms of Use, and the user must accept those terms and
authenticate with Hugging Face before acquisition.

## Install

Python 3.13 on macOS arm64 is the currently supported production target:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[semantic-index]"
```

The reviewed exact dependency set is recorded in
`requirements/locks/semantic-index-macos-arm64-py313.txt`. It can be installed
with pip's hash enforcement before installing AMIS itself without dependencies:

```bash
python -m pip install --require-hashes \
  -r requirements/locks/semantic-index-macos-arm64-py313.txt
python -m pip install --no-build-isolation --no-deps -e .
```

Ingestion, chunking, general command help, index loading, and vector-level search
do not import or initialize the optional PyTorch model runtime.

## Acquire and Verify the Model

Authenticate using a supported Hugging Face login mechanism. Do not put a token
in an AMIS command, repository file, or shell history. Then invoke the only
network-capable model action:

```bash
amis model acquire
```

Acquisition requests the exact pinned revision, stages only the required runtime
files, verifies sizes and SHA-256 hashes, and publishes the snapshot atomically.
AMIS does not accept the upstream terms on the user's behalf.

The cache root is selected in this order:

1. `--cache-root PATH`;
2. `AMIS_MODEL_CACHE`;
3. `~/.cache/amis/models`.

Local verification performs no network access:

```bash
amis model verify
```

An explicit `--model-snapshot PATH` may select an existing read-only snapshot,
but it must remain contained by the selected cache root. Required symbolic-link
targets must also remain inside that root.

## Offline Index Build

Pass one complete chunk-policy directory from the version-one chunker and a
disjoint output root:

```bash
amis index build <chunk-policy-directory> --output data/indexes
```

The build re-verifies the local snapshot, validates every chunk and identity,
counts each fully prompted document with truncation disabled, and rejects any
document above 1,984 tokens before embedding. It never downloads a model,
truncates text, rechunks input, or stores passage text in index metadata.

The final layout is:

```text
<output-root>/
  <document-id>/
    <policy-id>/
      <index-config-id>/
        index_manifest.json
        metadata.jsonl
        vectors.npy
```

`vectors.npy` is a full 768-dimensional float32 unit-normalized matrix loaded
with pickle disabled. `metadata.jsonl` maps rows to stable source identities and
coordinates. The manifest binds input, model, prompts, runtime, vectors,
metadata, and deterministic ranking behavior by hash.

Equivalent builds are idempotent. Conflicting or corrupt existing destinations
are preserved and reported as errors. A failed build cannot replace a prior
valid index.

Common errors identify missing optional dependencies, unauthenticated gated
access, absent or mismatched snapshot files, token overflow, malformed vectors,
stale input, unsafe paths, or conflicting output. Errors never include passage
text or credentials.

## Local Lexical Search

AMIS can also search one chunk-policy directory directly with BM25 lexical
retrieval:

```bash
amis lexical-search "synthetic query" \
  --chunks data/processed/chunks/doc_sha256_<source-sha256>/\
chunk_policy_sha256_<policy-sha256> \
  --top-k 5 \
  --excerpt-chars 320
```

Lexical search validates the chunk manifest and chunk stream before displaying
citations. It does not require a semantic index, embedding model, model cache,
or network access.

The lexical analyzer applies Unicode NFKC normalization, case folding, and token
splitting over contiguous Unicode letter, mark, and number categories. It does
not use stemming, stopwords, synonym expansion, reranking, or answer generation.
This makes it predictable for exact names, titles, quote fragments, rare words,
and spelling-sensitive lookups, but less flexible for paraphrases.

Each result row includes rank, BM25 lexical score, relative source path,
document/chunk/section identifiers, source coordinates, text hash, and a bounded
display excerpt. BM25 lexical scores are internal lexical ranking scores and
should not be compared directly to vector cosine scores from semantic search.

## Local Search and Citations

Search uses one existing semantic index, the matching chunk-policy directory,
and a verified local model snapshot:

```bash
amis search "synthetic query" \
  --index data/indexes/doc_sha256_<source-sha256>/\
chunk_policy_sha256_<policy-sha256>/index_config_sha256_<config-sha256> \
  --chunks data/processed/chunks/doc_sha256_<source-sha256>/\
chunk_policy_sha256_<policy-sha256> \
  --top-k 5 \
  --excerpt-chars 320
```

The command applies the pinned query transform, counts the fully transformed
query without truncation, rejects empty or over-limit queries, embeds exactly one
query vector, and delegates ranking to the exact cosine top-k primitive. It
loads the index with the supplied chunk artifact as expected input before any
citation text is displayed, so stale or mismatched chunks cannot be cited.

Each result row includes rank, score, relative source path, document/chunk/section
identifiers, source coordinates, text hash, and a bounded display excerpt. The
coordinates refer to the full retrieved chunk; the excerpt is only a shortened
runtime display. Search does not acquire models, download files, generate
answers, rerank, evaluate quality, log queries, or write result artifacts.
