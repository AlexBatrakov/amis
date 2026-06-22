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
