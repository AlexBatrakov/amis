# Local Data Layout

AMIS operates on user-supplied local documents. Copyrighted source material and
generated retrieval data are intentionally excluded from the repository.

Expected local layout:

```text
data/
  raw/
    books/       source EPUB or text files
  processed/     normalized documents and chunks
  indexes/       vector and lexical search indexes
```

The `raw`, `processed`, and `indexes` directories are ignored by Git. Public
tests must use small synthetic or clearly redistributable fixtures stored under
the test suite rather than files from the local corpus.

Semantic indexes are namespaced by document, chunk policy, and complete index
configuration. Each final directory contains `index_manifest.json`,
`metadata.jsonl`, and `vectors.npy`. Metadata retains source coordinates and
hashes but never stores passage text.

Future ingestion commands will accept configurable paths; application code must
not rely on machine-specific absolute locations.
