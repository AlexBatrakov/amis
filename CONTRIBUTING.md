# Contributing

## Project Focus

AMIS is a local-first retrieval and question-answering system for a literary
corpus. The initial product milestone is deliberately narrow: ingest one book,
produce stable text chunks, build a searchable index, and inspect ranked
retrieval results from the command line.

Prioritize retrieval quality, traceable metadata, and reproducibility before
generation, user interfaces, graph features, or deployment infrastructure.

## Public and Private Data

- Never commit copyrighted book text or complete book files.
- Never commit local datasets, generated indexes, model weights, caches,
  credentials, databases, or machine-specific configuration.
- Public fixtures must be synthetic, openly licensed, or small enough to be
  unambiguously safe for the test they support.
- Runtime code must not depend on ignored local workspace directories.
- Use configuration or environment variables for local data paths; do not
  hard-code absolute paths.

## Development Principles

- Work in small, reviewable vertical slices.
- Inspect the current implementation before choosing an abstraction.
- Keep document, embedding, retrieval, and generation boundaries replaceable.
- Preserve source metadata and stable identifiers through ingestion and
  chunking.
- Prefer deterministic processing where practical.
- Avoid introducing infrastructure before the current milestone requires it.
- Treat insufficient evidence as a normal outcome rather than fabricating an
  answer.

## Testing and Verification

- Add tests with behavior changes.
- Cover malformed input and failure paths when changing ingestion, persistence,
  jobs, or external integrations.
- Run the most relevant checks for the changed surface before committing.
- Record any verification that could not be completed and explain why.
- Keep test fixtures small, readable, and legally safe to publish.

## Documentation

- Keep the README focused on what the project does and how to run it.
- Update public architecture or usage documentation when observable behavior,
  setup, or stable boundaries change.
- Document decisions at the level useful to future maintainers; avoid turning
  public documentation into a development diary.

## Git Hygiene

- Keep one coherent change per local working branch.
- Use small, descriptive commits when a change is approved for commit.
- Do not commit secrets, local corpus files, generated artifacts, or temporary
  experiments.
- Do not push, merge, amend, or rewrite history without explicit approval.
- Keep `main` runnable and reviewable.

Preferred commit message examples:

- `build(project): bootstrap Python package`
- `feat(ingestion): add EPUB document loader`
- `test(chunking): cover malformed section metadata`
- `docs(readme): document local corpus setup`
