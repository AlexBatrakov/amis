# Public-Domain Demo Corpus

AMIS can acquire a reviewed public-domain demo source so examples, screenshots,
and future public evaluation artifacts do not depend on a private local corpus.
The first supported corpus is the English Constance Garnett translation of
Fyodor Dostoevsky's `Crime and Punishment`.

The repository does not track the downloaded book, normalized sections, chunks,
indexes, model files, or query artifacts. Acquisition is explicit and writes to
ignored local data by default.

## Supported Corpus

```text
corpus_id: crime-and-punishment-garnett
title: Crime and Punishment
author: Fyodor Dostoevsky
translator: Constance Garnett
language: English
source: Project Gutenberg eBook 2554
format: EPUB 2, no images
```

Primary source:
<https://www.gutenberg.org/ebooks/2554.epub.noimages>

Catalog and terms:

- Project Gutenberg catalog page:
  <https://www.gutenberg.org/ebooks/2554>
- Project Gutenberg license and trademark terms:
  <https://www.gutenberg.org/policy/license.html>

The Project Gutenberg catalog currently identifies this eBook as public domain
in the United States. Copyright status is jurisdiction-specific; users outside
the United States should check local law before downloading, distributing, or
using the source. Project Gutenberg trademark and redistribution terms are
separate from the underlying text's U.S. copyright status.

Standard Ebooks also publishes a Constance Garnett edition and source
repository:

- <https://standardebooks.org/ebooks/fyodor-dostoevsky/crime-and-punishment/constance-garnett>
- <https://github.com/standardebooks/fyodor-dostoevsky_crime-and-punishment_constance-garnett>

Those sources are useful references and may be a better curated source for a
future loader, but their downloadable/source forms are EPUB 3/source-folder
inputs. The current AMIS ingestion command supports one local EPUB 2 file, so
the first demo-corpus intake uses the Project Gutenberg EPUB 2 variant.

## Acquire

```bash
amis corpus acquire crime-and-punishment-garnett \
  --output data/raw/public-domain
```

The command writes:

```text
data/raw/public-domain/
  crime-and-punishment-garnett/
    crime-and-punishment-garnett.epub
    source_manifest.json
```

`source_manifest.json` records passage-free provenance: corpus ID, source URL,
final URL, byte size, SHA-256, retrieval timestamp, source format, and legal
caveats. It does not contain book text.

Acquisition verifies the reviewed SHA-256:

```text
45c4d898bf915fd903ecdcc010551e48eed5128bd92940518fc27969e0fe428a
```

If the source changes upstream, acquisition fails rather than silently accepting
different bytes. Re-review the source and update the registry intentionally.

## Ingest and Chunk

After acquisition, run the EPUB 2 ingestion path:

```bash
amis ingest \
  data/raw/public-domain/crime-and-punishment-garnett/crime-and-punishment-garnett.epub \
  --output data/processed
```

The Project Gutenberg source uses leading external XHTML DOCTYPE declarations in
its spine content. AMIS accepts those reviewed XHTML declarations for content
documents while continuing to reject custom entity declarations and unsafe
package/navigation declarations.

Then chunk the produced normalized document directory:

```bash
amis chunk data/processed/doc_sha256_<source-sha256> \
  --output data/processed/chunks
```

Lexical search can run without a model once chunks exist. Semantic and hybrid
search still require the separately acquired local embedding model and a local
index; corpus acquisition does not download models or build indexes.
