# Production paper sync and embedding generation

Promote eligible papers from `paper_new` into `paper_prod` with Azure embeddings.

## Setup

From `vitality2-dataset/`:

```bash
cp .env.example .env   # fill in Zilliz + Azure keys
python3 -m pip install -r script_prod/requirements.txt
```

Required `.env` keys: `ZILLIZ_URI`, `ZILLIZ_TOKEN`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_EMBED_DEPLOYMENT`, `AZURE_OPENAI_EMBED_API_VERSION`.

`AZURE_OPENAI_EMBED_DEPLOYMENT` accepts one name, or a comma-separated list (round-robin; on rate-limit try the next deployment before sleeping).

Non-secret settings (collections, batch sizes, embedding dim / expected model / timeout) live in `script_prod/config.toml`.

## Usage

```bash
python3 script_prod/sync.py
```

It creates `paper_prod` if needed, then: count → per-batch preview (y/n) → embed/upsert → backfill missing embeddings → report.

Each production row stores `embedding_model` from `config.toml` (logical model id, e.g. `text-embedding-3-small`). Azure's `AZURE_OPENAI_EMBED_DEPLOYMENT` is only used to call the API and is never written to rows.

## Sync behavior

Eligible rows: all `eligibility_fields` in `config.toml` are non-empty (default: `title`). DOI and abstract are optional; records without an abstract receive a title-only embedding.

Each batch is classified as:

- **new** — embed + full upsert (`has_embedding` = embed succeeded)
- **embed-input change** — any `embedding_fields` differ after `trim().lower()`, or `embedding_model` mismatch, or `has_embedding` is not true → re-embed + full upsert
- **metadata-only change** — `partial_update` scalars only (existing `embedding` / `embedding_model` / `has_embedding` untouched)
- **unchanged** — skip

Classification never downloads dense vectors. Missing embeddings are tracked with the
BOOL field `has_embedding` (set `true` only when a vector was written successfully;
filter: `has_embedding != true`). Embedding failures still upsert the row with
`has_embedding=false` / null `embedding`; a final pass backfills those.

## UMAP coordinates

Generate two-dimensional visualization coordinates from existing `paper_prod.embedding`
values and store them in the nullable JSON field `umap`:

```bash
python3 script_prod/umap_projection.py fit --execute
```

This fits a new `umap-learn` reducer with cosine distance, saves the fitted model
under `data/zilliz/umap/`, and partial-upserts only `paper_uid` plus `umap`.

For later new papers, keep the saved model and project only rows that do not
already have `umap`:

```bash
python3 script_prod/umap_projection.py transform --execute
```

Use `fit` only when you intentionally want a new global layout. Use `transform`
for incremental updates so existing papers keep stable coordinates.

`script/update_papers.py` runs this as interactive step 9 after an incremental
production sync. It transforms only records whose embedding was newly written
or regenerated in that run, including title-only records without an abstract.
It does not refit the model or scan all of `paper_prod`.

The temporary incomplete-paper production sync can run the same incremental
transform after successful production upserts. It only considers title-bearing
records that are missing a DOI or abstract, and skips rows that already have
UMAP coordinates:

```bash
python3 tmp/sync_incomplete_paper_new_to_prod.py --execute --update-umap
```
