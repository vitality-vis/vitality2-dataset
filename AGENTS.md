# AGENTS.md

This repository is for creating and updating the Vitality2 database.

This is still an early version, and the full plan is not finalized yet. The rough workflow is:

1. Download the DBLP dump file.
2. Clean and organize DBLP data.
3. Select the required sources from the cleaned data.
4. Use OpenAlex to enrich abstracts and related metadata.
5. Generate embeddings for the processed data.
6. Upload the data to Zilliz.


## DBLP workflow

The DBLP dump is `data/dblp/dump/dblp.xml` and is several GB. Always process it with streaming XML parsing. Do not load the whole file or the full paper list into memory.

Only these DBLP record types are treated as papers:

- `article`: source is `<journal>`
- `inproceedings`: source is `<booktitle>`

Other record types such as `proceedings` and `www` are not exported as papers.

Key files:

- `data/source_list.txt`: manually maintained target source list.
- `data/dblp/dump/dblp.xml`: local DBLP XML dump.
- `data/dblp/dump/dblp.dtd`: DBLP DTD file required for XML entity parsing.
- `data/dblp/dblp_source_list.txt`: DBLP sources extracted from the dump, with paper counts and DOI examples.
- `data/dblp/source_mapping_candidates_report.md`: manual candidate review file. Indented table rows mark selected DBLP sources.
- `data/dblp/source_mapping.csv`: final mapping with columns `source,dblp_source,Full paper`.
- `data/dblp/split_source/`: JSON output split by normalized `source`.
- `data/zilliz/paper_new_dblp_keys.txt`: exported existing `dblp_key` values from Zilliz for update filtering.
- `data/zilliz/paper_new_dblp_keys.txt.manifest.json`: metadata for the DBLP key export.
- `data/papers/enriched/`: enriched paper JSON files ready for upload.
- `data/papers/missing/`: paper JSON files still missing abstracts after enrichment.
- `data/papers/cache/`: local DOI lookup caches for OpenAlex, Semantic Scholar, and Crossref.
- `data/papers/*_manifest.json`: enrichment run manifests.
- `data/papers/updateYYYYMMDD/`: incremental update batch output, with its own split, enriched, missing, cache, and manifest files.
- `data/report/data_report/`: generated data report bundle.
- `paper_new`: main Zilliz paper collection.
- `paper_stats`: Zilliz statistics collection generated from `paper_new`.
- `dashboard/index.html`: simple local dashboard for viewing paper statistics.

Scripts:

- `script/download_dblp_dump.py`: downloads and unpacks the DBLP dump.
- `script/extract_dblp_sources.py`: streams the DBLP dump and writes `data/dblp/dblp_source_list.txt`.
- `script/split_dblp_by_source.py`: streams the DBLP dump, keeps only DBLP sources listed in `source_mapping.csv`, and writes split JSON files to `data/dblp/split_source/`.
- `script/export_zilliz_paper_uids.py`: reads existing scalar values such as `dblp_key` or `paper_uid` from Zilliz and writes a local file.
- `script/filter_new_dblp_papers.py`: filters split DBLP papers against the local DBLP key file and writes an update batch under `data/papers/updateYYYYMMDD/`.
- `script/enrich_openalex_by_doi.py`: uses OpenAlex by DOI to enrich abstract, keywords, and citation count, writing results under `data/papers/`.
- `script/enrich_semantic_scholar_missing.py`: uses Semantic Scholar to fill records still missing abstracts after OpenAlex.
- `script/enrich_crossref_missing.py`: uses Crossref to fill remaining records still missing abstracts after the previous enrichment steps.
- `script/create_zilliz_collection.py`: creates the Zilliz collection schema.
- `script/index_zilliz_collection.py`: creates indexes and loads the Zilliz paper collection.
- `script/upload_papers_to_zilliz.py`: uploads enriched and missing paper records to Zilliz.
- `script/create_paper_stats_collection.py`: creates the `paper_stats` statistics collection.
- `script/materialize_paper_stats.py`: reads all records from `paper_new`, builds statistics, and writes them to `paper_stats`.
- `script/report/data_report.py`: reads split JSON files and generates the report bundle under `data/report/data_report/`.
- `dashboard/index.html`: simple local dashboard for viewing paper statistics.

## Script parameters

- `script/download_dblp_dump.py`: `--output-dir`, `--xml-url`, `--dtd-url`, `--dtd-only`, `--xml-only`, `--keep-gz`.
- `script/extract_dblp_sources.py`: `--input`, `--output`.
- `tmp/extract_selected_source_mapping.py`: `--report`, `--output`, `--vitality-mapping`.
- `script/split_dblp_by_source.py`: `--input`, `--output-dir`, `--mapping`, `--overwrite`, `--max-open-files`, `--limit`.
- `script/export_zilliz_paper_uids.py`: `--collection`, `--field`, `--output`, `--batch-size`, `--query-timeout`, `--limit`, `--load`.
- `script/filter_new_dblp_papers.py`: `--split-dir`, `--existing-file`, `--field`, `--uid-file`, `--output-dir`, `--overwrite`, `--limit`.
- `script/enrich_openalex_by_doi.py`: `--input-dir`, `--output-dir`, `--cache`, `--env-file`, `--api-key`, `--use-env-api-key`, `--email`, `--source`, `--limit`, `--sleep`, `--workers`, `--max-pending`, `--request-timeout`, `--progress-every`, `--max-retries`, `--retry-backoff`, `--overwrite`.
- `script/enrich_semantic_scholar_missing.py`: `--papers-dir`, `--cache`, `--env-file`, `--api-key`, `--use-env-api-key`, `--use-empty-api-key-header`, `--source`, `--batch-size`, `--request-timeout`, `--max-retries`, `--retry-backoff`, `--rate-limit-retry-sleep`, `--sleep`, `--progress-every`.
- `script/enrich_crossref_missing.py`: `--papers-dir`, `--cache`, `--env-file`, `--mailto`, `--use-env-mailto`, `--source`, `--request-timeout`, `--max-retries`, `--retry-backoff`, `--rate-limit-retry-sleep`, `--workers`, `--max-pending`, `--rate-limit`, `--sleep`, `--progress-every`.
- `script/report/data_report.py`: `--input-dir`, `--papers-dir`, `--output-dir`.
- `script/create_zilliz_collection.py`: `--collection`, `--database`, `--create-database`, `--embedding-dim`, `--metric-type`, `--index-type`, `--load`, `--defer-index`, `--keep-existing`, `--drop-existing`, `--execute`.
- `script/index_zilliz_collection.py`: `--collection`, `--metric-type`, `--index-type`, `--only`, `--no-load`.
- `script/upload_papers_to_zilliz.py`: `--collection`, `--papers-dir`, `--existing-uid-file`, `--batch-size`, `--skip`, `--allow-non-empty`, `--resume`.
- `script/create_paper_stats_collection.py`: `--collection`, `--drop-existing`, `--execute`.
- `script/materialize_paper_stats.py`: `--source-collection`, `--stats-collection`, `--read-batch-size`, `--write-batch-size`, `--replace` / `--no-replace`.

Common commands:

```bash
python3 script/download_dblp_dump.py
python3 script/extract_dblp_sources.py
python3 tmp/extract_selected_source_mapping.py
python3 script/split_dblp_by_source.py --overwrite
python3 script/export_zilliz_paper_uids.py --field dblp_key
python3 script/filter_new_dblp_papers.py
python3 script/enrich_openalex_by_doi.py --overwrite
python3 script/enrich_semantic_scholar_missing.py
python3 script/enrich_crossref_missing.py
python3 script/report/data_report.py
python3 script/create_zilliz_collection.py
python3 script/create_zilliz_collection.py --execute
python3 script/upload_papers_to_zilliz.py
python3 script/index_zilliz_collection.py
python3 script/create_paper_stats_collection.py --execute
python3 script/materialize_paper_stats.py
```

## Enrichment workflow

After DBLP splitting, enrich metadata in this order:

1. Run OpenAlex first. It is the main source for abstract, keywords, and citation count.
2. Run Semantic Scholar next to fill papers still missing abstracts.
3. Run Crossref last to fill the remaining missing abstracts.

Enriched records are stored under `data/papers/enriched/`; records still missing abstracts stay under `data/papers/missing/`. Cache files are stored under `data/papers/cache/`.

## Incremental update workflow

Use this workflow after downloading a fresh DBLP dump or changing `data/dblp/source_mapping.csv`.

1. Rebuild the split DBLP files:

```bash
python3 script/split_dblp_by_source.py --overwrite
```

2. Export existing `dblp_key` values from Zilliz to a local file:

```bash
python3 script/export_zilliz_paper_uids.py --collection paper_new --field dblp_key --load --batch-size 5000
```

This writes:

- `data/zilliz/paper_new_dblp_keys.txt`
- `data/zilliz/paper_new_dblp_keys.txt.manifest.json`

Use `dblp_key` as the update baseline. `paper_uid` may change from `dblp:...` to `doi:...` when DOI extraction improves, but DBLP's `key` identifies the same DBLP record across that change.

The export script loads the requested scalar field with `search_sparse` because Zilliz requires a vector field in `load_fields`. If load fails because the sparse index is missing, run:

```bash
python3 script/index_zilliz_collection.py --collection paper_new --only search_sparse
```

3. Filter new papers into an update batch:

```bash
python3 script/filter_new_dblp_papers.py --overwrite
```

This creates `data/papers/updateYYYYMMDD/` with:

- `split_source/`: only papers whose `dblp_key` is not in the exported Zilliz DBLP key file.
- `enriched/`: empty directory for enrichment output.
- `missing/`: empty directory for enrichment output.
- `filter_manifest.json`: filtering counts and source-level totals.

4. Enrich only the update batch:

```bash
python3 script/enrich_openalex_by_doi.py --input-dir data/papers/updateYYYYMMDD/split_source --output-dir data/papers/updateYYYYMMDD --overwrite
python3 script/enrich_semantic_scholar_missing.py --papers-dir data/papers/updateYYYYMMDD
python3 script/enrich_crossref_missing.py --papers-dir data/papers/updateYYYYMMDD
```

The update batch then contains its own `enriched/`, `missing/`, `cache/`, and manifest files. Do not run the default full-data enrichment commands for incremental updates unless a full rebuild is intended.

This workflow stops after enrichment. Upload/update to Zilliz is a separate step and should not be assumed.

## Zilliz workflow

After enrichment, create the Zilliz collection first, then upload records from `data/papers/enriched/` and `data/papers/missing/`.

Use `script/create_zilliz_collection.py` without `--execute` to preview the schema. Use `--execute` only when ready to create the collection. Then run `script/upload_papers_to_zilliz.py` to upload the data.

For statistics, create `paper_stats` with `script/create_paper_stats_collection.py`, then run `script/materialize_paper_stats.py` to materialize stats from `paper_new`.

Use `dashboard/index.html` as a simple local dashboard for the statistics.

Exported paper fields:

```json
{
  "paper_uid": "doi:10....",
  "dblp_key": "conf/chi/...",
  "title": "...",
  "authors": ["..."],
  "source": "CHI",
  "dblp_source": "CHI Extended Abstracts",
  "year": "2026",
  "doi": "10....",
  "abstract": "",
  "keywords": [],
  "citation_count": null,
  "full_paper": false
}
```

Notes:

- DBLP has no abstract, keywords, or citation counts; these stay empty until later enrichment.
- DOI is extracted from `<ee>` when possible.
- `data/dblp/dump/dblp.dtd` should stay next to `dblp.xml` because DBLP uses XML entities.
- Do not regenerate `source_mapping_candidates_report.md` after manual marking unless the user explicitly asks.
