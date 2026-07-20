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
- `data/dblp/dblp_source_list.txt`: DBLP sources extracted from the dump, with paper counts and DOI examples.
- `data/dblp/source_mapping_candidates_report.md`: manual candidate review file. Indented table rows mark selected DBLP sources.
- `data/dblp/source_mapping.csv`: final mapping with columns `source,dblp_source,Full paper`.
- `data/dblp/split_source/`: JSON output split by normalized `source`.
- `data/report/data_report/`: generated data report bundle.

Scripts:

- `script/extract_dblp_sources.py`: streams the DBLP dump and writes `data/dblp/dblp_source_list.txt`.
- `script/split_dblp_by_source.py`: streams the DBLP dump, keeps only DBLP sources listed in `source_mapping.csv`, and writes split JSON files to `data/dblp/split_source/`.
- `script/enrich_openalex_by_doi.py`: uses OpenAlex by DOI to enrich abstract, keywords, and citation count, writing results under `data/papers/`.
- `script/enrich_semantic_scholar_missing.py`: uses Semantic Scholar to fill records still missing abstracts after OpenAlex.
- `script/enrich_crossref_missing.py`: uses Crossref to fill remaining records still missing abstracts after the previous enrichment steps.
- `script/report/data_report.py`: reads split JSON files and generates the report bundle under `data/report/data_report/`.

Common commands:

```bash
python3 script/extract_dblp_sources.py
python3 tmp/extract_selected_source_mapping.py
python3 script/split_dblp_by_source.py --overwrite
python3 script/enrich_openalex_by_doi.py --overwrite
python3 script/enrich_semantic_scholar_missing.py
python3 script/enrich_crossref_missing.py
python3 script/report/data_report.py
```

## Enrichment workflow

After DBLP splitting, enrich metadata in this order:

1. Run OpenAlex first. It is the main source for abstract, keywords, and citation count.
2. Run Semantic Scholar next to fill papers still missing abstracts.
3. Run Crossref last to fill the remaining missing abstracts.

Enriched records are stored under `data/papers/enriched/`; records still missing abstracts stay under `data/papers/missing/`. Cache files are stored under `data/papers/cache/`.

Exported paper fields:

```json
{
  "title": "...",
  "authors": ["..."],
  "source": "CHI",
  "dblp_source": "CHI Extended Abstracts",
  "year": "2026",
  "doi": "10....",
  "abstract": "",
  "keywords": [],
  "citationCounts": null,
  "fullpaper": false
}
```

Notes:

- DBLP has no abstract, keywords, or citation counts; these stay empty until later enrichment.
- DOI is extracted from `<ee>` when possible.
- `data/dblp/dump/dblp.dtd` should stay next to `dblp.xml` because DBLP uses XML entities.
- Do not regenerate `source_mapping_candidates_report.md` after manual marking unless the user explicitly asks.
