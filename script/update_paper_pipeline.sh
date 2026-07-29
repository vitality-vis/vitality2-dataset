#!/usr/bin/env bash
# Run the incremental new-paper pipeline and prepare existing-paper update inputs:
# DBLP download -> split -> Zilliz key/DOI export -> existing missing-abstract
# candidate export -> new-paper filter -> new-paper enrichment/upload -> stats refresh.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

UPDATE_DATE="${UPDATE_DATE:-$(date +%Y%m%d)}"
UPDATE_DIR="${UPDATE_DIR:-data/papers/update${UPDATE_DATE}}"

PAPER_COLLECTION="${PAPER_COLLECTION:-paper_new}"
STATS_COLLECTION="${STATS_COLLECTION:-paper_stats}"
DBLP_KEY_FILE="${DBLP_KEY_FILE:-data/zilliz/${PAPER_COLLECTION}_dblp_keys.txt}"
DOI_FILE="${DOI_FILE:-data/zilliz/${PAPER_COLLECTION}_dois.txt}"
EXISTING_METADATA_FILE="${EXISTING_METADATA_FILE:-data/zilliz/${PAPER_COLLECTION}_update_metadata.jsonl}"
EXISTING_UPDATE_DIR="${EXISTING_UPDATE_DIR:-data/papers/update_missing${UPDATE_DATE}}"
EXISTING_UPDATE_YEAR="${EXISTING_UPDATE_YEAR:-$(date +%Y)}"

DOWNLOAD_DBLP="${DOWNLOAD_DBLP:-1}"
SPLIT_DBLP="${SPLIT_DBLP:-1}"
EXPORT_EXISTING_PAPERS="${EXPORT_EXISTING_PAPERS:-1}"
FILTER_NEW="${FILTER_NEW:-1}"
RUN_ENRICH="${RUN_ENRICH:-1}"
RUN_UPLOAD="${RUN_UPLOAD:-1}"
RUN_STATS="${RUN_STATS:-1}"
RUN_POST_DOI_DEDUP="${RUN_POST_DOI_DEDUP:-1}"

SPLIT_MAX_OPEN_FILES="${SPLIT_MAX_OPEN_FILES:-128}"
DBLP_DOWNLOAD_WORKERS="${DBLP_DOWNLOAD_WORKERS:-16}"
DBLP_SEGMENT_SIZE_MB="${DBLP_SEGMENT_SIZE_MB:-32}"
ZILLIZ_EXPORT_BATCH_SIZE="${ZILLIZ_EXPORT_BATCH_SIZE:-5000}"
UPLOAD_BATCH_SIZE="${UPLOAD_BATCH_SIZE:-500}"
STATS_READ_BATCH_SIZE="${STATS_READ_BATCH_SIZE:-5000}"
STATS_WRITE_BATCH_SIZE="${STATS_WRITE_BATCH_SIZE:-500}"

OPENALEX_WORKERS="${OPENALEX_WORKERS:-16}"
OPENALEX_MAX_PENDING="${OPENALEX_MAX_PENDING:-128}"
OPENALEX_SLEEP="${OPENALEX_SLEEP:-0}"
OPENALEX_SEARCH_SLEEP="${OPENALEX_SEARCH_SLEEP:-1}"

S2_BATCH_SIZE="${S2_BATCH_SIZE:-100}"
S2_SLEEP="${S2_SLEEP:-1}"

CROSSREF_WORKERS="${CROSSREF_WORKERS:-1}"
CROSSREF_MAX_PENDING="${CROSSREF_MAX_PENDING:-16}"
CROSSREF_RATE_LIMIT="${CROSSREF_RATE_LIMIT:-5}"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run() {
  log "+ $*"
  "$@"
}

run_nonfatal() {
  log "+ $*"
  if ! "$@"; then
    log "Command failed; continuing with the remaining update flow."
  fi
}

json_number() {
  local path="$1"
  local key="$2"
  "$PYTHON_BIN" - "$path" "$key" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
if not path.exists():
    print(0)
    raise SystemExit
data = json.loads(path.read_text(encoding="utf-8"))
value = data
for part in key.split("."):
    if isinstance(value, dict):
        value = value.get(part, 0)
    else:
        value = 0
print(int(value or 0))
PY
}

has_non_missing_doi_missing_files() {
  find "$UPDATE_DIR/missing" -maxdepth 1 -type f -name '*.json' ! -name '_missing_doi.json' -size +0c \
    | grep -q .
}

has_missing_doi_file() {
  local papers_dir="${1:-$UPDATE_DIR}"
  [[ -s "$papers_dir/missing/_missing_doi.json" ]]
}

has_non_missing_doi_missing_files_in() {
  local papers_dir="$1"
  find "$papers_dir/missing" -maxdepth 1 -type f -name '*.json' ! -name '_missing_doi.json' -size +0c \
    | grep -q .
}

has_split_source_files() {
  local papers_dir="$1"
  find "$papers_dir/split_source" -maxdepth 1 -type f -name '*.json' -size +0c | grep -q .
}

run_openalex_doi_enrichment() {
  local papers_dir="$1"
  local input_dir="$2"
  local cache_prefix="$3"
  local overwrite_flag="$4"

  run "$PYTHON_BIN" script/enrich_openalex_by_doi.py \
    --input-dir "$input_dir" \
    --output-dir "$papers_dir" \
    --cache "$papers_dir/cache/${cache_prefix}openalex_doi_cache.jsonl" \
    --workers "$OPENALEX_WORKERS" \
    --max-pending "$OPENALEX_MAX_PENDING" \
    --sleep "$OPENALEX_SLEEP" \
    "$overwrite_flag"
}

run_semantic_scholar_and_crossref_enrichment() {
  local papers_dir="$1"
  local cache_prefix="$2"

  if has_non_missing_doi_missing_files_in "$papers_dir"; then
    run_nonfatal "$PYTHON_BIN" script/enrich_semantic_scholar_missing.py \
      --papers-dir "$papers_dir" \
      --cache "$papers_dir/cache/${cache_prefix}semantic_scholar_doi_cache.jsonl" \
      --use-env-api-key \
      --batch-size "$S2_BATCH_SIZE" \
      --sleep "$S2_SLEEP"
  else
    log "No DOI-based missing files for Semantic Scholar in $papers_dir."
  fi

  if has_non_missing_doi_missing_files_in "$papers_dir"; then
    run_nonfatal "$PYTHON_BIN" script/enrich_crossref_missing.py \
      --papers-dir "$papers_dir" \
      --cache "$papers_dir/cache/${cache_prefix}crossref_doi_cache.jsonl" \
      --use-env-mailto \
      --workers "$CROSSREF_WORKERS" \
      --max-pending "$CROSSREF_MAX_PENDING" \
      --rate-limit "$CROSSREF_RATE_LIMIT"
  else
    log "No DOI-based missing files for Crossref in $papers_dir."
  fi
}

if [[ "$DOWNLOAD_DBLP" == "1" ]]; then
  run "$PYTHON_BIN" script/download_dblp_dump.py \
    --download-workers "$DBLP_DOWNLOAD_WORKERS" \
    --segment-size-mb "$DBLP_SEGMENT_SIZE_MB"
fi

if [[ "$SPLIT_DBLP" == "1" ]]; then
  run "$PYTHON_BIN" script/split_dblp_by_source.py \
    --overwrite \
    --max-open-files "$SPLIT_MAX_OPEN_FILES"
fi

if [[ "$EXPORT_EXISTING_PAPERS" == "1" ]]; then
  run "$PYTHON_BIN" script/export_zilliz_paper_update_candidates.py \
    --collection "$PAPER_COLLECTION" \
    --keys-output "$DBLP_KEY_FILE" \
    --dois-output "$DOI_FILE" \
    --metadata-output "$EXISTING_METADATA_FILE" \
    --candidates-dir "$EXISTING_UPDATE_DIR/split_source" \
    --candidate-year "$EXISTING_UPDATE_YEAR" \
    --load \
    --batch-size "$ZILLIZ_EXPORT_BATCH_SIZE" \
    --overwrite
fi

if [[ "$FILTER_NEW" == "1" ]]; then
  if [[ -n "${LIMIT:-}" ]]; then
    run "$PYTHON_BIN" script/filter_new_dblp_papers.py \
      --existing-file "$DBLP_KEY_FILE" \
      --existing-doi-file "$DOI_FILE" \
      --field dblp_key \
      --output-dir "$UPDATE_DIR" \
      --overwrite \
      --limit "$LIMIT"
  else
    run "$PYTHON_BIN" script/filter_new_dblp_papers.py \
      --existing-file "$DBLP_KEY_FILE" \
      --existing-doi-file "$DOI_FILE" \
      --field dblp_key \
      --output-dir "$UPDATE_DIR" \
      --overwrite
  fi
fi

NEW_PAPERS="$(json_number "$UPDATE_DIR/filter_manifest.json" new_papers)"
EXISTING_CANDIDATES="$(json_number "$EXISTING_UPDATE_DIR/existing_missing_abstract_manifest.json" existing_doi_missing_abstract)"
log "Update directory: $UPDATE_DIR"
log "Existing missing-abstract directory: $EXISTING_UPDATE_DIR"
log "New DBLP papers: $NEW_PAPERS"
log "Existing DOI/no-abstract papers for $EXISTING_UPDATE_YEAR: $EXISTING_CANDIDATES"

if [[ "$NEW_PAPERS" -eq 0 ]] && [[ "$EXISTING_CANDIDATES" -eq 0 ]]; then
  log "No new papers or existing DOI/no-abstract candidates found. Skipping enrichment, upload, and stats refresh."
  exit 0
fi

if [[ "$RUN_ENRICH" == "1" ]] && [[ "$NEW_PAPERS" -gt 0 ]]; then
  run_openalex_doi_enrichment "$UPDATE_DIR" "$UPDATE_DIR/split_source" "" "--overwrite"

  if has_missing_doi_file "$UPDATE_DIR"; then
    run "$PYTHON_BIN" script/enrich_openalex_missing_doi_by_search.py \
      --papers-dir "$UPDATE_DIR" \
      --cache "$UPDATE_DIR/cache/openalex_missing_doi_search_cache.jsonl" \
      --use-env-api-key \
      --sleep "$OPENALEX_SEARCH_SLEEP"
  else
    log "No _missing_doi.json records for OpenAlex title search."
  fi

  if [[ "$RUN_POST_DOI_DEDUP" == "1" ]]; then
    run "$PYTHON_BIN" script/filter_existing_dois_from_paper_batch.py \
      --papers-dir "$UPDATE_DIR" \
      --existing-doi-file "$DOI_FILE"
    NEW_PAPERS="$(json_number "$UPDATE_DIR/post_doi_dedupe_manifest.json" retained_papers)"
    log "New DBLP papers after recovered-DOI deduplication: $NEW_PAPERS"
  fi

  if has_non_missing_doi_missing_files; then
    run_semantic_scholar_and_crossref_enrichment "$UPDATE_DIR" ""
  fi
fi

if [[ "$RUN_UPLOAD" == "1" ]] && [[ "$NEW_PAPERS" -gt 0 ]]; then
  run "$PYTHON_BIN" script/upload_papers_to_zilliz.py \
    --collection "$PAPER_COLLECTION" \
    --papers-dir "$UPDATE_DIR" \
    --batch-size "$UPLOAD_BATCH_SIZE" \
    --allow-non-empty \
    --skip-missing-doi
fi

if [[ "$RUN_STATS" == "1" ]] && [[ "$NEW_PAPERS" -gt 0 ]]; then
  run "$PYTHON_BIN" script/materialize_paper_stats.py \
    --source-collection "$PAPER_COLLECTION" \
    --stats-collection "$STATS_COLLECTION" \
    --read-batch-size "$STATS_READ_BATCH_SIZE" \
    --write-batch-size "$STATS_WRITE_BATCH_SIZE" \
    --replace
fi

if [[ "$EXISTING_CANDIDATES" -gt 0 ]]; then
  log "Existing-paper update inputs are ready. Run script/update_existing_paper_abstracts.sh to enrich/upsert them."
fi

log "Pipeline finished: $UPDATE_DIR"
