#!/usr/bin/env bash
# Enrich and partial-update existing paper_new records that already have DOI but
# still lack abstracts. Run after script/update_paper_pipeline.sh prepares
# EXISTING_UPDATE_DIR/split_source.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

UPDATE_DATE="${UPDATE_DATE:-$(date +%Y%m%d)}"
EXISTING_UPDATE_DIR="${EXISTING_UPDATE_DIR:-data/papers/update_missing${UPDATE_DATE}}"

PAPER_COLLECTION="${PAPER_COLLECTION:-paper_new}"
STATS_COLLECTION="${STATS_COLLECTION:-paper_stats}"

RUN_EXISTING_ENRICH="${RUN_EXISTING_ENRICH:-1}"
RUN_EXISTING_UPSERT="${RUN_EXISTING_UPSERT:-1}"
RUN_STATS="${RUN_STATS:-1}"

UPLOAD_BATCH_SIZE="${UPLOAD_BATCH_SIZE:-500}"
STATS_READ_BATCH_SIZE="${STATS_READ_BATCH_SIZE:-5000}"
STATS_WRITE_BATCH_SIZE="${STATS_WRITE_BATCH_SIZE:-500}"

OPENALEX_WORKERS="${OPENALEX_WORKERS:-16}"
OPENALEX_MAX_PENDING="${OPENALEX_MAX_PENDING:-128}"
OPENALEX_SLEEP="${OPENALEX_SLEEP:-0}"

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

has_non_missing_doi_missing_files_in() {
  local papers_dir="$1"
  find "$papers_dir/missing" -maxdepth 1 -type f -name '*.json' ! -name '_missing_doi.json' -size +0c \
    | grep -q .
}

has_split_source_files() {
  local papers_dir="$1"
  find "$papers_dir/split_source" -maxdepth 1 -type f -name '*.json' -size +0c | grep -q .
}

run_three_layer_enrichment() {
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

EXISTING_CANDIDATES="$(json_number "$EXISTING_UPDATE_DIR/existing_missing_abstract_manifest.json" existing_doi_missing_abstract)"
log "Existing missing-abstract directory: $EXISTING_UPDATE_DIR"
log "Existing DOI/no-abstract candidates: $EXISTING_CANDIDATES"

if [[ "$EXISTING_CANDIDATES" -eq 0 ]] || ! has_split_source_files "$EXISTING_UPDATE_DIR"; then
  log "No existing-paper candidates found. Nothing to enrich or upsert."
  exit 0
fi

if [[ "$RUN_EXISTING_ENRICH" == "1" ]]; then
  run_three_layer_enrichment "$EXISTING_UPDATE_DIR" "$EXISTING_UPDATE_DIR/split_source" "existing_" "--overwrite"
fi

if [[ "$RUN_EXISTING_UPSERT" == "1" ]]; then
  run "$PYTHON_BIN" script/upsert_enriched_papers_to_zilliz.py \
    --collection "$PAPER_COLLECTION" \
    --papers-dir "$EXISTING_UPDATE_DIR" \
    --batch-size "$UPLOAD_BATCH_SIZE"
fi

if [[ "$RUN_STATS" == "1" ]]; then
  run "$PYTHON_BIN" script/materialize_paper_stats.py \
    --source-collection "$PAPER_COLLECTION" \
    --stats-collection "$STATS_COLLECTION" \
    --read-batch-size "$STATS_READ_BATCH_SIZE" \
    --write-batch-size "$STATS_WRITE_BATCH_SIZE" \
    --replace
fi

log "Existing-paper update finished: $EXISTING_UPDATE_DIR"
