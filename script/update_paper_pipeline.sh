#!/usr/bin/env bash
# Compatibility wrapper for the in-process incremental update orchestrator.
# The orchestrator now writes directly to paper_prod; paper_new is no longer used.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "$PYTHON_BIN" script/update_papers.py "$@"
