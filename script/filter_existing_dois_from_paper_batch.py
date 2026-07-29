#!/usr/bin/env python3
"""Remove papers from an enrichment batch when their DOI already exists in Zilliz.

This is intended to run after OpenAlex title search recovers DOI values for
records initially filtered as DOI-less. It also collapses duplicate DOI values
within the batch, preferring enriched records over records still missing an
abstract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def normalize_doi(value: Any) -> str:
    doi = str(value or "").strip()
    lower = doi.lower()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ):
        if lower.startswith(prefix):
            return doi[len(prefix) :].strip().casefold()
    return doi.casefold()


def load_existing_dois(path: Path) -> set[str]:
    if not path.exists():
        raise SystemExit(f"Existing DOI file does not exist: {path}")
    return {
        normalized
        for line in path.read_text(encoding="utf-8").splitlines()
        if (normalized := normalize_doi(line)) and not normalized.startswith("#")
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array: {path}")
    return data


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(records, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove recovered DOI records that already exist in Zilliz."
    )
    parser.add_argument("--papers-dir", type=Path, required=True)
    parser.add_argument("--existing-doi-file", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    existing_dois = load_existing_dois(args.existing_doi_file)
    enriched_dir = args.papers_dir / "enriched"
    missing_dir = args.papers_dir / "missing"
    paths = [
        *sorted(enriched_dir.glob("*.json")),
        *sorted(path for path in missing_dir.glob("*.json") if path.name != "_missing_doi.json"),
    ]

    seen_dois: set[str] = set()
    stats = {
        "existing_dois": len(existing_dois),
        "input_papers": 0,
        "input_papers_with_doi": 0,
        "removed_existing_doi": 0,
        "removed_duplicate_doi": 0,
        "retained_papers": 0,
        "retained_papers_with_doi": 0,
        "retained_missing_doi": 0,
        "files": {},
    }

    for path in paths:
        records = load_records(path)
        retained: list[dict[str, Any]] = []
        removed_existing = 0
        removed_duplicate = 0
        for record in records:
            stats["input_papers"] += 1
            doi = normalize_doi(record.get("doi"))
            if not doi:
                retained.append(record)
                stats["retained_missing_doi"] += 1
                continue
            stats["input_papers_with_doi"] += 1
            if doi in existing_dois:
                stats["removed_existing_doi"] += 1
                removed_existing += 1
                continue
            if doi in seen_dois:
                stats["removed_duplicate_doi"] += 1
                removed_duplicate += 1
                continue
            seen_dois.add(doi)
            retained.append(record)
            stats["retained_papers_with_doi"] += 1

        stats["retained_papers"] += len(retained)
        write_records(path, retained)
        stats["files"][str(path.relative_to(args.papers_dir))] = {
            "input": len(records),
            "retained": len(retained),
            "removed_existing_doi": removed_existing,
            "removed_duplicate_doi": removed_duplicate,
        }

    missing_doi_path = missing_dir / "_missing_doi.json"
    if missing_doi_path.exists():
        remaining_missing_doi = len(load_records(missing_doi_path))
        stats["retained_papers"] += remaining_missing_doi
        stats["retained_missing_doi"] += remaining_missing_doi

    manifest_path = args.papers_dir / "post_doi_dedupe_manifest.json"
    manifest = {
        "papers_dir": str(args.papers_dir),
        "existing_doi_file": str(args.existing_doi_file),
        **stats,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
