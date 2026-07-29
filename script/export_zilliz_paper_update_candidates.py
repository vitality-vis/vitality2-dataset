#!/usr/bin/env python3
"""Export DBLP keys and existing missing-abstract candidates from Zilliz."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, TextIO

try:
    from create_zilliz_collection import PROJECT_ROOT, load_dotenv_file
except ModuleNotFoundError:
    from script.create_zilliz_collection import PROJECT_ROOT, load_dotenv_file


DEFAULT_COLLECTION = "paper_new"
DEFAULT_KEYS_OUTPUT = PROJECT_ROOT / "data" / "zilliz" / "paper_new_dblp_keys.txt"
DEFAULT_DOIS_OUTPUT = PROJECT_ROOT / "data" / "zilliz" / "paper_new_dois.txt"
DEFAULT_METADATA_OUTPUT = PROJECT_ROOT / "data" / "zilliz" / "paper_new_update_metadata.jsonl"
DEFAULT_CANDIDATES_DIR = PROJECT_ROOT / "data" / "papers" / "existing_missing_abstract" / "split_source"
DEFAULT_BATCH_SIZE = 5000
VECTOR_LOAD_FIELD = "search_sparse"
PRIMARY_KEY_FIELD = "paper_uid"
EXPORT_FIELDS = ["paper_uid", "dblp_key", "doi", "year", "has_doi", "has_abstract"]
STATIC_LOAD_FIELDS = [
    field for field in EXPORT_FIELDS if field not in {"has_doi", "has_abstract"}
]


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_doi(value: Any) -> str:
    doi = normalize_text(value)
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


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return bool(value)


def normalize_year(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def safe_filename(source: Any) -> str:
    safe = re.sub(r"[^A-Za-z0-9._ -]+", "_", normalize_text(source))
    safe = re.sub(r"\s+", " ", safe).strip(" ._")
    return f"{safe or 'Unknown'}.json"


def json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    if type(value).__name__ in {"RepeatedScalarContainer", "RepeatedCompositeContainer"}:
        return list(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class JsonArrayWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: TextIO | None = None
        self.count = 0

    def write(self, item: dict[str, Any]) -> None:
        if self.handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("w", encoding="utf-8")
            self.handle.write("[\n")
        else:
            self.handle.write(",\n")
        json.dump(item, self.handle, ensure_ascii=False, separators=(",", ":"), default=json_default)
        self.count += 1

    def close(self) -> None:
        if self.handle is None:
            return
        self.handle.write("\n]\n")
        self.handle.close()
        self.handle = None


def minimal_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_uid": normalize_text(row.get("paper_uid")),
        "dblp_key": normalize_text(row.get("dblp_key")),
        "doi": normalize_doi(row.get("doi")),
        "year": normalize_year(row.get("year")),
        "abstract": "",
        "source": "Unknown",
        "dblp_source": "Unknown",
    }


def connect_collection(collection_name: str):
    from pymilvus import Collection, connections, utility

    load_dotenv_file(PROJECT_ROOT / ".env")
    uri = os.environ.get("ZILLIZ_URI")
    token = os.environ.get("ZILLIZ_TOKEN")
    if not uri or not token:
        raise SystemExit("Missing ZILLIZ_URI or ZILLIZ_TOKEN in environment or project .env.")

    connections.connect(uri=uri, token=token)
    if not utility.has_collection(collection_name):
        raise SystemExit(f"Collection does not exist: {collection_name}")
    return Collection(collection_name)


def iter_rows(collection, batch_size: int, timeout: float | None) -> Iterable[dict[str, Any]]:
    iterator = collection.query_iterator(
        batch_size=batch_size,
        expr="",
        output_fields=EXPORT_FIELDS,
        timeout=timeout,
    )
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            yield from batch
    finally:
        iterator.close()


def clean_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {field: row.get(field) for field in EXPORT_FIELDS if field in row}
    out["year"] = normalize_year(row.get("year"))
    out["has_doi"] = as_bool(row.get("has_doi")) if "has_doi" in row else has_value(row.get("doi"))
    out["has_abstract"] = as_bool(row.get("has_abstract")) if "has_abstract" in row else False
    return out


def prepare_candidates_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"Candidates directory is not empty: {path}. Use --overwrite.")
    if overwrite and path.exists():
        for child in path.glob("*.json"):
            child.unlink()
    path.mkdir(parents=True, exist_ok=True)
    (path.parent / "enriched").mkdir(parents=True, exist_ok=True)
    (path.parent / "missing").mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export existing DBLP keys and DOI-without-abstract papers from Zilliz."
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--keys-output", type=Path, default=DEFAULT_KEYS_OUTPUT)
    parser.add_argument("--dois-output", type=Path, default=DEFAULT_DOIS_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--candidates-dir", type=Path, default=DEFAULT_CANDIDATES_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--query-timeout", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=None, help="Optional scanned-row limit for testing.")
    parser.add_argument(
        "--candidate-year",
        type=int,
        default=dt.datetime.now().year,
        help="Only export existing DOI/no-abstract candidates from this year. Defaults to current year.",
    )
    parser.add_argument(
        "--all-candidate-years",
        action="store_true",
        help="Export existing DOI/no-abstract candidates from all years.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load scalar fields plus search_sparse before querying.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.query_timeout <= 0:
        raise SystemExit("--query-timeout must be > 0")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    prepare_candidates_dir(args.candidates_dir, args.overwrite)
    args.keys_output.parent.mkdir(parents=True, exist_ok=True)
    args.dois_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)

    collection = connect_collection(args.collection)
    if args.load:
        load_fields = list(dict.fromkeys([PRIMARY_KEY_FIELD, *STATIC_LOAD_FIELDS, VECTOR_LOAD_FIELD]))
        print(f"Loading fields for export: {', '.join(load_fields)}", flush=True)
        collection.load(load_fields=load_fields)
        print("Collection load request completed.", flush=True)

    scanned_rows = 0
    exported_keys: set[str] = set()
    exported_dois: set[str] = set()
    candidate_rows = 0
    writers: dict[str, JsonArrayWriter] = {}

    with args.metadata_output.open("w", encoding="utf-8") as metadata:
        for row in iter_rows(collection, args.batch_size, args.query_timeout):
            scanned_rows += 1
            cleaned = clean_row(row)

            dblp_key = normalize_text(cleaned.get("dblp_key"))
            if dblp_key:
                exported_keys.add(dblp_key)
            doi = normalize_doi(cleaned.get("doi"))
            if doi:
                exported_dois.add(doi)

            metadata.write(json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"), default=json_default))
            metadata.write("\n")

            candidate_year_ok = args.all_candidate_years or cleaned.get("year") == args.candidate_year
            if cleaned["has_doi"] and not cleaned["has_abstract"] and candidate_year_ok:
                filename = safe_filename("Unknown")
                writer = writers.get(filename)
                if writer is None:
                    writer = JsonArrayWriter(args.candidates_dir / filename)
                    writers[filename] = writer
                writer.write(minimal_candidate(cleaned))
                candidate_rows += 1

            if scanned_rows % 100000 == 0:
                print(
                    f"Scanned {scanned_rows}; DBLP keys {len(exported_keys)}; DOIs {len(exported_dois)}; "
                    f"existing DOI/no-abstract candidates {candidate_rows}",
                    flush=True,
                )
            if args.limit is not None and scanned_rows >= args.limit:
                break

    for writer in writers.values():
        writer.close()

    tmp_keys = args.keys_output.with_suffix(args.keys_output.suffix + ".tmp")
    with tmp_keys.open("w", encoding="utf-8") as handle:
        for key in sorted(exported_keys):
            handle.write(key)
            handle.write("\n")
    tmp_keys.replace(args.keys_output)

    tmp_dois = args.dois_output.with_suffix(args.dois_output.suffix + ".tmp")
    with tmp_dois.open("w", encoding="utf-8") as handle:
        for doi in sorted(exported_dois):
            handle.write(doi)
            handle.write("\n")
    tmp_dois.replace(args.dois_output)

    manifest = {
        "collection": args.collection,
        "keys_output": str(args.keys_output),
        "dois_output": str(args.dois_output),
        "metadata_output": str(args.metadata_output),
        "candidates_dir": str(args.candidates_dir),
        "scanned_rows": scanned_rows,
        "dblp_keys": len(exported_keys),
        "dois": len(exported_dois),
        "candidate_year": None if args.all_candidate_years else args.candidate_year,
        "existing_doi_missing_abstract": candidate_rows,
        "candidate_files": {name: writer.count for name, writer in sorted(writers.items())},
    }
    manifest_path = args.candidates_dir.parent / "existing_missing_abstract_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
