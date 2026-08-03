#!/usr/bin/env python3
"""Create and populate the paper_exclude Zilliz collection."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable

try:
    from create_zilliz_collection import PROJECT_ROOT, load_dotenv_file
except ModuleNotFoundError:
    from script.create_zilliz_collection import PROJECT_ROOT, load_dotenv_file


DEFAULT_COLLECTION = "paper_exclude"
DEFAULT_MATCHES = PROJECT_ROOT / "data" / "zilliz" / "paper_new_exclude_title_matches.csv"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "data" / "dblp" / "split_source"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "zilliz" / "paper_exclude_seed_rows.json"
DEFAULT_BATCH_SIZE = 500
EXCLUDE_VECTOR = [0.0, 0.0]


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def load_target_keys(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Matches CSV does not exist: {path}")
    keys: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = normalize_text(row.get("dblp_key"))
            if key and key not in seen:
                seen.add(key)
                keys.append(key)
    if not keys:
        raise SystemExit(f"No dblp_key values found in: {path}")
    return keys


def iter_split_records(split_dir: Path) -> Iterable[dict[str, Any]]:
    if not split_dir.exists():
        raise SystemExit(f"Split source directory does not exist: {split_dir}")
    for path in sorted(split_dir.glob("*.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise SystemExit(f"Expected JSON array in: {path}")
        for record in records:
            if isinstance(record, dict):
                yield record


def build_seed_rows(target_keys: list[str], split_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    remaining = set(target_keys)
    by_key: dict[str, dict[str, Any]] = {}
    for record in iter_split_records(split_dir):
        dblp_key = normalize_text(record.get("dblp_key"))
        if dblp_key not in remaining:
            continue
        by_key[dblp_key] = {
            "key": dblp_key,
            "dblp_key": dblp_key,
            "doi": normalize_text(record.get("doi")),
            "title": normalize_text(record.get("title")),
            "_exclude_vector": EXCLUDE_VECTOR,
        }
        remaining.remove(dblp_key)
        if not remaining:
            break
    rows = [by_key[key] for key in target_keys if key in by_key]
    missing = [key for key in target_keys if key not in by_key]
    return rows, missing


def connect():
    from pymilvus import connections

    load_dotenv_file(PROJECT_ROOT / ".env")
    uri = os.environ.get("ZILLIZ_URI")
    token = os.environ.get("ZILLIZ_TOKEN")
    if not uri or not token:
        raise SystemExit("Missing ZILLIZ_URI or ZILLIZ_TOKEN in environment or project .env.")
    connections.connect(uri=uri, token=token)


def create_schema():
    from pymilvus import CollectionSchema, DataType, FieldSchema

    fields = [
        FieldSchema(name="key", dtype=DataType.VARCHAR, max_length=1024, is_primary=True),
        FieldSchema(name="dblp_key", dtype=DataType.VARCHAR, max_length=1024, nullable=True),
        FieldSchema(name="doi", dtype=DataType.VARCHAR, max_length=512, nullable=True),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="_exclude_vector", dtype=DataType.FLOAT_VECTOR, dim=2),
    ]
    return CollectionSchema(
        fields=fields,
        description="Excluded paper keys with DOI and title only.",
        enable_dynamic_field=False,
    )


def get_or_create_collection(name: str):
    from pymilvus import Collection, utility

    if utility.has_collection(name):
        return Collection(name), False
    collection = Collection(name=name, schema=create_schema())
    collection.create_index(
        field_name="_exclude_vector",
        index_params={"index_type": "AUTOINDEX", "metric_type": "L2", "params": {}},
    )
    return collection, True


def quote_key(value: str) -> str:
    return json.dumps(value)


def existing_keys(collection, keys: list[str], chunk_size: int) -> set[str]:
    found: set[str] = set()
    for start in range(0, len(keys), chunk_size):
        chunk = keys[start : start + chunk_size]
        expr = f"key in [{', '.join(quote_key(key) for key in chunk)}]"
        for row in collection.query(expr=expr, output_fields=["key"]):
            key = normalize_text(row.get("key"))
            if key:
                found.add(key)
    return found


def insert_batches(collection, rows: list[dict[str, Any]], batch_size: int) -> int:
    inserted = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        collection.insert(batch)
        inserted += len(batch)
    if rows:
        collection.flush()
    return inserted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and seed paper_exclude.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    target_keys = load_target_keys(args.matches)
    rows, missing = build_seed_rows(target_keys, args.split_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Target keys: {len(target_keys)}", flush=True)
    print(f"Rows built from split_source: {len(rows)}", flush=True)
    print(f"Missing in split_source: {len(missing)}", flush=True)
    print(f"Seed rows written: {args.output}", flush=True)
    if missing:
        print("Missing keys:", flush=True)
        for key in missing:
            print(f" - {key}", flush=True)
    if not args.execute:
        print("Dry run only. Re-run with --execute to create/insert into Zilliz.", flush=True)
        return 0

    connect()
    collection, created = get_or_create_collection(args.collection)
    print(f"{'Created' if created else 'Using existing'} collection: {args.collection}", flush=True)

    present: set[str] = set()
    if not created:
        collection.load()
        present = existing_keys(collection, [row["key"] for row in rows], args.batch_size)
    to_insert = [row for row in rows if row["key"] not in present]
    inserted = insert_batches(collection, to_insert, args.batch_size)

    manifest = {
        "collection": args.collection,
        "matches": str(args.matches),
        "split_dir": str(args.split_dir),
        "target_keys": len(target_keys),
        "rows_built": len(rows),
        "missing_in_split_source": missing,
        "existing_rows": len(present),
        "inserted_rows": inserted,
        "seed_output": str(args.output),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
