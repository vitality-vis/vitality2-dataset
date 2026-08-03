#!/usr/bin/env python3
"""Move papers identified by DBLP key into paper_exclude and delete them."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from create_zilliz_collection import PROJECT_ROOT, load_dotenv_file
except ModuleNotFoundError:
    from script.create_zilliz_collection import PROJECT_ROOT, load_dotenv_file


DEFAULT_SOURCE_COLLECTIONS = ("paper_new", "paper_prod")
DEFAULT_EXCLUDE_COLLECTION = "paper_exclude"
EXCLUDE_VECTOR = [0.0, 0.0]
LOOKUP_FIELDS = ["paper_uid", "dblp_key", "doi", "title", "source", "dblp_source", "year"]


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def quote(value: str) -> str:
    return json.dumps(value)


def load_keys(path: Path | None) -> list[str]:
    if path is None:
        return []
    if not path.exists():
        raise SystemExit(f"DBLP key file does not exist: {path}")
    keys: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        key = raw_line.strip()
        if not key or key.startswith("#") or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def connect():
    from pymilvus import connections

    load_dotenv_file(PROJECT_ROOT / ".env")
    uri = os.environ.get("ZILLIZ_URI")
    token = os.environ.get("ZILLIZ_TOKEN")
    if not uri or not token:
        raise SystemExit("Missing ZILLIZ_URI or ZILLIZ_TOKEN in environment or project .env.")
    connections.connect(uri=uri, token=token, timeout=30)


def get_collection(name: str):
    from pymilvus import Collection, utility

    if not utility.has_collection(name):
        return None
    return Collection(name)


def query_by_dblp_keys(collection, keys: list[str], chunk_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(keys), chunk_size):
        chunk = keys[start : start + chunk_size]
        expr = f"dblp_key in [{', '.join(quote(key) for key in chunk)}]"
        rows.extend(collection.query(expr=expr, output_fields=LOOKUP_FIELDS, timeout=60))
    return rows


def query_exclude_rows(collection_name: str, keys: list[str], chunk_size: int) -> dict[str, dict[str, Any]]:
    collection = get_collection(collection_name)
    if collection is None:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for start in range(0, len(keys), chunk_size):
        chunk = keys[start : start + chunk_size]
        expr = f"key in [{', '.join(quote(key) for key in chunk)}]"
        for row in collection.query(expr=expr, output_fields=["key", "dblp_key", "doi", "title"], timeout=60):
            key = normalize_text(row.get("key"))
            if key:
                rows[key] = row
    return rows


def choose_exclude_rows(
    keys: list[str],
    rows_by_collection: dict[str, list[dict[str, Any]]],
    existing_exclude: dict[str, dict[str, Any]],
    source_collections: tuple[str, ...],
) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for collection_name in source_collections:
        for row in rows_by_collection.get(collection_name, []):
            dblp_key = normalize_text(row.get("dblp_key"))
            if not dblp_key or dblp_key in by_key:
                continue
            by_key[dblp_key] = row

    exclude_rows: list[dict[str, Any]] = []
    for key in keys:
        row = by_key.get(key) or existing_exclude.get(key)
        exclude_rows.append(
            {
                "key": key,
                "dblp_key": key,
                "doi": normalize_text(row.get("doi")) if row else "",
                "title": normalize_text(row.get("title")) if row else "",
                "_exclude_vector": EXCLUDE_VECTOR,
            }
        )
    return exclude_rows


def upsert_exclude(collection_name: str, rows: list[dict[str, Any]], batch_size: int) -> int:
    collection = get_collection(collection_name)
    if collection is None:
        raise SystemExit(f"Exclude collection does not exist: {collection_name}")
    upserted = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        collection.upsert(batch)
        upserted += len(batch)
    if rows:
        collection.flush()
    return upserted


def delete_by_dblp_key(collection_name: str, keys: list[str], chunk_size: int) -> int:
    collection = get_collection(collection_name)
    if collection is None:
        return 0
    deleted = 0
    for start in range(0, len(keys), chunk_size):
        chunk = keys[start : start + chunk_size]
        expr = f"dblp_key in [{', '.join(quote(key) for key in chunk)}]"
        result = collection.delete(expr)
        deleted += int(getattr(result, "delete_count", 0) or 0)
    if keys:
        collection.flush()
    return deleted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Given DBLP keys, upsert key/doi/title into paper_exclude and delete from paper_new/paper_prod."
    )
    parser.add_argument("dblp_key", nargs="*", help="DBLP key to exclude. Can pass multiple.")
    parser.add_argument("--key-file", type=Path, default=None, help="Text file with one DBLP key per line.")
    parser.add_argument("--exclude-collection", default=DEFAULT_EXCLUDE_COLLECTION)
    parser.add_argument("--source-collection", action="append", default=[], help="Collection to delete from. Repeatable.")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--execute", action="store_true", help="Actually upsert paper_exclude and delete rows.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be >= 1")

    keys = list(dict.fromkeys([*(key.strip() for key in args.dblp_key if key.strip()), *load_keys(args.key_file)]))
    if not keys:
        raise SystemExit("No DBLP keys provided.")
    source_collections = tuple(args.source_collection or DEFAULT_SOURCE_COLLECTIONS)

    connect()
    rows_by_collection: dict[str, list[dict[str, Any]]] = {}
    for collection_name in source_collections:
        collection = get_collection(collection_name)
        if collection is None:
            rows_by_collection[collection_name] = []
            continue
        rows_by_collection[collection_name] = query_by_dblp_keys(collection, keys, args.chunk_size)

    existing_exclude = query_exclude_rows(args.exclude_collection, keys, args.chunk_size)
    exclude_rows = choose_exclude_rows(keys, rows_by_collection, existing_exclude, source_collections)
    found_keys = {normalize_text(row.get("dblp_key")) for rows in rows_by_collection.values() for row in rows}
    missing_keys = [key for key in keys if key not in found_keys]

    print(f"DBLP keys requested: {len(keys)}", flush=True)
    for collection_name in source_collections:
        print(f"{collection_name} matched rows: {len(rows_by_collection.get(collection_name, []))}", flush=True)
    print(f"paper_exclude rows prepared: {len(exclude_rows)}", flush=True)
    if missing_keys:
        print("Keys not found in source collections; preserving existing paper_exclude data when present:", flush=True)
        for key in missing_keys:
            status = "existing exclude row" if key in existing_exclude else "empty doi/title"
            print(f" - {key} ({status})", flush=True)

    if not args.execute:
        print("Dry run only. Re-run with --execute to upsert paper_exclude and delete source rows.", flush=True)
        print(json.dumps({"exclude_rows": exclude_rows}, ensure_ascii=False, indent=2), flush=True)
        return 0

    upserted = upsert_exclude(args.exclude_collection, exclude_rows, args.chunk_size)
    deleted_by_collection = {
        collection_name: delete_by_dblp_key(collection_name, keys, args.chunk_size)
        for collection_name in source_collections
    }
    print(
        json.dumps(
            {
                "requested_keys": len(keys),
                "exclude_collection": args.exclude_collection,
                "exclude_upserted_rows": upserted,
                "deleted_by_collection": deleted_by_collection,
                "missing_keys": missing_keys,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
