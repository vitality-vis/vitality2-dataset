#!/usr/bin/env python3
"""Add/backfill authors_normalised for a Zilliz paper collection.

The backfill uses partial upsert with only:

    paper_uid, authors_normalised

so it does not rewrite embedding, search_text, or other production fields.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Iterable

try:
    from create_zilliz_collection import PROJECT_ROOT, load_dotenv_file
except ModuleNotFoundError:
    from script.create_zilliz_collection import PROJECT_ROOT, load_dotenv_file


DEFAULT_COLLECTION = "paper_new"
FIELD_NAME = "authors_normalised"
VECTOR_LOAD_FIELD = "search_sparse"


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalise_author_name(value: Any) -> str:
    text = normalize_text(value).casefold()
    text = re.sub(r"\s+\d{4}$", "", text).strip()
    return text[:512]


def build_authors_normalised(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]

    normalised: list[str] = []
    seen: set[str] = set()
    for item in items[:256]:
        text = normalise_author_name(item)
        if text and text not in seen:
            seen.add(text)
            normalised.append(text)
    return normalised or None


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
    return Collection(collection_name), uri, token


def add_field_if_missing(collection_name: str, uri: str, token: str) -> bool:
    from pymilvus import Collection, DataType, MilvusClient

    collection = Collection(collection_name)
    if FIELD_NAME in {field.name for field in collection.schema.fields}:
        print(f"Field already exists: {FIELD_NAME}", flush=True)
        return False

    client = MilvusClient(uri=uri, token=token)
    client.add_collection_field(
        collection_name=collection_name,
        field_name=FIELD_NAME,
        data_type=DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=256,
        max_length=512,
        nullable=True,
    )
    print(f"Added field: {FIELD_NAME}", flush=True)
    return True


def iter_rows(collection, expr: str, batch_size: int, timeout: float | None, limit: int | None) -> Iterable[dict[str, Any]]:
    scanned = 0
    iterator = collection.query_iterator(
        batch_size=batch_size,
        expr=expr,
        output_fields=["paper_uid", "authors"],
        timeout=timeout,
    )
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            for row in batch:
                scanned += 1
                yield row
                if limit is not None and scanned >= limit:
                    return
    finally:
        iterator.close()


def upsert_batch(collection, batch: list[dict[str, Any]]) -> int:
    if not batch:
        return 0
    collection.upsert(batch, partial_update=True)
    count = len(batch)
    batch.clear()
    return count


def backfill(collection, args: argparse.Namespace) -> int:
    expr = "" if args.all_rows else f"{FIELD_NAME} is null"
    scanned = 0
    updated = 0
    batch: list[dict[str, Any]] = []
    for row in iter_rows(collection, expr, args.read_batch_size, args.query_timeout, args.limit):
        scanned += 1
        if scanned % args.progress_every == 0:
            print(f"Scanned {scanned}; updated {updated}", flush=True)
        batch.append(
            {
                "paper_uid": row["paper_uid"],
                FIELD_NAME: build_authors_normalised(row.get("authors")),
            }
        )
        if len(batch) >= args.write_batch_size:
            updated += upsert_batch(collection, batch)

    updated += upsert_batch(collection, batch)
    collection.flush()
    print(f"Scanned rows: {scanned}", flush=True)
    print(f"Backfilled rows: {updated}", flush=True)
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add and backfill authors_normalised in Zilliz.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--read-batch-size", type=int, default=1000)
    parser.add_argument("--write-batch-size", type=int, default=500)
    parser.add_argument("--query-timeout", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--skip-add-field", action="store_true")
    parser.add_argument("--skip-backfill", action="store_true")
    parser.add_argument("--all-rows", action="store_true", help="Backfill every row, not only NULL values.")
    parser.add_argument("--execute", action="store_true", help="Apply changes. Default is dry-run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    collection, uri, token = connect_collection(args.collection)
    field_exists = FIELD_NAME in {field.name for field in collection.schema.fields}

    if not args.execute:
        print(
            json.dumps(
                {
                    "collection": args.collection,
                    "execute": False,
                    "field_exists": field_exists,
                    "sample": {
                        "input": ["John Smith 0001", "Mary Ann Jones", "Wei Zhang 0012"],
                        "output": build_authors_normalised(
                            ["John Smith 0001", "Mary Ann Jones", "Wei Zhang 0012"]
                        ),
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    if not args.skip_add_field:
        added = add_field_if_missing(args.collection, uri, token)
        if added:
            collection, _, _ = connect_collection(args.collection)

    if not args.skip_backfill:
        collection.load(load_fields=["paper_uid", "authors", FIELD_NAME, VECTOR_LOAD_FIELD])
        backfill(collection, args)

    fields = [field.name for field in collection.schema.fields]
    print(json.dumps({"collection": args.collection, "field_present": FIELD_NAME in fields}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
