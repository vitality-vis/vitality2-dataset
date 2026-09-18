#!/usr/bin/env python3
"""Backfill paper_new dynamic metadata into paper_prod.

This script is intentionally conservative:
  - dry-run by default;
  - partial updates only target paper_uids already present in paper_prod.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    BATCH_SIZE,
    DEV_COLLECTION,
    PROD_COLLECTION,
    QUERY_TIMEOUT_SECONDS,
    connect_zilliz,
    uid_in_expr,
)
from create_prod_collection import _collection_field_names  # noqa: E402


SOURCE_REQUIRED_FIELDS = ["paper_uid", "doi", "abstract"]
DYNAMIC_BACKFILL_FIELDS = [
    "has_doi",
    "has_abstract",
    "fullpaper_status",
    "fullpaper_arxiv_id",
    "fullpaper_arxiv_method",
    "fullpaper_error",
    "fullpaper_md_bytes",
    "fullpaper_chunks",
    "fullpaper_chunk_count",
]


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def existing_collections(client) -> set[str]:
    return {str(name) for name in client.list_collections()}


def source_output_fields(client, source_collection: str) -> list[str]:
    # Dynamic fields do not appear in the fixed schema, but Milvus/Zilliz can
    # return them by name when they exist in row metadata.
    return [*SOURCE_REQUIRED_FIELDS, *DYNAMIC_BACKFILL_FIELDS]


def collection_dynamic_enabled(client, collection_name: str) -> bool:
    desc = client.describe_collection(collection_name)
    for key in ("enable_dynamic_field", "enableDynamicField"):
        if key in desc:
            return bool(desc[key])
    schema = desc.get("schema")
    if isinstance(schema, dict):
        for key in ("enable_dynamic_field", "enableDynamicField"):
            if key in schema:
                return bool(schema[key])
    return False


def iter_source_batches(
    client,
    collection_name: str,
    output_fields: list[str],
    *,
    batch_size: int,
    timeout: float,
    limit: int | None,
) -> Iterator[list[dict[str, Any]]]:
    scanned = 0
    iterator = client.query_iterator(
        collection_name=collection_name,
        batch_size=batch_size,
        filter='paper_uid != ""',
        output_fields=output_fields,
        timeout=timeout,
    )
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            if limit is not None:
                remaining = limit - scanned
                if remaining <= 0:
                    break
                batch = batch[:remaining]
            scanned += len(batch)
            yield batch
            if limit is not None and scanned >= limit:
                break
    finally:
        iterator.close()


def prod_existing_uids(client, collection_name: str, uids: list[str]) -> set[str]:
    found: set[str] = set()
    for start in range(0, len(uids), BATCH_SIZE):
        chunk = uids[start : start + BATCH_SIZE]
        if not chunk:
            continue
        rows = client.query(
            collection_name=collection_name,
            filter=uid_in_expr(chunk),
            output_fields=["paper_uid"],
            limit=len(chunk) + 10,
        )
        found.update(str(row["paper_uid"]) for row in rows if row.get("paper_uid"))
    return found


def build_backfill_entity(row: dict[str, Any]) -> dict[str, Any] | None:
    uid = normalize_text(row.get("paper_uid"))
    if not uid:
        return None

    doi = normalize_text(row.get("doi"))
    abstract = normalize_text(row.get("abstract"))
    entity: dict[str, Any] = {
        "paper_uid": uid,
        "has_doi": bool(doi),
        "has_abstract": bool(abstract),
    }
    for field_name in DYNAMIC_BACKFILL_FIELDS:
        if field_name in {"has_doi", "has_abstract"}:
            continue
        if field_name in row:
            entity[field_name] = row.get(field_name)
    return entity


def flush_batch(client, collection_name: str, batch: list[dict[str, Any]], *, execute: bool) -> int:
    if not batch:
        return 0
    count = len(batch)
    if execute:
        client.upsert(collection_name=collection_name, data=batch, partial_update=True)
    batch.clear()
    return count


def backfill(args: argparse.Namespace) -> dict[str, Any]:
    client = connect_zilliz()
    collections = existing_collections(client)
    for name in (args.source_collection, args.target_collection):
        if name not in collections:
            raise SystemExit(f"Collection does not exist: {name}")

    target_dynamic_enabled = collection_dynamic_enabled(client, args.target_collection)
    if args.execute and not target_dynamic_enabled:
        raise SystemExit(
            f"Target collection must enable dynamic fields for: {', '.join(DYNAMIC_BACKFILL_FIELDS)}."
        )

    source_fields = source_output_fields(client, args.source_collection)
    if args.execute:
        try:
            client.load_collection(args.target_collection)
        except Exception as exc:  # noqa: BLE001
            print(f"Note: could not load {args.target_collection} ({exc})", flush=True)

    scanned = 0
    matched = 0
    skipped_not_in_prod = 0
    updated = 0
    update_batch: list[dict[str, Any]] = []

    for source_batch in iter_source_batches(
        client,
        args.source_collection,
        source_fields,
        batch_size=args.read_batch_size,
        timeout=args.query_timeout,
        limit=args.limit,
    ):
        scanned += len(source_batch)
        uids = [normalize_text(row.get("paper_uid")) for row in source_batch]
        uids = [uid for uid in uids if uid]
        existing = prod_existing_uids(client, args.target_collection, uids)

        for row in source_batch:
            uid = normalize_text(row.get("paper_uid"))
            if not uid or uid not in existing:
                skipped_not_in_prod += 1
                continue
            entity = build_backfill_entity(row)
            if entity is None:
                skipped_not_in_prod += 1
                continue
            matched += 1
            update_batch.append(entity)
            if len(update_batch) >= args.write_batch_size:
                updated += flush_batch(client, args.target_collection, update_batch, execute=args.execute)

        if scanned % args.progress_every == 0:
            print(
                f"Scanned {scanned}; matched {matched}; "
                f"skipped_not_in_prod {skipped_not_in_prod}; updates {updated}",
                flush=True,
            )

    updated += flush_batch(client, args.target_collection, update_batch, execute=args.execute)
    if args.execute:
        client.flush(args.target_collection)

    return {
        "execute": args.execute,
        "source_collection": args.source_collection,
        "target_collection": args.target_collection,
        "source_output_fields": source_fields,
        "target_dynamic_enabled": target_dynamic_enabled,
        "dynamic_backfill_fields": DYNAMIC_BACKFILL_FIELDS,
        "scanned_source_rows": scanned,
        "matched_existing_prod_rows": matched,
        "skipped_not_in_prod": skipped_not_in_prod,
        "partial_updates": updated,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge paper_new-only fields into paper_prod.")
    parser.add_argument("--source-collection", default=DEV_COLLECTION)
    parser.add_argument("--target-collection", default=PROD_COLLECTION)
    parser.add_argument("--read-batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--write-batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--query-timeout", type=float, default=QUERY_TIMEOUT_SECONDS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--execute", action="store_true", help="Apply dynamic-field partial updates.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.read_batch_size < 1 or args.write_batch_size < 1:
        raise SystemExit("Batch sizes must be >= 1.")
    result = backfill(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
    if not args.execute:
        print("Dry run only. Re-run with --execute to alter schema and backfill fields.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
