#!/usr/bin/env python3
"""Upload exported paper_full JSONL rows back into Zilliz."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

try:
    from upsert_enriched_papers_to_zilliz import connect_client
    from upload_papers_to_zilliz import DEFAULT_BATCH_SIZE, connect_collection
except ModuleNotFoundError:
    from script.upsert_enriched_papers_to_zilliz import connect_client
    from script.upload_papers_to_zilliz import DEFAULT_BATCH_SIZE, connect_collection


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            yield item


def to_entity(row: dict[str, Any]) -> dict[str, Any]:
    entity = dict(row)
    entity.pop("search_sparse", None)
    if "_chunk_vector" not in entity:
        entity["_chunk_vector"] = [0.0, 0.0]
    return entity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload exported paper_full chunks to Zilliz.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--collection", default="paper_full")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    connect_collection(args.collection)
    client = None if args.dry_run else connect_client()
    if client is not None:
        client.load_collection(args.collection)

    scanned = 0
    uploaded = 0
    batch: list[dict[str, Any]] = []
    max_text_bytes = 0
    for row in iter_jsonl(args.input):
        scanned += 1
        if args.limit is not None and scanned > args.limit:
            break
        entity = to_entity(row)
        max_text_bytes = max(max_text_bytes, len(str(entity.get("text") or "").encode("utf-8")))
        if args.dry_run:
            continue
        batch.append(entity)
        if len(batch) >= args.batch_size:
            client.upsert(collection_name=args.collection, data=batch)
            uploaded += len(batch)
            batch.clear()
            print(f"Uploaded {uploaded}; scanned {scanned}", flush=True)
    if not args.dry_run and batch:
        client.upsert(collection_name=args.collection, data=batch)
        uploaded += len(batch)
    print(
        json.dumps(
            {
                "collection": args.collection,
                "dry_run": args.dry_run,
                "input": str(args.input),
                "max_text_bytes": max_text_bytes,
                "scanned_rows": scanned if args.limit is None else min(scanned, args.limit),
                "uploaded_rows": uploaded,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
