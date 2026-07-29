#!/usr/bin/env python3
"""Upsert enriched paper records into a Zilliz collection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from create_zilliz_collection import PROJECT_ROOT, load_dotenv_file
    from upload_papers_to_zilliz import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_COLLECTION,
        build_search_text,
        connect_collection,
        iter_json_files,
        load_records,
        normalize_doi,
        normalize_text,
        validate_schema,
    )
except ModuleNotFoundError:
    from script.create_zilliz_collection import PROJECT_ROOT, load_dotenv_file
    from script.upload_papers_to_zilliz import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_COLLECTION,
        build_search_text,
        connect_collection,
        iter_json_files,
        load_records,
        normalize_doi,
        normalize_text,
        validate_schema,
    )


def has_abstract(record: dict[str, Any]) -> bool:
    return bool(str(record.get("abstract") or "").strip())


def connect_client():
    import os

    from pymilvus import MilvusClient

    load_dotenv_file(PROJECT_ROOT / ".env")
    uri = os.environ.get("ZILLIZ_URI")
    token = os.environ.get("ZILLIZ_TOKEN")
    if not uri or not token:
        raise SystemExit("Missing ZILLIZ_URI or ZILLIZ_TOKEN in environment or project .env.")
    return MilvusClient(uri=uri, token=token)


def upsert_batch(client, collection_name: str, batch: list[dict[str, Any]]) -> int:
    if not batch:
        return 0
    client.upsert(collection_name=collection_name, data=batch, partial_update=True)
    upserted = len(batch)
    batch.clear()
    return upserted


def to_update_entity(record: dict[str, Any]) -> dict[str, Any]:
    paper_uid = normalize_text(record.get("paper_uid"))
    if not paper_uid:
        raise ValueError("Enriched record is missing paper_uid; cannot partial-update Zilliz.")
    doi = normalize_doi(record.get("doi"))
    abstract = normalize_text(record.get("abstract"))
    entity: dict[str, Any] = {
        "paper_uid": paper_uid,
        "doi": doi or None,
        "abstract": abstract or None,
        "search_text": build_search_text(record),
        "has_doi": bool(doi),
        "has_abstract": bool(abstract),
    }
    if record.get("keywords"):
        entity["keywords"] = record["keywords"]
    if record.get("citation_count") is not None:
        entity["citation_count"] = record["citation_count"]
    return entity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upsert enriched paper records to Zilliz.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument(
        "--papers-dir",
        type=Path,
        required=True,
        help="Directory containing enriched/ and optionally missing/. Only enriched/*.json is upserted by default.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--include-missing", action="store_true", help="Also upsert missing/*.json records.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    collection = connect_collection(args.collection)
    validate_schema(collection)
    client = connect_client()

    input_dirs = [args.papers_dir / "enriched"]
    if args.include_missing:
        input_dirs.append(args.papers_dir / "missing")
    files = list(iter_json_files(input_dirs))
    if not files:
        print(f"No JSON files found for upsert in: {', '.join(str(path) for path in input_dirs)}")
        return 0

    scanned_rows = 0
    skipped_without_abstract = 0
    upserted_rows = 0
    batch: list[dict[str, Any]] = []

    for path in files:
        records = load_records(path)
        file_upserted = 0
        for record in records:
            scanned_rows += 1
            if not args.include_missing and not has_abstract(record):
                skipped_without_abstract += 1
                continue
            batch.append(to_update_entity(record))
            if len(batch) >= args.batch_size:
                count = upsert_batch(client, args.collection, batch)
                file_upserted += count
                upserted_rows += count
                print(f"Upserted batch: total {upserted_rows}, scanned {scanned_rows}", flush=True)
        print(f"{path}: scanned {len(records)}, upserted {file_upserted}", flush=True)

    upserted_rows += upsert_batch(client, args.collection, batch)
    collection.flush()
    print(f"Scanned rows: {scanned_rows}", flush=True)
    print(f"Skipped without abstract: {skipped_without_abstract}", flush=True)
    print(f"Upserted rows: {upserted_rows}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
