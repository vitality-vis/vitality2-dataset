#!/usr/bin/env python3
"""Export existing paper_uid values from a Zilliz paper collection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

try:
    from create_zilliz_collection import PROJECT_ROOT, load_dotenv_file
except ModuleNotFoundError:
    from script.create_zilliz_collection import PROJECT_ROOT, load_dotenv_file


DEFAULT_COLLECTION = "paper_new"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "zilliz" / "paper_new_paper_uids.txt"
DEFAULT_BATCH_SIZE = 5000
# Zilliz rejects scalar-only load_fields for this collection, so include the
# sparse vector field and avoid loading the embedding field.
LOAD_FIELDS = ["paper_uid", "search_sparse"]


def normalize_uid(value: Any) -> str:
    return str(value or "").strip()


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


def iter_paper_uids(collection, batch_size: int, timeout: float | None) -> Iterable[str]:
    iterator = collection.query_iterator(
        batch_size=batch_size,
        expr="",
        output_fields=["paper_uid"],
        timeout=timeout,
    )
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            for row in batch:
                uid = normalize_uid(row.get("paper_uid"))
                if uid:
                    yield uid
    finally:
        iterator.close()


def write_uid_file(path: Path, uids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for uid in sorted(uids):
            handle.write(uid)
            handle.write("\n")
    tmp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper_uid values from Zilliz.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--query-timeout", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=None, help="Optional UID limit for testing.")
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load paper_uid plus the lightweight search_sparse vector field before querying.",
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

    collection = connect_collection(args.collection)
    if args.load:
        print(f"Loading fields for UID export: {', '.join(LOAD_FIELDS)}", flush=True)
        try:
            collection.load(load_fields=LOAD_FIELDS)
        except TypeError:
            raise SystemExit(
                "Installed pymilvus does not support load_fields. "
                "Create/load the sparse index first with: python3 script/index_zilliz_collection.py --only search_sparse --collection "
                f"{args.collection}"
            )
        except Exception as exc:
            message = str(exc)
            if "there is no vector index" in message or "does not contain vector field" in message:
                raise SystemExit(
                    "Could not load the fields needed for UID export. "
                    "Create/load the sparse index first with: python3 script/index_zilliz_collection.py --only search_sparse --collection "
                    f"{args.collection}"
                ) from exc
            raise
        print("Collection load request completed.", flush=True)

    seen: set[str] = set()
    scanned = 0
    duplicates = 0
    print(
        f"Querying paper_uid values from {args.collection} "
        f"(batch_size={args.batch_size}, timeout={args.query_timeout:g}s)",
        flush=True,
    )
    for uid in iter_paper_uids(collection, args.batch_size, args.query_timeout):
        scanned += 1
        before = len(seen)
        seen.add(uid)
        if len(seen) == before:
            duplicates += 1
        if scanned % 100000 == 0:
            print(f"Scanned {scanned} rows; unique uids {len(seen)}", flush=True)
        if args.limit is not None and scanned >= args.limit:
            break

    write_uid_file(args.output, seen)
    manifest = {
        "collection": args.collection,
        "output": str(args.output),
        "scanned_rows": scanned,
        "unique_uids": len(seen),
        "duplicate_uids": duplicates,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
