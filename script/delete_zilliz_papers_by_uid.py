#!/usr/bin/env python3
"""Delete paper records from a Zilliz collection by paper_uid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from create_zilliz_collection import DEFAULT_COLLECTION, PROJECT_ROOT, load_dotenv_file
except ModuleNotFoundError:
    from script.create_zilliz_collection import DEFAULT_COLLECTION, PROJECT_ROOT, load_dotenv_file


def quote_uid(value: str) -> str:
    return json.dumps(value)


def uid_expr(uids: list[str]) -> str:
    if not uids:
        raise ValueError("No paper_uid values provided.")
    return f"paper_uid in [{', '.join(quote_uid(uid) for uid in uids)}]"


def load_uids(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"UID file does not exist: {path}")
    uids: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        uid = raw_line.strip()
        if not uid or uid.startswith("#") or uid in seen:
            continue
        seen.add(uid)
        uids.append(uid)
    return uids


def connect_collection(collection_name: str):
    import os

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete Zilliz paper records by paper_uid.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--uid", action="append", default=[], help="paper_uid to delete. Repeatable.")
    parser.add_argument("--uid-file", type=Path, default=None, help="Text file with one paper_uid per line.")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be >= 1")

    uids = list(dict.fromkeys([*(uid.strip() for uid in args.uid if uid.strip())]))
    if args.uid_file is not None:
        uids = list(dict.fromkeys([*uids, *load_uids(args.uid_file)]))
    if not uids:
        raise SystemExit("No paper_uid values provided.")

    collection = connect_collection(args.collection)
    deleted = 0
    for start in range(0, len(uids), args.chunk_size):
        chunk = uids[start : start + args.chunk_size]
        expr = uid_expr(chunk)
        print(f"Deleting {len(chunk)} rows with expr: {expr}", flush=True)
        if args.dry_run:
            continue
        result = collection.delete(expr)
        deleted += int(getattr(result, "delete_count", 0) or 0)
    if not args.dry_run:
        collection.flush()
    print(f"Requested delete rows: {len(uids)}", flush=True)
    print(f"Deleted rows reported: {deleted}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
