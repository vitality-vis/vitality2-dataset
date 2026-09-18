#!/usr/bin/env python3
"""Export paper_new rows with fullpaper_status for fullpaper fetching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, TextIO

try:
    from create_zilliz_collection import PROJECT_ROOT
    from upload_papers_to_zilliz import DEFAULT_COLLECTION, connect_collection, normalize_text
except ModuleNotFoundError:
    from script.create_zilliz_collection import PROJECT_ROOT
    from script.upload_papers_to_zilliz import DEFAULT_COLLECTION, connect_collection, normalize_text


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "papers" / "fullpaper_candidates.json"
DEFAULT_BATCH_SIZE = 500
STATUS_FIELD = "fullpaper_status"
VECTOR_LOAD_FIELD = "search_sparse"
OUTPUT_FIELDS = [
    "paper_uid",
    "dblp_key",
    "doi",
    "title",
    "abstract",
    "authors",
    "year",
    STATUS_FIELD,
]


def iter_rows(collection, batch_size: int, timeout: float | None) -> Iterable[dict[str, Any]]:
    iterator = collection.query_iterator(batch_size=batch_size, expr="", output_fields=OUTPUT_FIELDS, timeout=timeout)
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            yield from batch
    finally:
        iterator.close()


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


def wanted_status(row_status: str, statuses: set[str]) -> bool:
    status = row_status or "not_searched"
    return not statuses or status in statuses


def clean_row(row: dict[str, Any]) -> dict[str, Any]:
    cleaned = {field: row.get(field) for field in OUTPUT_FIELDS if field in row}
    cleaned[STATUS_FIELD] = normalize_text(row.get(STATUS_FIELD)) or "not_searched"
    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fullpaper candidates from paper_new.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--query-timeout", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--status",
        action="append",
        default=[],
        help="Export only rows with this fullpaper_status. Repeatable. Missing status is treated as not_searched.",
    )
    parser.add_argument("--load", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    collection = connect_collection(args.collection)
    if args.load:
        collection.load(load_fields=["paper_uid", "doi", "title", "abstract", VECTOR_LOAD_FIELD])

    statuses = {normalize_text(status) for status in args.status if normalize_text(status)}
    scanned = 0
    exported = 0
    counts: dict[str, int] = {}
    writer = JsonArrayWriter(args.output)
    try:
        for row in iter_rows(collection, args.batch_size, args.query_timeout):
            scanned += 1
            cleaned = clean_row(row)
            status = cleaned[STATUS_FIELD]
            counts[status] = counts.get(status, 0) + 1
            if not wanted_status(status, statuses):
                continue
            writer.write(cleaned)
            exported += 1
            if args.limit is not None and exported >= args.limit:
                break
    finally:
        writer.close()

    print(
        json.dumps(
            {
                "collection": args.collection,
                "output": str(args.output),
                "scanned_rows": scanned,
                "exported_rows": exported,
                "status_filter": sorted(statuses),
                "status_counts": counts,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
