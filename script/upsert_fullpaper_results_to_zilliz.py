#!/usr/bin/env python3
"""Partial-upsert fullpaper fetch results into paper_new."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

try:
    from create_zilliz_collection import PROJECT_ROOT
    from upload_papers_to_zilliz import DEFAULT_BATCH_SIZE, DEFAULT_COLLECTION, connect_collection, normalize_text
    from upsert_enriched_papers_to_zilliz import connect_client, upsert_batch
except ModuleNotFoundError:
    from script.create_zilliz_collection import PROJECT_ROOT
    from script.upload_papers_to_zilliz import DEFAULT_BATCH_SIZE, DEFAULT_COLLECTION, connect_collection, normalize_text
    from script.upsert_enriched_papers_to_zilliz import connect_client, upsert_batch


STATUS_FIELD = "fullpaper_status"
FINAL_STATUSES = {"found", "not_found", "not_applicable", "failed", "failed_s2", "failed_arxivsearch"}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            yield row


def resolve_path(value: Any) -> Path | None:
    text = normalize_text(value)
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def to_entity(row: dict[str, Any], *, include_md: bool, max_md_bytes: int | None) -> dict[str, Any] | None:
    if row.get("skipped"):
        return None
    paper_uid = normalize_text(row.get("paper_uid"))
    status = normalize_text(row.get(STATUS_FIELD))
    if not paper_uid or not status:
        return None
    if status == "not_searched":
        return None
    if status not in FINAL_STATUSES:
        raise ValueError(f"Unsupported {STATUS_FIELD}: {status}")

    entity: dict[str, Any] = {"paper_uid": paper_uid, STATUS_FIELD: status}
    for key in ("arxiv_id", "arxiv_method", "error"):
        value = normalize_text(row.get(key))
        if value:
            entity[f"fullpaper_{key}" if key.startswith("arxiv_") else "fullpaper_error"] = value

    if include_md and status == "found":
        md_path = resolve_path(row.get("fullpaper_md_path"))
        if md_path is None:
            raise ValueError(f"Found result is missing fullpaper_md_path for {paper_uid}")
        md = md_path.read_text(encoding="utf-8")
        md_bytes = len(md.encode("utf-8"))
        if max_md_bytes is not None and md_bytes > max_md_bytes:
            raise ValueError(f"Markdown too large for {paper_uid}: {md_bytes} > {max_md_bytes}")
        entity["fullpaper_md"] = md
        entity["fullpaper_md_bytes"] = md_bytes
    elif row.get("fullpaper_md_bytes") is not None:
        entity["fullpaper_md_bytes"] = int(row["fullpaper_md_bytes"])

    return entity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upsert fullpaper result JSONL rows into paper_new.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--include-md", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-md-bytes",
        type=int,
        default=None,
        help="Refuse to upsert Markdown larger than this. Default has no local size cap.",
    )
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

    scanned = 0
    skipped = 0
    prepared = 0
    upserted = 0
    status_counts: dict[str, int] = {}
    batch: list[dict[str, Any]] = []

    for row in iter_jsonl(args.results):
        scanned += 1
        if args.limit is not None and scanned > args.limit:
            break
        entity = to_entity(row, include_md=args.include_md, max_md_bytes=args.max_md_bytes)
        if entity is None:
            skipped += 1
            continue
        status_counts[entity[STATUS_FIELD]] = status_counts.get(entity[STATUS_FIELD], 0) + 1
        prepared += 1
        if args.dry_run:
            continue
        batch.append(entity)
        if len(batch) >= args.batch_size:
            upserted += upsert_batch(client, args.collection, batch)
            print(f"Upserted {upserted}; scanned {scanned}", flush=True)

    if not args.dry_run:
        upserted += upsert_batch(client, args.collection, batch)

    print(
        json.dumps(
            {
                "collection": args.collection,
                "dry_run": args.dry_run,
                "results": str(args.results),
                "scanned_rows": scanned,
                "skipped_rows": skipped,
                "prepared_rows": prepared,
                "upserted_rows": upserted,
                "status_counts": status_counts,
                "include_md": args.include_md,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
