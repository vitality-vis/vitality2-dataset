#!/usr/bin/env python3
"""Materialize dashboard statistics from a Zilliz paper collection into paper_stats."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from collections import Counter, defaultdict
from typing import Any, Iterable

try:
    from create_zilliz_collection import PROJECT_ROOT, load_dotenv_file
except ModuleNotFoundError:
    from script.create_zilliz_collection import PROJECT_ROOT, load_dotenv_file


DEFAULT_SOURCE_COLLECTION = "paper_new"
DEFAULT_STATS_COLLECTION = "paper_stats"
DEFAULT_BATCH_SIZE = 5000
RAW_COMPLETENESS_FIELDS = ["doi", "abstract"]
BASE_READ_FIELDS = ["source", "dblp_source", "year"]
FLAG_READ_FIELDS = [*BASE_READ_FIELDS, "has_doi", "has_abstract"]
RAW_READ_FIELDS = [*BASE_READ_FIELDS, *RAW_COMPLETENESS_FIELDS]
STATS_VECTOR = [0.0, 0.0]


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return bool(value)


def normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    return text or "Unknown"


def normalize_year(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def stat_key(stat_type: str, source_collection: str, *parts: Any) -> str:
    raw = "::".join([stat_type, source_collection, *(str(part) for part in parts)])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return f"{stat_type}::{source_collection}::{digest}"


def connect():
    from pymilvus import connections

    load_dotenv_file(PROJECT_ROOT / ".env")
    uri = os.environ.get("ZILLIZ_URI")
    token = os.environ.get("ZILLIZ_TOKEN")
    if not uri or not token:
        raise SystemExit("Missing ZILLIZ_URI or ZILLIZ_TOKEN.")
    connections.connect(uri=uri, token=token)


def get_collection(name: str):
    from pymilvus import Collection, utility

    if not utility.has_collection(name):
        raise SystemExit(f"Collection does not exist: {name}")
    return Collection(name)


def iter_rows(collection, batch_size: int, output_fields: list[str]) -> Iterable[dict[str, Any]]:
    iterator = collection.query_iterator(
        batch_size=batch_size,
        expr="",
        output_fields=output_fields,
    )
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            yield from batch
    finally:
        iterator.close()


def compute_stats(source_collection, batch_size: int, progress_every: int, use_dynamic_flags: bool):
    total = 0
    started_at = time.monotonic()
    output_fields = FLAG_READ_FIELDS if use_dynamic_flags else RAW_READ_FIELDS
    source_year_counts: Counter[tuple[str, int | None]] = Counter()
    source_dblp_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    source_summary_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for row in iter_rows(source_collection, batch_size, output_fields):
        total += 1
        if progress_every and total % progress_every == 0:
            elapsed = max(time.monotonic() - started_at, 0.001)
            print(f"Scanned {total} rows ({total / elapsed:,.0f} rows/s)...", flush=True)

        source = normalize_text(row.get("source"))
        dblp_source = normalize_text(row.get("dblp_source"))
        year = normalize_year(row.get("year"))
        if use_dynamic_flags:
            has_doi = as_bool(row.get("has_doi"))
            has_abstract = as_bool(row.get("has_abstract"))
        else:
            has_doi = has_value(row.get("doi"))
            has_abstract = has_value(row.get("abstract"))
        complete = has_doi and has_abstract

        source_year_counts[(source, year)] += 1

        pair_counts = source_dblp_counts[(source, dblp_source)]
        pair_counts["total"] += 1
        pair_counts["missing_doi"] += 0 if has_doi else 1
        pair_counts["missing_abstract"] += 0 if has_abstract else 1
        pair_counts["complete"] += 1 if complete else 0

        source_counts = source_summary_counts[source]
        source_counts["total"] += 1
        source_counts["missing_doi"] += 0 if has_doi else 1
        source_counts["missing_abstract"] += 0 if has_abstract else 1
        source_counts["complete"] += 1 if complete else 0

    return total, source_year_counts, source_dblp_counts, source_summary_counts


def build_stats_rows(
    *,
    source_collection_name: str,
    generated_at: int,
    source_year_counts: Counter[tuple[str, int | None]],
    source_dblp_counts: dict[tuple[str, str], Counter[str]],
    source_summary_counts: dict[str, Counter[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for (source, year), count in sorted(
        source_year_counts.items(),
        key=lambda item: (item[0][0].casefold(), item[0][1] is None, item[0][1] or -1),
    ):
        rows.append(
            {
                "stat_key": stat_key("source_year", source_collection_name, source, year),
                "stat_type": "source_year",
                "source_collection": source_collection_name,
                "source": source,
                "dblp_source": None,
                "year": year,
                "paper_count": count,
                "missing_doi_count": None,
                "missing_abstract_count": None,
                "complete_count": None,
                "generated_at": generated_at,
                "_stats_vector": STATS_VECTOR,
            }
        )

    for (source, dblp_source), counts in sorted(
        source_dblp_counts.items(), key=lambda item: (item[0][0].casefold(), item[0][1].casefold())
    ):
        rows.append(
            {
                "stat_key": stat_key("source_dblp_completeness", source_collection_name, source, dblp_source),
                "stat_type": "source_dblp_completeness",
                "source_collection": source_collection_name,
                "source": source,
                "dblp_source": dblp_source,
                "year": None,
                "paper_count": counts["total"],
                "missing_doi_count": counts["missing_doi"],
                "missing_abstract_count": counts["missing_abstract"],
                "complete_count": counts["complete"],
                "generated_at": generated_at,
                "_stats_vector": STATS_VECTOR,
            }
        )

    for source, counts in sorted(source_summary_counts.items(), key=lambda item: item[0].casefold()):
        rows.append(
            {
                "stat_key": stat_key("source_summary", source_collection_name, source),
                "stat_type": "source_summary",
                "source_collection": source_collection_name,
                "source": source,
                "dblp_source": None,
                "year": None,
                "paper_count": counts["total"],
                "missing_doi_count": counts["missing_doi"],
                "missing_abstract_count": counts["missing_abstract"],
                "complete_count": counts["complete"],
                "generated_at": generated_at,
                "_stats_vector": STATS_VECTOR,
            }
        )

    return rows


def insert_batches(collection, rows: list[dict[str, Any]], batch_size: int) -> int:
    inserted = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        collection.insert(batch)
        inserted += len(batch)
    collection.flush()
    return inserted


def materialize(args: argparse.Namespace) -> None:
    connect()
    source_collection = get_collection(args.source_collection)
    stats_collection = get_collection(args.stats_collection)

    print(f"Reading from {args.source_collection}", flush=True)
    total, source_year_counts, source_dblp_counts, source_summary_counts = compute_stats(
        source_collection, args.read_batch_size, args.progress_every, args.use_dynamic_flags
    )
    generated_at = int(time.time())
    rows = build_stats_rows(
        source_collection_name=args.source_collection,
        generated_at=generated_at,
        source_year_counts=source_year_counts,
        source_dblp_counts=source_dblp_counts,
        source_summary_counts=source_summary_counts,
    )

    if args.replace:
        stats_collection.load()
        expr = f'source_collection == "{args.source_collection}"'
        result = stats_collection.delete(expr)
        stats_collection.flush()
        print(f"Deleted old stats for {args.source_collection}: {result.delete_count}", flush=True)

    inserted = insert_batches(stats_collection, rows, args.write_batch_size)
    print(f"Scanned papers: {total}", flush=True)
    print(f"Stats rows inserted: {inserted}", flush=True)
    print(f"source_year rows: {len(source_year_counts)}", flush=True)
    print(f"source_dblp_completeness rows: {len(source_dblp_counts)}", flush=True)
    print(f"source_summary rows: {len(source_summary_counts)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize paper stats into Zilliz paper_stats.")
    parser.add_argument("--source-collection", default=DEFAULT_SOURCE_COLLECTION)
    parser.add_argument("--stats-collection", default=DEFAULT_STATS_COLLECTION)
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Zilliz query iterator batch size.",
    )
    parser.add_argument("--write-batch-size", type=int, default=500)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument(
        "--use-dynamic-flags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use dynamic has_doi/has_abstract fields instead of reading full doi/abstract values.",
    )
    parser.add_argument(
        "--replace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete existing stats rows for the source collection before inserting new rows.",
    )
    return parser.parse_args()


def main() -> None:
    materialize(parse_args())


if __name__ == "__main__":
    main()
