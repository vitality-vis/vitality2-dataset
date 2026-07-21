#!/usr/bin/env python3
"""HTTP service for Zilliz aggregate statistics.

This service intentionally does not scan the full collection. It uses Zilliz
aggregate query support (`groupByFields` + `count(*)`). If the target Zilliz
cluster/API does not support aggregate queries, endpoints return HTTP 501.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

try:
    from create_zilliz_collection import PROJECT_ROOT, load_dotenv_file
except ModuleNotFoundError:
    from script.create_zilliz_collection import PROJECT_ROOT, load_dotenv_file


DEFAULT_COLLECTION = "paper_new"
DEFAULT_CACHE_TTL_SECONDS = 600


class SourceYearRow(BaseModel):
    source: str
    year: int | None
    count: int


class SourceDblpCompletenessRow(BaseModel):
    source: str
    dblp_source: str
    total: int
    missing_doi: int
    missing_abstract: int
    complete: int


class SourceSummaryRow(BaseModel):
    source: str
    total: int
    missing_doi: int
    missing_abstract: int
    complete: int


class StatsResponse(BaseModel):
    collection: str
    generated_at: float
    total: int
    source_year_distribution: list[SourceYearRow]
    source_dblp_completeness: list[SourceDblpCompletenessRow]
    source_summary: list[SourceSummaryRow]


@dataclass
class CacheEntry:
    generated_at: float
    stats: StatsResponse


class AggregateUnavailable(RuntimeError):
    pass


app = FastAPI(title="Vitality2 Zilliz Aggregate Stats Service", version="0.2.0")
_cache: dict[str, CacheEntry] = {}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _collection_name(collection: str | None = None) -> str:
    return collection or os.environ.get("ZILLIZ_COLLECTION") or DEFAULT_COLLECTION


def _zilliz_query_url() -> str:
    load_dotenv_file(PROJECT_ROOT / ".env")
    uri = os.environ.get("ZILLIZ_URI")
    if not uri:
        raise HTTPException(status_code=500, detail="Missing ZILLIZ_URI.")
    return uri.rstrip("/") + "/v2/vectordb/entities/query"


def _headers() -> dict[str, str]:
    load_dotenv_file(PROJECT_ROOT / ".env")
    token = os.environ.get("ZILLIZ_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="Missing ZILLIZ_TOKEN.")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _normalize_source(value: Any) -> str:
    text = str(value or "").strip()
    return text or "Unknown"


def _normalize_year(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _count_value(row: dict[str, Any]) -> int:
    for key in ("count(*)", "count"):
        if key in row:
            return int(row[key])
    raise AggregateUnavailable("Aggregate response is missing count(*).")


def aggregate_count(
    collection: str,
    group_by_fields: list[str],
    *,
    filter_expr: str = "",
    timeout: int = 30,
) -> list[dict[str, Any]]:
    body = {
        "collectionName": collection,
        "filter": filter_expr,
        "groupByFields": group_by_fields,
        "outputFields": [*group_by_fields, "count(*)"],
    }
    response = requests.post(_zilliz_query_url(), headers=_headers(), json=body, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    if payload.get("code") != 0:
        message = str(payload.get("message") or payload)
        if "count(*)" in message or "group" in message.lower():
            raise AggregateUnavailable(message)
        raise HTTPException(status_code=502, detail=message)

    data = payload.get("data")
    if not isinstance(data, list):
        raise HTTPException(status_code=502, detail="Unexpected Zilliz aggregate response shape.")
    return data


def _pair_key(row: dict[str, Any]) -> tuple[str, str]:
    return _normalize_source(row.get("source")), _normalize_source(row.get("dblp_source"))


def compute_stats_with_aggregation(collection: str) -> StatsResponse:
    source_year_rows = aggregate_count(collection, ["source", "year"])
    total_pair_rows = aggregate_count(collection, ["source", "dblp_source"])
    missing_doi_rows = aggregate_count(collection, ["source", "dblp_source"], filter_expr="doi IS NULL")
    missing_abstract_rows = aggregate_count(
        collection, ["source", "dblp_source"], filter_expr="abstract IS NULL"
    )
    complete_rows = aggregate_count(
        collection,
        ["source", "dblp_source"],
        filter_expr="doi IS NOT NULL and abstract IS NOT NULL",
    )

    pairs: dict[tuple[str, str], dict[str, int]] = {}
    for row in total_pair_rows:
        pairs[_pair_key(row)] = {
            "total": _count_value(row),
            "missing_doi": 0,
            "missing_abstract": 0,
            "complete": 0,
        }

    for rows, field in (
        (missing_doi_rows, "missing_doi"),
        (missing_abstract_rows, "missing_abstract"),
        (complete_rows, "complete"),
    ):
        for row in rows:
            key = _pair_key(row)
            pairs.setdefault(key, {"total": 0, "missing_doi": 0, "missing_abstract": 0, "complete": 0})
            pairs[key][field] = _count_value(row)

    source_summary: dict[str, dict[str, int]] = {}
    for (source, _dblp_source), counts in pairs.items():
        summary = source_summary.setdefault(
            source, {"total": 0, "missing_doi": 0, "missing_abstract": 0, "complete": 0}
        )
        for field in ("total", "missing_doi", "missing_abstract", "complete"):
            summary[field] += counts[field]

    source_year_distribution = [
        SourceYearRow(
            source=_normalize_source(row.get("source")),
            year=_normalize_year(row.get("year")),
            count=_count_value(row),
        )
        for row in source_year_rows
    ]
    source_year_distribution.sort(key=lambda row: (row.source.casefold(), row.year is None, row.year or -1))

    source_dblp_completeness = [
        SourceDblpCompletenessRow(
            source=source,
            dblp_source=dblp_source,
            total=counts["total"],
            missing_doi=counts["missing_doi"],
            missing_abstract=counts["missing_abstract"],
            complete=counts["complete"],
        )
        for (source, dblp_source), counts in sorted(
            pairs.items(), key=lambda item: (item[0][0].casefold(), item[0][1].casefold())
        )
    ]

    summary_rows = [
        SourceSummaryRow(
            source=source,
            total=counts["total"],
            missing_doi=counts["missing_doi"],
            missing_abstract=counts["missing_abstract"],
            complete=counts["complete"],
        )
        for source, counts in sorted(source_summary.items(), key=lambda item: item[0].casefold())
    ]

    return StatsResponse(
        collection=collection,
        generated_at=time.time(),
        total=sum(row.count for row in source_year_distribution),
        source_year_distribution=source_year_distribution,
        source_dblp_completeness=source_dblp_completeness,
        source_summary=summary_rows,
    )


def get_stats(collection: str, refresh: bool) -> StatsResponse:
    ttl = _env_int("STATS_CACHE_TTL_SECONDS", DEFAULT_CACHE_TTL_SECONDS)
    cached = _cache.get(collection)
    if cached and not refresh and time.time() - cached.generated_at <= ttl:
        return cached.stats

    try:
        stats = compute_stats_with_aggregation(collection)
    except AggregateUnavailable as exc:
        raise HTTPException(
            status_code=501,
            detail=(
                "Zilliz aggregate query is unavailable for this cluster/API. "
                "Use an On-Demand cluster/API that supports groupByFields + count(*), "
                f"or maintain a materialized stats collection. Original error: {exc}"
            ),
        ) from exc

    _cache[collection] = CacheEntry(generated_at=stats.generated_at, stats=stats)
    return stats


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stats", response_model=StatsResponse)
def stats(collection: str | None = Query(default=None), refresh: bool = Query(default=False)) -> StatsResponse:
    return get_stats(_collection_name(collection), refresh)


@app.get("/stats/source-year-distribution", response_model=list[SourceYearRow])
def source_year_distribution(
    collection: str | None = Query(default=None), refresh: bool = Query(default=False)
) -> list[SourceYearRow]:
    return get_stats(_collection_name(collection), refresh).source_year_distribution


@app.get("/stats/source-dblp-completeness", response_model=list[SourceDblpCompletenessRow])
def source_dblp_completeness(
    collection: str | None = Query(default=None), refresh: bool = Query(default=False)
) -> list[SourceDblpCompletenessRow]:
    return get_stats(_collection_name(collection), refresh).source_dblp_completeness


@app.get("/stats/source-summary", response_model=list[SourceSummaryRow])
def source_summary(
    collection: str | None = Query(default=None), refresh: bool = Query(default=False)
) -> list[SourceSummaryRow]:
    return get_stats(_collection_name(collection), refresh).source_summary
