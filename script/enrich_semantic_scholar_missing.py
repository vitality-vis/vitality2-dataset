#!/usr/bin/env python3
"""Use Semantic Scholar to fill missing paper abstracts after OpenAlex enrichment.

The script reads data/papers/missing/{source}.json files, excluding
_missing_doi.json. Records that receive a non-empty Semantic Scholar abstract
are appended to data/papers/enriched/{source}.json and removed from the
corresponding missing file. Records still lacking an abstract remain in missing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import requests


CACHE_SCHEMA_VERSION = 1
S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = "paperId,externalIds,title,abstract,citationCount,fieldsOfStudy,s2FieldsOfStudy"


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_doi(value: object) -> str:
    doi = normalize_text(value)
    lower = doi.lower()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ):
        if lower.startswith(prefix):
            return doi[len(prefix) :].strip()
    return doi


def doi_key(doi: str) -> str:
    return normalize_doi(doi).casefold()


def safe_filename(source: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._ -]+", "_", normalize_text(source))
    safe = re.sub(r"\s+", " ", safe).strip(" ._")
    return f"{safe or 'Unknown'}.json"


def load_dotenv_key(path: Path, names: list[str]) -> str:
    if not path.exists():
        return ""
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            values[name.strip()] = value.strip().strip("'\"")
    for name in names:
        if values.get(name):
            return values[name]
    return ""


def load_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a top-level JSON array")
    return data


def write_json_array_atomic(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        for index, item in enumerate(items):
            if index:
                handle.write(",\n")
            json.dump(item, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n]\n")
    tmp_path.replace(path)


class SemanticScholarCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.items: dict[str, dict[str, Any]] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid cache line {line_no} in {path}: {exc}") from exc
                    if item.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
                        continue
                    if item.get("status") not in {"ok", "not_found"}:
                        continue
                    doi = item.get("doi")
                    if doi:
                        self.items[doi_key(doi)] = item

    def get(self, doi: str) -> dict[str, Any] | None:
        return self.items.get(doi_key(doi))

    def put(self, doi: str, item: dict[str, Any]) -> None:
        key = doi_key(doi)
        if key in self.items:
            return
        if item.get("status") not in {"ok", "not_found"}:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(item, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        self.items[key] = item


def paper_to_cache_item(doi: str, paper: dict[str, Any] | None) -> dict[str, Any]:
    if not paper:
        return {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "doi": doi,
            "status": "not_found",
            "paper": None,
        }

    fields = paper.get("fieldsOfStudy") or []
    s2_fields = paper.get("s2FieldsOfStudy") or []
    categories: list[str] = []
    for value in fields:
        text = normalize_text(value)
        if text and text not in categories:
            categories.append(text)
    if isinstance(s2_fields, list):
        for item in s2_fields:
            if isinstance(item, dict):
                text = normalize_text(item.get("category"))
                if text and text not in categories:
                    categories.append(text)

    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "doi": doi,
        "status": "ok",
        "paper": {
            "paperId": paper.get("paperId"),
            "externalIds": paper.get("externalIds"),
            "title": paper.get("title"),
            "abstract": normalize_text(paper.get("abstract")),
            "citationCount": paper.get("citationCount"),
            "fieldsOfStudy": categories,
        },
    }


def fetch_batch(
    dois: list[str],
    api_key: str,
    use_empty_api_key_header: bool,
    timeout: float,
    max_retries: int,
    retry_backoff: float,
    rate_limit_retry_sleep: float,
) -> list[dict[str, Any] | None]:
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    elif use_empty_api_key_header:
        headers["x-api-key"] = ""

    body = {"ids": [f"DOI:{doi}" for doi in dois]}
    params = {"fields": S2_FIELDS}
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                S2_BATCH_URL,
                params=params,
                json=body,
                headers=headers,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_error = str(exc)
        else:
            if response.status_code == 429:
                retry_after = response.headers.get("retry-after") or ""
                if attempt < max_retries:
                    try:
                        sleep_seconds = float(retry_after) if retry_after else rate_limit_retry_sleep
                    except ValueError:
                        sleep_seconds = rate_limit_retry_sleep
                    print(
                        f"Semantic Scholar 429; sleeping {sleep_seconds:.1f}s before retry "
                        f"{attempt + 1}/{max_retries}",
                        file=sys.stderr,
                    )
                    time.sleep(sleep_seconds)
                    continue
                raise RateLimited(
                    f"Semantic Scholar rate limited the run; retry_after={retry_after or 'unknown'}"
                )
            if response.status_code == 403:
                raise PermissionError(
                    "Semantic Scholar returned 403 Forbidden. Check SEMANTIC_SCHOLAR_API_KEY/S2_API_KEY."
                )
            if response.status_code == 400:
                if len(dois) == 1:
                    print(
                        f"Semantic Scholar rejected DOI {dois[0]!r}: {response.text[:300]}",
                        file=sys.stderr,
                    )
                    return [None]
                midpoint = len(dois) // 2
                print(
                    f"Semantic Scholar rejected a batch of {len(dois)} DOI ids; splitting batch",
                    file=sys.stderr,
                )
                left = fetch_batch(
                    dois[:midpoint],
                    api_key,
                    use_empty_api_key_header,
                    timeout,
                    max_retries,
                    retry_backoff,
                    rate_limit_retry_sleep,
                )
                right = fetch_batch(
                    dois[midpoint:],
                    api_key,
                    use_empty_api_key_header,
                    timeout,
                    max_retries,
                    retry_backoff,
                    rate_limit_retry_sleep,
                )
                return left + right
            if response.status_code not in {500, 502, 503, 504}:
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, list):
                    raise ValueError(f"Unexpected Semantic Scholar response: {data!r}")
                return data
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"

        if attempt < max_retries:
            time.sleep(retry_backoff * (2**attempt))

    raise RuntimeError(f"Semantic Scholar request failed after retries: {last_error}")


def enrich_from_cache_item(paper: dict[str, Any], cache_item: dict[str, Any]) -> dict[str, Any]:
    out = dict(paper)
    s2_paper = cache_item.get("paper") if isinstance(cache_item, dict) else None
    if isinstance(s2_paper, dict):
        old_citation_field = "citation" + "Counts"
        abstract = normalize_text(s2_paper.get("abstract"))
        if abstract:
            out["abstract"] = abstract
        if (
            out.get("citation_count") is None
            and out.get(old_citation_field) is None
            and s2_paper.get("citationCount") is not None
        ):
            out["citation_count"] = s2_paper.get("citationCount")
        if not out.get("keywords") and s2_paper.get("fieldsOfStudy"):
            out["keywords"] = s2_paper["fieldsOfStudy"]
    return out


def merge_enriched(
    existing: list[dict[str, Any]], moved: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_doi: dict[str, int] = {}
    for idx, paper in enumerate(existing):
        doi = normalize_doi(paper.get("doi"))
        if doi:
            by_doi[doi_key(doi)] = idx

    merged = list(existing)
    for paper in moved:
        doi = normalize_doi(paper.get("doi"))
        key = doi_key(doi)
        if doi and key in by_doi:
            merged[by_doi[key]] = paper
        else:
            if doi:
                by_doi[key] = len(merged)
            merged.append(paper)
    return merged


def iter_missing_files(missing_dir: Path, only_source: str | None) -> list[Path]:
    files = []
    for path in sorted(missing_dir.glob("*.json")):
        if path.name == "_missing_doi.json":
            continue
        if only_source and path.stem != only_source:
            continue
        files.append(path)
    return files


def process_source(
    missing_path: Path,
    enriched_dir: Path,
    cache: SemanticScholarCache,
    args: argparse.Namespace,
    stats: dict[str, int],
) -> tuple[int, int]:
    missing = load_json_array(missing_path)
    source = missing_path.stem
    print(f"processing {missing_path.name}: {len(missing)} missing papers", file=sys.stderr)

    dois_to_fetch: list[str] = []
    seen: set[str] = set()
    for paper in missing:
        stats["papers"] += 1
        doi = normalize_doi(paper.get("doi"))
        if not doi:
            stats["skipped_no_doi"] += 1
            continue
        if cache.get(doi) is not None:
            stats["cache_hits"] += 1
            continue
        key = doi_key(doi)
        if key not in seen:
            seen.add(key)
            dois_to_fetch.append(doi)

    for start in range(0, len(dois_to_fetch), args.batch_size):
        batch = dois_to_fetch[start : start + args.batch_size]
        if not batch:
            continue
        response_items = fetch_batch(
            batch,
            args.api_key,
            args.use_empty_api_key_header,
            args.request_timeout,
            args.max_retries,
            args.retry_backoff,
            args.rate_limit_retry_sleep,
        )
        for doi, response_item in zip(batch, response_items):
            cache_item = paper_to_cache_item(doi, response_item)
            cache.put(doi, cache_item)
            stats["fetched"] += 1
        if args.sleep:
            time.sleep(args.sleep)
        if args.progress_every and stats["fetched"] % args.progress_every == 0:
            print_progress(stats)

    moved: list[dict[str, Any]] = []
    still_missing: list[dict[str, Any]] = []
    for paper in missing:
        doi = normalize_doi(paper.get("doi"))
        cache_item = cache.get(doi) if doi else None
        enriched = enrich_from_cache_item(paper, cache_item or {})
        if normalize_text(enriched.get("abstract")):
            moved.append(enriched)
            stats["enriched"] += 1
        else:
            still_missing.append(enriched)
            stats["still_missing"] += 1

    if moved:
        enriched_path = enriched_dir / safe_filename(source)
        existing_enriched = load_json_array(enriched_path)
        write_json_array_atomic(enriched_path, merge_enriched(existing_enriched, moved))
    write_json_array_atomic(missing_path, still_missing)

    return len(moved), len(still_missing)


def print_progress(stats: dict[str, int]) -> None:
    print(
        "progress: "
        f"papers={stats['papers']} "
        f"fetched={stats['fetched']} "
        f"cache_hits={stats['cache_hits']} "
        f"enriched={stats['enriched']} "
        f"still_missing={stats['still_missing']} "
        f"skipped_no_doi={stats['skipped_no_doi']}",
        file=sys.stderr,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill data/papers/missing abstracts with Semantic Scholar by DOI."
    )
    parser.add_argument("--papers-dir", type=Path, default=Path("data/papers"))
    parser.add_argument("--cache", type=Path, default=Path("data/papers/cache/semantic_scholar_doi_cache.jsonl"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key", default="", help="Semantic Scholar API key.")
    parser.add_argument(
        "--use-env-api-key",
        action="store_true",
        help="Read SEMANTIC_SCHOLAR_API_KEY or S2_API_KEY from env/.env.",
    )
    parser.add_argument(
        "--use-empty-api-key-header",
        action="store_true",
        help="Send an empty x-api-key header when no API key is configured.",
    )
    parser.add_argument("--source", default="", help="Optional source name to process.")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=1.0)
    parser.add_argument(
        "--rate-limit-retry-sleep",
        type=float,
        default=5.0,
        help="Seconds to wait before retrying a Semantic Scholar 429 with no Retry-After header.",
    )
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 500:
        parser.error("--batch-size must be between 1 and 500")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be > 0")
    if args.rate_limit_retry_sleep < 0:
        parser.error("--rate-limit-retry-sleep must be >= 0")
    if not args.api_key and args.use_env_api_key:
        args.api_key = os.environ.get("S2_API_KEY", "") or os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        if not args.api_key:
            args.api_key = load_dotenv_key(args.env_file, ["S2_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"])
    args.missing_dir = args.papers_dir / "missing"
    args.enriched_dir = args.papers_dir / "enriched"
    return args


def main() -> int:
    args = parse_args()
    print(
        "Semantic Scholar auth mode: " + ("api_key" if args.api_key else "anonymous"),
        file=sys.stderr,
    )
    cache = SemanticScholarCache(args.cache)
    files = iter_missing_files(args.missing_dir, args.source or None)
    if not files:
        raise SystemExit(f"No missing source files found in {args.missing_dir}")

    stats = {
        "papers": 0,
        "fetched": 0,
        "cache_hits": 0,
        "enriched": 0,
        "still_missing": 0,
        "skipped_no_doi": 0,
    }
    outputs: dict[str, dict[str, int]] = {}
    stopped_reason = ""

    try:
        for path in files:
            moved, remaining = process_source(path, args.enriched_dir, cache, args, stats)
            outputs[path.name] = {"moved_to_enriched": moved, "remaining_missing": remaining}
            print(
                f"completed {path.name}: moved={moved} remaining={remaining}",
                file=sys.stderr,
            )
    except (RateLimited, PermissionError) as exc:
        stopped_reason = str(exc)
        print(stopped_reason, file=sys.stderr)

    manifest = {
        "papers_dir": str(args.papers_dir),
        "cache": str(args.cache),
        "source": args.source or None,
        "batch_size": args.batch_size,
        "auth_mode": "api_key" if args.api_key else "anonymous",
        "stopped_reason": stopped_reason,
        "stats": stats,
        "outputs": outputs,
    }
    manifest_path = args.papers_dir / "semantic_scholar_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), file=sys.stderr)
    return 0


class RateLimited(Exception):
    pass


if __name__ == "__main__":
    raise SystemExit(main())
