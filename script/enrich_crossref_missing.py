#!/usr/bin/env python3
"""Use Crossref to fill missing paper abstracts by DOI.

The script reads data/papers/missing/{source}.json files, excluding
_missing_doi.json. Records that receive a non-empty Crossref abstract are
appended to data/papers/enriched/{source}.json and removed from the
corresponding missing file. Records still lacking an abstract remain in missing.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


CACHE_SCHEMA_VERSION = 1
CROSSREF_WORKS_URL = "https://api.crossref.org/works/"


class RateLimiter:
    def __init__(self, rate_per_second: float) -> None:
        self.min_interval = 1.0 / rate_per_second if rate_per_second > 0 else 0.0
        self.lock = threading.Lock()
        self.next_allowed = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            if now < self.next_allowed:
                sleep_for = self.next_allowed - now
                self.next_allowed += self.min_interval
            else:
                sleep_for = 0.0
                self.next_allowed = now + self.min_interval
        if sleep_for:
            time.sleep(sleep_for)


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


def clean_crossref_abstract(value: object) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    text = re.sub(r"</?jats:[^>]*>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return normalize_text(text)


class CrossrefCache:
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


def message_to_cache_item(doi: str, message: dict[str, Any] | None) -> dict[str, Any]:
    if not message:
        return {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "doi": doi,
            "status": "not_found",
            "message": None,
        }
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "doi": doi,
        "status": "ok",
        "message": {
            "DOI": message.get("DOI"),
            "title": message.get("title"),
            "abstract": clean_crossref_abstract(message.get("abstract")),
            "subject": message.get("subject") or [],
            "is-referenced-by-count": message.get("is-referenced-by-count"),
            "publisher": message.get("publisher"),
            "type": message.get("type"),
        },
    }


def fetch_doi(
    doi: str,
    mailto: str,
    timeout: float,
    max_retries: int,
    retry_backoff: float,
    rate_limit_retry_sleep: float,
    rate_limiter: RateLimiter,
) -> dict[str, Any]:
    params = {}
    if mailto:
        params["mailto"] = mailto
    headers = {"User-Agent": "vitality2-dataset-enrichment/0.1"}
    url = CROSSREF_WORKS_URL + quote(doi, safe="/")
    last_error = ""

    for attempt in range(max_retries + 1):
        rate_limiter.wait()
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            last_error = str(exc)
        else:
            if response.status_code == 404:
                return message_to_cache_item(doi, None)
            if response.status_code in {429, 503}:
                retry_after = response.headers.get("retry-after") or ""
                if attempt < max_retries:
                    try:
                        sleep_seconds = float(retry_after) if retry_after else rate_limit_retry_sleep
                    except ValueError:
                        sleep_seconds = rate_limit_retry_sleep
                    print(
                        f"Crossref {response.status_code}; sleeping {sleep_seconds:.1f}s before retry "
                        f"{attempt + 1}/{max_retries}",
                        file=sys.stderr,
                    )
                    time.sleep(sleep_seconds)
                    continue
                raise RateLimited(
                    f"Crossref rate limited the run; status={response.status_code} "
                    f"retry_after={retry_after or 'unknown'}"
                )
            if response.status_code not in {500, 502, 504}:
                response.raise_for_status()
                data = response.json()
                message = data.get("message") if isinstance(data, dict) else None
                return message_to_cache_item(doi, message)
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"

        if attempt < max_retries:
            time.sleep(retry_backoff * (2**attempt))

    raise RuntimeError(f"Crossref request failed after retries: {last_error}")


def enrich_from_cache_item(paper: dict[str, Any], cache_item: dict[str, Any]) -> dict[str, Any]:
    out = dict(paper)
    message = cache_item.get("message") if isinstance(cache_item, dict) else None
    if isinstance(message, dict):
        abstract = normalize_text(message.get("abstract"))
        if abstract:
            out["abstract"] = abstract
        if out.get("citationCounts") is None and message.get("is-referenced-by-count") is not None:
            out["citationCounts"] = message.get("is-referenced-by-count")
        if not out.get("keywords") and message.get("subject"):
            out["keywords"] = message["subject"]
    return out


def merge_enriched(existing: list[dict[str, Any]], moved: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    cache: CrossrefCache,
    args: argparse.Namespace,
    stats: dict[str, int],
) -> tuple[int, int]:
    missing = load_json_array(missing_path)
    source = missing_path.stem
    print(f"processing {missing_path.name}: {len(missing)} missing papers", file=sys.stderr)
    rate_limiter = RateLimiter(args.rate_limit)
    pending: dict[Future[dict[str, Any]], str] = {}

    for paper in missing:
        stats["papers"] += 1
        doi = normalize_doi(paper.get("doi"))
        if not doi:
            stats["skipped_no_doi"] += 1
            continue
        if cache.get(doi) is not None:
            stats["cache_hits"] += 1
            continue
        stats["to_fetch"] += 1

    def collect_completed(done: set[Future[dict[str, Any]]]) -> None:
        for future in done:
            doi = pending.pop(future)
            cache_item = future.result()
            cache.put(doi, cache_item)
            stats["fetched"] += 1
            if args.progress_every and stats["fetched"] % args.progress_every == 0:
                print_progress(stats, len(pending))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for paper in missing:
            doi = normalize_doi(paper.get("doi"))
            if not doi or cache.get(doi) is not None:
                continue
            future = executor.submit(
                fetch_doi,
                doi,
                args.mailto,
                args.request_timeout,
                args.max_retries,
                args.retry_backoff,
                args.rate_limit_retry_sleep,
                rate_limiter,
            )
            pending[future] = doi
            if len(pending) >= args.max_pending:
                done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
                collect_completed(done)

        while pending:
            done, _ = wait(set(pending), timeout=5.0, return_when=FIRST_COMPLETED)
            if done:
                collect_completed(done)
            else:
                print_progress(stats, len(pending))

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


def print_progress(stats: dict[str, int], pending: int = 0) -> None:
    print(
        "progress: "
        f"papers={stats['papers']} "
        f"to_fetch={stats['to_fetch']} "
        f"fetched={stats['fetched']} "
        f"cache_hits={stats['cache_hits']} "
        f"enriched={stats['enriched']} "
        f"still_missing={stats['still_missing']} "
        f"skipped_no_doi={stats['skipped_no_doi']} "
        f"pending={pending}",
        file=sys.stderr,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill data/papers/missing abstracts with Crossref by DOI."
    )
    parser.add_argument("--papers-dir", type=Path, default=Path("data/papers"))
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--mailto", default="", help="Email for Crossref polite pool.")
    parser.add_argument(
        "--use-env-mailto",
        action="store_true",
        help="Read CROSSREF_MAILTO or EMAIL from env/.env.",
    )
    parser.add_argument("--source", default="", help="Optional source name to process.")
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=1.0)
    parser.add_argument("--rate-limit-retry-sleep", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-pending", type=int, default=16)
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=5.0,
        help="Global Crossref request rate limit. Public pool: 5, polite pool: 10.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Deprecated; use --rate-limit instead.",
    )
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be > 0")
    if args.rate_limit_retry_sleep < 0:
        parser.error("--rate-limit-retry-sleep must be >= 0")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.max_pending < args.workers:
        args.max_pending = args.workers
    if args.rate_limit <= 0:
        parser.error("--rate-limit must be > 0")
    if not args.mailto and args.use_env_mailto:
        args.mailto = os.environ.get("CROSSREF_MAILTO", "") or os.environ.get("EMAIL", "")
        if not args.mailto:
            args.mailto = load_dotenv_key(args.env_file, ["CROSSREF_MAILTO", "EMAIL"])
    args.missing_dir = args.papers_dir / "missing"
    args.enriched_dir = args.papers_dir / "enriched"
    if args.cache is None:
        args.cache = args.papers_dir / "cache" / "crossref_doi_cache.jsonl"
    return args


def main() -> int:
    args = parse_args()
    print("Crossref polite pool: " + ("enabled" if args.mailto else "disabled"), file=sys.stderr)
    cache = CrossrefCache(args.cache)
    files = iter_missing_files(args.missing_dir, args.source or None)
    if not files:
        raise SystemExit(f"No missing source files found in {args.missing_dir}")

    stats = {
        "papers": 0,
        "fetched": 0,
        "cache_hits": 0,
        "to_fetch": 0,
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
            print(f"completed {path.name}: moved={moved} remaining={remaining}", file=sys.stderr)
    except RateLimited as exc:
        stopped_reason = str(exc)
        print(stopped_reason, file=sys.stderr)

    manifest = {
        "papers_dir": str(args.papers_dir),
        "cache": str(args.cache),
        "source": args.source or None,
        "mailto": bool(args.mailto),
        "stopped_reason": stopped_reason,
        "stats": stats,
        "outputs": outputs,
    }
    manifest_path = args.papers_dir / "crossref_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), file=sys.stderr)
    return 0


class RateLimited(Exception):
    pass


if __name__ == "__main__":
    raise SystemExit(main())
