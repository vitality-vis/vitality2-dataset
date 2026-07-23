#!/usr/bin/env python3
"""Enrich DBLP split-source papers with OpenAlex metadata by DOI.

Papers with a DOI are looked up as singleton OpenAlex Work requests through
PyAlex. Records with a non-empty OpenAlex abstract are written to
data/papers/enriched/{source}.json. Records with a DOI but no usable
abstract are written to data/papers/missing/{source}.json. Records without a
DOI are collected in data/papers/missing/_missing_doi.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import requests
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import quote


CACHE_SCHEMA_VERSION = 2
OPENALEX_WORKS_URL = "https://api.openalex.org/works/"


@dataclass(frozen=True)
class OpenAlexClientConfig:
    api_key: str = ""
    email: str = ""
    timeout: float = 20.0
    max_retries: int = 3
    retry_backoff: float = 0.5
    auth_mode: str = "anonymous"


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


def safe_filename(source: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._ -]+", "_", normalize_text(source))
    safe = re.sub(r"\s+", " ", safe).strip(" ._")
    return f"{safe or 'Unknown'}.json"


def load_dotenv_key(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() != key:
                continue
            return value.strip().strip("'\"")
    return ""


def load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a top-level JSON array")
    return data


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
        json.dump(item, self.handle, ensure_ascii=False, separators=(",", ":"))
        self.count += 1

    def close(self) -> None:
        if self.handle is None:
            return
        self.handle.write("\n]\n")
        self.handle.close()
        self.handle = None


class OutputSet:
    def __init__(self, enriched_dir: Path, missing_dir: Path) -> None:
        self.enriched_dir = enriched_dir
        self.missing_dir = missing_dir
        self.enriched: dict[str, JsonArrayWriter] = {}
        self.missing: dict[str, JsonArrayWriter] = {}
        self.missing_doi = JsonArrayWriter(missing_dir / "_missing_doi.json")

    def _source_writer(
        self, writers: dict[str, JsonArrayWriter], base_dir: Path, source: str
    ) -> JsonArrayWriter:
        filename = safe_filename(source)
        writer = writers.get(filename)
        if writer is None:
            writer = JsonArrayWriter(base_dir / filename)
            writers[filename] = writer
        return writer

    def write_enriched(self, paper: dict[str, Any]) -> None:
        source = normalize_text(paper.get("source")) or "Unknown"
        self._source_writer(self.enriched, self.enriched_dir, source).write(paper)

    def write_missing(self, paper: dict[str, Any]) -> None:
        source = normalize_text(paper.get("source")) or "Unknown"
        self._source_writer(self.missing, self.missing_dir, source).write(paper)

    def write_missing_doi(self, paper: dict[str, Any]) -> None:
        self.missing_doi.write(paper)

    def close(self) -> dict[str, Any]:
        for writer in list(self.enriched.values()) + list(self.missing.values()):
            writer.close()
        self.missing_doi.close()
        return {
            "enriched_files": {
                filename: writer.count for filename, writer in sorted(self.enriched.items())
            },
            "missing_files": {
                filename: writer.count for filename, writer in sorted(self.missing.items())
            },
            "missing_doi": self.missing_doi.count,
        }


class DoiCache:
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
                        self.items[str(doi).casefold()] = item

    def get(self, doi: str) -> dict[str, Any] | None:
        return self.items.get(doi.casefold())

    def put(self, doi: str, item: dict[str, Any]) -> None:
        if item.get("status") not in {"ok", "not_found"}:
            return
        key = doi.casefold()
        if key in self.items:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(item, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        self.items[key] = item


def iter_papers(input_files: list[Path]) -> tuple[Path, dict[str, Any]]:
    for input_file in input_files:
        papers = load_json_array(input_file)
        print(f"processing {input_file.name}: {len(papers)} papers", file=sys.stderr)
        for paper in papers:
            yield input_file, paper


def extract_names(items: object) -> list[str]:
    names: list[str] = []
    if not isinstance(items, list):
        return names
    for item in items:
        if not isinstance(item, dict):
            continue
        name = normalize_text(item.get("display_name"))
        if name and name not in names:
            names.append(name)
    return names


def abstract_from_inverted_index(index: object) -> str:
    if not isinstance(index, dict):
        return ""

    positions: list[tuple[int, str]] = []
    for word, word_positions in index.items():
        if not isinstance(word_positions, list):
            continue
        for position in word_positions:
            try:
                positions.append((int(position), str(word)))
            except (TypeError, ValueError):
                continue
    return normalize_text(" ".join(word for _, word in sorted(positions)))


def work_to_cache_item(doi: str, work: dict[str, Any] | None, error: str = "") -> dict[str, Any]:
    if work is None:
        return {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "doi": doi,
            "status": "not_found" if not error else "error",
            "error": error,
            "work": None,
        }

    abstract = normalize_text(work.get("abstract"))
    if not abstract:
        abstract = abstract_from_inverted_index(work.get("abstract_inverted_index"))
    keywords = extract_names(work.get("keywords"))
    if not keywords:
        keywords = extract_names(work.get("topics"))
    if not keywords:
        keywords = extract_names(work.get("concepts"))

    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "doi": doi,
        "status": "ok",
        "error": "",
        "work": {
            "id": work.get("id"),
            "doi": work.get("doi"),
            "display_name": work.get("display_name"),
            "publication_year": work.get("publication_year"),
            "cited_by_count": work.get("cited_by_count"),
            "abstract": abstract,
            "keywords": keywords,
            "type": work.get("type"),
            "open_access": work.get("open_access"),
            "primary_location": work.get("primary_location"),
        },
    }


def configure_openalex(args: argparse.Namespace) -> OpenAlexClientConfig:
    try:
        import pyalex
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pyalex. Install it with:\n"
            "  source ~/venvs/WEB/bin/activate && python3 -m pip install pyalex"
        ) from exc

    api_key = args.api_key
    if not api_key and args.use_env_api_key:
        api_key = os.environ.get("OPENALEX_API_KEY1", "")
    if not api_key and args.use_env_api_key:
        api_key = load_dotenv_key(args.env_file, "OPENALEX_API_KEY1")
    auth_mode = "api_key" if api_key else "anonymous"
    return OpenAlexClientConfig(
        api_key=api_key,
        email=args.email,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        auth_mode=auth_mode,
    )


def fetch_work_by_doi(client_config: OpenAlexClientConfig, doi: str) -> dict[str, Any]:
    url = OPENALEX_WORKS_URL + "doi:" + quote(doi, safe="/")
    headers = {}
    if client_config.api_key:
        headers["Authorization"] = f"Bearer {client_config.api_key}"
    if client_config.email:
        headers["From"] = client_config.email

    last_error = ""
    for attempt in range(client_config.max_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=client_config.timeout)
        except requests.RequestException as exc:
            last_error = str(exc)
        else:
            if response.status_code == 404:
                return work_to_cache_item(doi, None)
            if response.status_code == 429:
                retry_after = ""
                try:
                    retry_after = str(response.json().get("retryAfter") or "")
                except ValueError:
                    pass
                if attempt < client_config.max_retries:
                    try:
                        sleep_seconds = float(retry_after) if retry_after else client_config.retry_backoff
                    except ValueError:
                        sleep_seconds = client_config.retry_backoff
                    sleep_seconds = max(sleep_seconds, client_config.retry_backoff * (2**attempt))
                    print(
                        f"OpenAlex 429; sleeping {sleep_seconds:.2f}s before retry "
                        f"{attempt + 1}/{client_config.max_retries}",
                        file=sys.stderr,
                    )
                    time.sleep(sleep_seconds)
                    continue
                return {
                    "cache_schema_version": CACHE_SCHEMA_VERSION,
                    "doi": doi,
                    "status": "rate_limited",
                    "error": f"HTTP 429: {response.text[:300]}",
                    "retry_after": retry_after,
                    "work": None,
                }
            if response.status_code not in {429, 500, 503}:
                try:
                    response.raise_for_status()
                    return work_to_cache_item(doi, response.json())
                except Exception as exc:
                    return work_to_cache_item(doi, None, error=str(exc))
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"

        if attempt < client_config.max_retries:
            time.sleep(client_config.retry_backoff * (2**attempt))

    return work_to_cache_item(doi, None, error=last_error)


def prepare_output(args: argparse.Namespace) -> None:
    output_paths = [
        args.enriched_dir,
        args.missing_dir,
        args.output_dir / "openalex_manifest.json",
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            f"{args.output_dir} already contains JSON output. "
            "Use --overwrite to replace it."
        )
    if args.overwrite:
        for path in output_paths:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    args.enriched_dir.mkdir(parents=True, exist_ok=True)
    args.missing_dir.mkdir(parents=True, exist_ok=True)


def iter_input_files(input_dir: Path, only_source: str | None) -> list[Path]:
    files = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        if only_source and path.stem != only_source:
            continue
        files.append(path)
    return files


def enrich_paper(paper: dict[str, Any], cache_item: dict[str, Any]) -> dict[str, Any]:
    out = dict(paper)
    work = cache_item.get("work") if isinstance(cache_item, dict) else None
    if isinstance(work, dict):
        out["abstract"] = normalize_text(work.get("abstract"))
        if work.get("keywords"):
            out["keywords"] = work["keywords"]
        if work.get("cited_by_count") is not None:
            out["citation_count"] = work["cited_by_count"]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich DBLP split-source JSON files with OpenAlex abstracts by DOI."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/dblp/split_source"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/papers"))
    parser.add_argument("--cache", type=Path, default=Path("data/papers/cache/openalex_doi_cache.jsonl"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key", default="", help="Explicit OpenAlex API key. Not used by default.")
    parser.add_argument(
        "--use-env-api-key",
        action="store_true",
        help="Read OPENALEX_API_KEY from the environment or .env. Avoid this for the free DOI path.",
    )
    parser.add_argument("--email", default="", help="Optional polite-pool email for OpenAlex.")
    parser.add_argument("--source", default="", help="Optional source name to process, e.g. CHI.")
    parser.add_argument("--limit", type=int, default=None, help="Optional total paper limit for testing.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep after each uncached DOI lookup.")
    parser.add_argument("--workers", type=int, default=16, help="Concurrent uncached DOI lookups.")
    parser.add_argument(
        "--max-pending",
        type=int,
        default=128,
        help="Maximum queued concurrent DOI lookups before collecting results.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=20.0,
        help="Seconds before one OpenAlex DOI request times out.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress after this many processed papers. Use 0 to disable.",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be > 0")
    if args.max_pending < args.workers:
        args.max_pending = args.workers
    args.enriched_dir = args.output_dir / "enriched"
    args.missing_dir = args.output_dir / "missing"
    return args


def main() -> int:
    args = parse_args()
    client_config = configure_openalex(args)
    print(f"OpenAlex auth mode: {client_config.auth_mode}", file=sys.stderr)
    prepare_output(args)
    cache = DoiCache(args.cache)
    outputs = OutputSet(args.enriched_dir, args.missing_dir)
    started_at = time.monotonic()

    stats = {
        "papers": 0,
        "with_doi": 0,
        "missing_doi": 0,
        "cache_hits": 0,
        "fetched": 0,
        "enriched": 0,
        "missing_abstract": 0,
        "errors": 0,
    }

    input_files = iter_input_files(args.input_dir, args.source or None)
    if not input_files:
        raise SystemExit(f"No source JSON files found in {args.input_dir}")

    pending: dict[Future[dict[str, Any]], tuple[str, dict[str, Any]]] = {}
    in_flight_dois: dict[str, Future[dict[str, Any]]] = {}
    stopped_reason = ""

    def consume(cache_item: dict[str, Any], paper: dict[str, Any]) -> None:
        if cache_item.get("status") == "rate_limited":
            raise RateLimited(cache_item)
        enriched = enrich_paper(paper, cache_item)
        if normalize_text(enriched.get("abstract")):
            stats["enriched"] += 1
            outputs.write_enriched(enriched)
        else:
            stats["missing_abstract"] += 1
            if cache_item.get("error"):
                stats["errors"] += 1
            outputs.write_missing(enriched)
        print_progress()

    def print_progress(force: bool = False) -> None:
        if not force and (
            not args.progress_every or stats["papers"] % args.progress_every != 0
        ):
            return
        elapsed = max(time.monotonic() - started_at, 0.001)
        rate = stats["papers"] / elapsed
        print(
            "progress: "
            f"papers={stats['papers']} "
            f"with_doi={stats['with_doi']} "
            f"fetched={stats['fetched']} "
            f"cache_hits={stats['cache_hits']} "
            f"enriched={stats['enriched']} "
            f"missing_abstract={stats['missing_abstract']} "
            f"missing_doi={stats['missing_doi']} "
            f"errors={stats['errors']} "
            f"pending={len(pending)} "
            f"rate={rate:.2f}/s",
            file=sys.stderr,
        )

    def collect_completed(done: set[Future[dict[str, Any]]]) -> None:
        for future in done:
            doi, paper = pending.pop(future)
            in_flight_dois.pop(doi.casefold(), None)
            cache_item = future.result()
            cache.put(doi, cache_item)
            stats["fetched"] += 1
            consume(cache_item, paper)

    try:
        executor = ThreadPoolExecutor(max_workers=args.workers)
        try:
            for _, paper in iter_papers(input_files):
                if args.limit is not None and stats["papers"] >= args.limit:
                    break

                stats["papers"] += 1
                doi = normalize_doi(paper.get("doi"))
                if not doi:
                    stats["missing_doi"] += 1
                    outputs.write_missing_doi(paper)
                    print_progress()
                    continue

                stats["with_doi"] += 1
                cache_item = cache.get(doi)
                if cache_item is not None:
                    stats["cache_hits"] += 1
                    consume(cache_item, paper)
                    continue

                key = doi.casefold()
                existing_future = in_flight_dois.get(key)
                if existing_future is not None:
                    done, _ = wait({existing_future}, return_when=FIRST_COMPLETED)
                    collect_completed(done)
                    cached_after_wait = cache.get(doi)
                    if cached_after_wait is None:
                        raise RuntimeError(f"Pending DOI lookup did not populate cache: {doi}")
                    stats["cache_hits"] += 1
                    consume(cached_after_wait, paper)
                    continue

                future = executor.submit(fetch_work_by_doi, client_config, doi)
                pending[future] = (doi, paper)
                in_flight_dois[key] = future
                if args.sleep:
                    time.sleep(args.sleep)

                if len(pending) >= args.max_pending:
                    done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
                    collect_completed(done)

            while pending:
                done, _ = wait(set(pending), timeout=5.0, return_when=FIRST_COMPLETED)
                if done:
                    collect_completed(done)
                else:
                    print_progress(force=True)
        except RateLimited as exc:
            retry_after = exc.cache_item.get("retry_after") or "unknown"
            stopped_reason = f"OpenAlex rate limited the run; retry_after={retry_after} seconds"
            print(stopped_reason, file=sys.stderr)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    finally:
        output_counts = outputs.close()
        print_progress(force=True)

    manifest = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "cache": str(args.cache),
        "source": args.source or None,
        "limit": args.limit,
        "workers": args.workers,
        "max_pending": args.max_pending,
        "request_timeout": args.request_timeout,
        "auth_mode": client_config.auth_mode,
        "stopped_reason": stopped_reason,
        "stats": stats,
        "outputs": output_counts,
    }
    manifest_path = args.output_dir / "openalex_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(manifest, ensure_ascii=False, indent=2), file=sys.stderr)
    return 0


class RateLimited(Exception):
    def __init__(self, cache_item: dict[str, Any]) -> None:
        super().__init__(cache_item.get("error") or "OpenAlex rate limited the run")
        self.cache_item = cache_item


if __name__ == "__main__":
    raise SystemExit(main())
