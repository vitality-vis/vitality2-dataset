#!/usr/bin/env python3
"""Batch-resolve arXiv IDs from OpenAlex locations by DOI.

The output JSONL is compatible with fetch_arxiv_fullpaper_md.py's OpenAlex
cache. OpenAlex supports OR filters with up to 100 DOI values in one request,
which is much faster than one Work request per DOI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from fetch_arxiv_fullpaper_md import (  # noqa: E402
    arxiv_id_from_openalex_work,
    normalize_arxiv_id,
    normalize_doi,
)


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
SELECT_FIELDS = "id,doi,title,primary_location,best_oa_location,locations"


def load_json_array(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON array: {path}")
    return data


def load_dotenv_key(path: Path, name: str) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                return value.split("#", 1)[0].strip().strip("'\"")
    return ""


def load_cached_dois(path: Path) -> set[str]:
    cached: set[str] = set()
    if not path.exists():
        return cached
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            doi = normalize_doi(item.get("doi")).casefold()
            if doi and item.get("status_code") in {200, 404}:
                cached.add(doi)
    return cached


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def request_batch(
    session: requests.Session,
    dois: list[str],
    *,
    mailto: str,
    timeout: float,
    max_retries: int,
    retry_backoff: float,
    rate_limit_retry_sleep: float,
    max_retry_after: float,
    verbose: bool,
) -> tuple[list[dict[str, Any]], str]:
    filter_value = "doi:" + "|".join(f"https://doi.org/{doi}" for doi in dois)
    params = {"filter": filter_value, "per-page": len(dois), "select": SELECT_FIELDS}
    if mailto:
        params["mailto"] = mailto
    for attempt in range(max_retries + 1):
        try:
            response = session.get(OPENALEX_WORKS_URL, params=params, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results") or []
                if isinstance(results, list):
                    return [item for item in results if isinstance(item, dict)], ""
                return [], "OpenAlex response results was not a list"
            if response.status_code == 429 and attempt < max_retries:
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else rate_limit_retry_sleep
                except ValueError:
                    delay = rate_limit_retry_sleep
                if max_retry_after > 0:
                    delay = min(delay, max_retry_after)
                if verbose:
                    print(
                        f"OpenAlex batch 429; retry {attempt + 1}/{max_retries} after {delay:g}s",
                        file=sys.stderr,
                        flush=True,
                    )
                time.sleep(max(delay, 0.0))
                continue
            if 500 <= response.status_code < 600 and attempt < max_retries:
                delay = max(retry_backoff * (2**attempt), 0.0)
                if verbose:
                    print(
                        f"OpenAlex batch HTTP {response.status_code}; "
                        f"retry {attempt + 1}/{max_retries} after {delay:g}s",
                        file=sys.stderr,
                        flush=True,
                    )
                time.sleep(delay)
                continue
            return [], f"OpenAlex returned HTTP {response.status_code}: {response.text[:500]}"
        except requests.RequestException as exc:
            if attempt < max_retries:
                delay = max(retry_backoff * (2**attempt), 0.0)
                if verbose:
                    print(
                        f"OpenAlex batch request failed: {exc}; "
                        f"retry {attempt + 1}/{max_retries} after {delay:g}s",
                        file=sys.stderr,
                        flush=True,
                    )
                time.sleep(delay)
                continue
            return [], str(exc)
    return [], "OpenAlex batch request failed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefetch OpenAlex arXiv cache in DOI batches.")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.12)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--rate-limit-retry-sleep", type=float, default=10.0)
    parser.add_argument(
        "--max-retry-after",
        type=float,
        default=300.0,
        help="Cap OpenAlex Retry-After waits. Set <=0 to honor the full server value.",
    )
    parser.add_argument("--mailto", default="", help="Email passed to OpenAlex polite pool.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key", default="", help="Explicit OpenAlex API key.")
    parser.add_argument(
        "--api-key-env",
        default="OPENALEX_API_KEY1",
        help="Environment/.env key name used with --use-env-api-key.",
    )
    parser.add_argument("--use-env-api-key", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--user-agent", default="Vitality2 OpenAlex batch fetcher (mailto:unknown@example.com)")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.batch_size <= 100:
        raise SystemExit("--batch-size must be between 1 and 100")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    records = load_json_array(args.input_file)
    seen: set[str] = set()
    dois: list[str] = []
    for record in records:
        doi = normalize_doi(record.get("doi"))
        key = doi.casefold()
        if not doi or key in seen:
            continue
        seen.add(key)
        dois.append(doi)
        if args.limit is not None and len(dois) >= args.limit:
            break

    if args.overwrite and args.output.exists():
        args.output.unlink()
    cached = load_cached_dois(args.output)
    pending = [doi for doi in dois if doi.casefold() not in cached]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent})
    api_key = args.api_key
    if not api_key and args.use_env_api_key:
        api_key = os.environ.get(args.api_key_env, "") or load_dotenv_key(args.env_file, args.api_key_env)
    if api_key:
        session.headers.update({"Authorization": f"Bearer {api_key}"})
    if args.mailto:
        session.headers.update({"From": args.mailto})

    counts = {"cached": len(dois) - len(pending), "found": 0, "not_found": 0, "failed": 0}
    processed = 0
    with args.output.open("a", encoding="utf-8") as out:
        for batch in chunks(pending, args.batch_size):
            works, error = request_batch(
                session,
                batch,
                mailto=args.mailto,
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_backoff=args.retry_backoff,
                rate_limit_retry_sleep=args.rate_limit_retry_sleep,
                max_retry_after=args.max_retry_after,
                verbose=args.verbose,
            )
            returned: dict[str, dict[str, Any]] = {}
            for work in works:
                doi = normalize_doi(work.get("doi"))
                if doi:
                    returned[doi.casefold()] = work

            for doi in batch:
                processed += 1
                work = returned.get(doi.casefold())
                if work is not None:
                    arxiv_id = normalize_arxiv_id(arxiv_id_from_openalex_work(work))
                    item = {
                        "doi": doi,
                        "service": "openalex",
                        "status_code": 200,
                        "openalex_id": work.get("id") or "",
                        "arxiv_id": arxiv_id,
                    }
                    counts["found" if arxiv_id else "not_found"] += 1
                elif error:
                    item = {
                        "doi": doi,
                        "service": "openalex",
                        "status_code": 0,
                        "arxiv_id": "",
                        "error": error[:1000],
                    }
                    counts["failed"] += 1
                else:
                    item = {"doi": doi, "service": "openalex", "status_code": 404, "arxiv_id": ""}
                    counts["not_found"] += 1
                out.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
                out.flush()

            if args.verbose:
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "pending": len(pending),
                            "counts": counts,
                            "output": str(args.output),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(
        json.dumps(
            {"processed": processed, "total_input_dois": len(dois), "counts": counts, "output": str(args.output)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
