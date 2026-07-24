#!/usr/bin/env python3
"""Use Semantic Scholar title search to recover DOI/metadata for missing DOI papers.

The script reads data/papers/missing/_missing_doi.json and writes a standalone
update batch under data/papers/updateYYYYMMDD by default. It does not modify the
main data/papers/enriched or data/papers/missing directories.

Recovered records are split into:

- updateYYYYMMDD/enriched/{source}.json: DOI and abstract recovered
- updateYYYYMMDD/missing/{source}.json: DOI recovered, abstract still missing
- updateYYYYMMDD/missing/_missing_doi.json: DOI still missing after search
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import requests


CACHE_SCHEMA_VERSION = 1
S2_MATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/match"
S2_FIELDS = ",".join(
    [
        "paperId",
        "externalIds",
        "title",
        "abstract",
        "citationCount",
        "year",
        "authors",
        "fieldsOfStudy",
        "s2FieldsOfStudy",
        "venue",
        "publicationTypes",
    ]
)


class RateLimited(Exception):
    pass


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


def normalize_title(value: object) -> str:
    title = normalize_text(value).casefold()
    title = re.sub(r"&[a-z0-9#]+;", " ", title)
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def normalize_author(value: object) -> str:
    text = normalize_text(value).casefold()
    text = re.sub(r"\s+\d{4}$", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_filename(source: object) -> str:
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
        raise ValueError(f"{path} must contain a JSON array")
    return [item for item in data if isinstance(item, dict)]


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


def search_key(paper: dict[str, Any]) -> str:
    return json.dumps(
        {
            "title": normalize_title(paper.get("title")),
            "year": normalize_text(paper.get("year")),
            "source": normalize_text(paper.get("source")).casefold(),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def s2_authors(paper: dict[str, Any]) -> list[str]:
    authors = paper.get("authors")
    if not isinstance(authors, list):
        return []
    names: list[str] = []
    for author in authors:
        if isinstance(author, str):
            name = normalize_text(author)
            if name:
                names.append(name)
            continue
        if not isinstance(author, dict):
            continue
        name = normalize_text(author.get("name"))
        if name:
            names.append(name)
    return names


def author_overlap(paper_authors: object, candidate_authors: object) -> float:
    left = {
        normalize_author(author)
        for author in paper_authors
        if normalize_author(author)
    } if isinstance(paper_authors, list) else set()
    right = {
        normalize_author(author)
        for author in candidate_authors
        if normalize_author(author)
    } if isinstance(candidate_authors, list) else set()
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left), len(right))


def title_similarity(left: object, right: object) -> float:
    left_norm = normalize_title(left)
    right_norm = normalize_title(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def is_generic_title(value: object) -> bool:
    title = normalize_title(value)
    if len(title.split()) <= 2:
        return True
    return title in {
        "editorial",
        "preface",
        "introduction",
        "foreword",
        "contents",
        "back matter",
        "front matter",
        "panel session",
        "panel sessions",
    }


def year_distance(paper: dict[str, Any], candidate: dict[str, Any]) -> int | None:
    paper_year = normalize_text(paper.get("year"))
    candidate_year = candidate.get("year")
    try:
        return abs(int(paper_year) - int(candidate_year))
    except (TypeError, ValueError):
        return None


def candidate_score(paper: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    candidate_authors = s2_authors(candidate)
    title_score = title_similarity(paper.get("title"), candidate.get("title"))
    overlap = author_overlap(paper.get("authors"), candidate_authors)
    distance = year_distance(paper, candidate)
    has_year_match = distance is not None and distance <= 1
    match_score = candidate.get("matchScore")

    accepted = False
    reason = "below_threshold"
    if is_generic_title(paper.get("title")) and overlap <= 0:
        reason = "generic_title_without_author_overlap"
    elif title_score >= 0.985 and (has_year_match or overlap > 0):
        accepted = True
        reason = "near_exact_title"
    elif title_score >= 0.94 and has_year_match and overlap >= 0.25:
        accepted = True
        reason = "title_year_author"
    elif title_score >= 0.90 and has_year_match and overlap >= 0.5:
        accepted = True
        reason = "strong_author_overlap"

    return {
        "accepted": accepted,
        "reason": reason,
        "title_similarity": round(title_score, 4),
        "author_overlap": round(overlap, 4),
        "year_distance": distance,
        "semantic_scholar_match_score": match_score,
        "candidate_title": candidate.get("title"),
        "candidate_year": candidate.get("year"),
        "candidate_authors": candidate_authors[:10],
    }


def extract_doi(candidate: dict[str, Any]) -> str:
    external_ids = candidate.get("externalIds")
    if not isinstance(external_ids, dict):
        return ""
    return normalize_doi(external_ids.get("DOI") or external_ids.get("doi"))


def extract_keywords(candidate: dict[str, Any]) -> list[str]:
    keywords: list[str] = []
    fields = candidate.get("fieldsOfStudy") or []
    if isinstance(fields, list):
        for value in fields:
            text = normalize_text(value)
            if text and text not in keywords:
                keywords.append(text)
    s2_fields = candidate.get("s2FieldsOfStudy") or []
    if isinstance(s2_fields, list):
        for item in s2_fields:
            if isinstance(item, dict):
                text = normalize_text(item.get("category"))
                if text and text not in keywords:
                    keywords.append(text)
            elif isinstance(item, str):
                text = normalize_text(item)
                if text and text not in keywords:
                    keywords.append(text)
    return keywords


def cache_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "paperId": candidate.get("paperId"),
        "externalIds": candidate.get("externalIds"),
        "title": candidate.get("title"),
        "abstract": normalize_text(candidate.get("abstract")),
        "citationCount": candidate.get("citationCount"),
        "year": candidate.get("year"),
        "authors": s2_authors(candidate),
        "fieldsOfStudy": extract_keywords(candidate),
        "venue": candidate.get("venue"),
        "publicationTypes": candidate.get("publicationTypes"),
        "matchScore": candidate.get("matchScore"),
    }


def apply_candidate(paper: dict[str, Any], candidate: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    updated = dict(paper)
    fields: list[str] = []

    doi = extract_doi(candidate)
    if doi and not normalize_doi(updated.get("doi")):
        updated["doi"] = doi
        updated["paper_uid"] = f"doi:{doi.casefold()}"
        fields.append("doi")

    abstract = normalize_text(candidate.get("abstract"))
    if abstract and not normalize_text(updated.get("abstract")):
        updated["abstract"] = abstract
        fields.append("abstract")

    keywords = extract_keywords(candidate)
    if keywords and not updated.get("keywords"):
        updated["keywords"] = keywords
        fields.append("keywords")

    citation = candidate.get("citationCount")
    old_citation_field = "citation" + "Counts"
    if citation is not None:
        if updated.get("citation_count") is None:
            updated["citation_count"] = citation
            fields.append("citation_count")
        if updated.get("citation_counts") is None:
            updated["citation_counts"] = citation
        if updated.get(old_citation_field) is None:
            updated[old_citation_field] = citation

    return updated, fields


def is_empty_ok_candidate(item: dict[str, Any]) -> bool:
    if item.get("status") != "ok":
        return False
    candidate = item.get("candidate")
    if not isinstance(candidate, dict):
        return False
    return not any(
        normalize_text(candidate.get(field))
        for field in ("paperId", "title")
    ) and not isinstance(candidate.get("externalIds"), dict)


class SearchCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.items: dict[str, dict[str, Any]] = {}
        if not path.exists():
            return
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
                if is_empty_ok_candidate(item):
                    continue
                key = item.get("key")
                if key:
                    self.items[str(key)] = item

    def get(self, key: str) -> dict[str, Any] | None:
        return self.items.get(key)

    def put(self, key: str, item: dict[str, Any]) -> None:
        if item.get("status") == "rate_limited":
            return
        if key in self.items:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        item = dict(item)
        item["cache_schema_version"] = CACHE_SCHEMA_VERSION
        item["key"] = key
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(item, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        self.items[key] = item


def fetch_match(
    paper: dict[str, Any],
    args: argparse.Namespace,
    session: requests.Session,
) -> dict[str, Any]:
    title = normalize_text(paper.get("title"))
    if not title:
        return {"status": "no_query", "error": "", "candidate": None}

    params: dict[str, Any] = {"query": title, "fields": S2_FIELDS}
    year = normalize_text(paper.get("year"))
    if year.isdigit() and args.year_window >= 0:
        year_int = int(year)
        start = year_int - args.year_window
        end = year_int + args.year_window
        params["year"] = str(year_int) if start == end else f"{start}-{end}"

    headers = {}
    if args.api_key:
        headers["x-api-key"] = args.api_key
    elif args.use_empty_api_key_header:
        headers["x-api-key"] = ""

    last_error = ""
    for attempt in range(args.max_retries + 1):
        try:
            response = session.get(
                S2_MATCH_URL,
                params=params,
                headers=headers,
                timeout=args.request_timeout,
            )
        except requests.RequestException as exc:
            last_error = str(exc)
        else:
            if response.status_code == 404:
                return {"status": "not_found", "error": "", "candidate": None}
            if response.status_code == 429:
                retry_after = response.headers.get("retry-after") or response.headers.get("Retry-After") or ""
                if attempt < args.max_retries:
                    sleep_seconds = args.rate_limit_retry_sleep
                    if retry_after:
                        try:
                            sleep_seconds = max(float(retry_after), args.rate_limit_retry_sleep)
                        except ValueError:
                            pass
                    print(
                        f"Semantic Scholar 429; sleeping {sleep_seconds:.1f}s before retry "
                        f"{attempt + 1}/{args.max_retries}",
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
            if response.status_code in {500, 502, 503, 504}:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            else:
                try:
                    response.raise_for_status()
                    payload = response.json()
                except Exception as exc:
                    return {"status": "error", "error": str(exc), "candidate": None}
                if not isinstance(payload, dict):
                    return {"status": "error", "error": f"Unexpected response: {payload!r}", "candidate": None}
                data = payload.get("data")
                if not isinstance(data, list) or not data:
                    return {"status": "not_found", "error": "", "candidate": None}
                candidate = data[0]
                if not isinstance(candidate, dict):
                    return {"status": "error", "error": f"Unexpected candidate: {candidate!r}", "candidate": None}
                return {"status": "ok", "error": "", "candidate": cache_candidate(candidate)}

        if attempt < args.max_retries:
            time.sleep(args.retry_backoff * (2**attempt))

    return {"status": "error", "error": last_error, "candidate": None}


def append_by_source(bucket: dict[str, list[dict[str, Any]]], paper: dict[str, Any]) -> None:
    source = normalize_text(paper.get("source")) or "Unknown"
    bucket.setdefault(source, []).append(paper)


def write_source_outputs(directory: Path, by_source: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    output_counts: dict[str, int] = {}
    for source, papers in sorted(by_source.items()):
        path = directory / safe_filename(source)
        write_json_array_atomic(path, papers)
        output_counts[str(path)] = len(papers)
    return output_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover missing DOI papers by searching Semantic Scholar by title."
    )
    parser.add_argument("--papers-dir", type=Path, default=Path("data/papers"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--year-window", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=1.0)
    parser.add_argument(
        "--rate-limit-retry-sleep",
        type=float,
        default=60.0,
        help="Seconds to wait before retrying a Semantic Scholar 429 with no Retry-After header.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.sleep < 1.0:
        parser.error("--sleep must be >= 1.0 to respect the 1 request/second API-key limit")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be > 0")
    if args.rate_limit_retry_sleep < 0:
        parser.error("--rate-limit-retry-sleep must be >= 0")
    if args.year_window < 0:
        parser.error("--year-window must be >= 0")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    if not args.api_key and args.use_env_api_key:
        args.api_key = os.environ.get("S2_API_KEY", "") or os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        if not args.api_key:
            args.api_key = load_dotenv_key(args.env_file, ["S2_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"])

    if args.output_dir is None:
        args.output_dir = args.papers_dir / f"update{time.strftime('%Y%m%d')}"
    if args.cache is None:
        args.cache = args.output_dir / "cache" / "semantic_scholar_missing_doi_search_cache.jsonl"
    args.missing_doi_path = args.papers_dir / "missing" / "_missing_doi.json"
    args.output_enriched_dir = args.output_dir / "enriched"
    args.output_missing_dir = args.output_dir / "missing"
    return args


def main() -> int:
    args = parse_args()
    print(
        "Semantic Scholar title-search auth mode: " + ("api_key" if args.api_key else "anonymous"),
        file=sys.stderr,
    )
    print(f"input: {args.missing_doi_path}", file=sys.stderr)
    print(f"output: {args.output_dir}", file=sys.stderr)

    cache = SearchCache(args.cache)
    session = requests.Session()
    papers = load_json_array(args.missing_doi_path)
    update_enriched: dict[str, list[dict[str, Any]]] = {}
    update_missing: dict[str, list[dict[str, Any]]] = {}
    still_missing_doi: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    started_at = time.monotonic()
    stopped_reason = ""

    stats = Counter()
    stats["input_missing_doi"] = len(papers)

    def should_process(paper: dict[str, Any]) -> bool:
        if args.source and normalize_text(paper.get("source")) != args.source:
            return False
        return True

    def print_progress(force: bool = False) -> None:
        if not force and (
            not args.progress_every or stats["processed"] % args.progress_every != 0
        ):
            return
        elapsed = max(time.monotonic() - started_at, 0.001)
        print(
            "progress: "
            f"processed={stats['processed']} "
            f"searched={stats['searched']} "
            f"cache_hits={stats['cache_hits']} "
            f"matched={stats['matched']} "
            f"recovered_doi={stats['recovered_doi']} "
            f"to_enriched={stats['to_enriched']} "
            f"to_missing={stats['to_missing']} "
            f"still_missing_doi={stats['still_missing_doi']} "
            f"errors={stats['errors']} "
            f"rate={stats['processed'] / elapsed:.2f}/s",
            file=sys.stderr,
        )

    for index, paper in enumerate(papers):
        if not should_process(paper):
            still_missing_doi.append(paper)
            stats["skipped_source"] += 1
            continue
        if args.limit is not None and stats["processed"] >= args.limit:
            still_missing_doi.append(paper)
            stats["skipped_limit"] += 1
            continue

        stats["processed"] += 1
        key = search_key(paper)
        cache_item = cache.get(key)
        if cache_item is None:
            try:
                cache_item = fetch_match(paper, args, session)
            except (RateLimited, PermissionError) as exc:
                stopped_reason = str(exc)
                print(stopped_reason, file=sys.stderr)
                still_missing_doi.append(paper)
                still_missing_doi.extend(papers[index + 1 :])
                stats["stopped_unprocessed"] += len(papers) - index
                break
            cache.put(key, cache_item)
            stats["searched"] += 1
            if args.sleep:
                time.sleep(args.sleep)
        else:
            stats["cache_hits"] += 1

        status = cache_item.get("status")
        if status not in {"ok", "not_found", "no_query"}:
            stats["errors"] += 1

        candidate = cache_item.get("candidate")
        if not isinstance(candidate, dict):
            still_missing_doi.append(paper)
            stats["still_missing_doi"] += 1
            print_progress()
            continue

        match_info = candidate_score(paper, candidate)
        if not match_info["accepted"]:
            still_missing_doi.append(paper)
            stats[f"rejected_{match_info['reason']}"] += 1
            stats["still_missing_doi"] += 1
            print_progress()
            continue

        stats["matched"] += 1
        updated, fields = apply_candidate(paper, candidate)
        for field in fields:
            stats[f"field_{field}"] += 1

        if normalize_doi(updated.get("doi")):
            stats["recovered_doi"] += 1
            if normalize_text(updated.get("abstract")):
                append_by_source(update_enriched, updated)
                stats["to_enriched"] += 1
            else:
                append_by_source(update_missing, updated)
                stats["to_missing"] += 1
        else:
            still_missing_doi.append(updated)
            stats["still_missing_doi"] += 1
            if fields:
                stats["updated_still_missing_doi"] += 1

        if len(samples) < 20:
            samples.append(
                {
                    "title": paper.get("title"),
                    "year": paper.get("year"),
                    "source": paper.get("source"),
                    "dblp_key": paper.get("dblp_key"),
                    "fields": fields,
                    "doi": updated.get("doi"),
                    "semantic_scholar_paper_id": candidate.get("paperId"),
                    "match": match_info,
                }
            )
        print_progress()

    outputs: dict[str, Any] = {}
    if not args.dry_run:
        args.output_enriched_dir.mkdir(parents=True, exist_ok=True)
        args.output_missing_dir.mkdir(parents=True, exist_ok=True)
        outputs["enriched"] = write_source_outputs(args.output_enriched_dir, update_enriched)
        outputs["missing"] = write_source_outputs(args.output_missing_dir, update_missing)
        missing_doi_out = args.output_missing_dir / "_missing_doi.json"
        write_json_array_atomic(missing_doi_out, still_missing_doi)
        outputs["missing_doi"] = {str(missing_doi_out): len(still_missing_doi)}

    stats["output_missing_doi"] = len(still_missing_doi)
    manifest = {
        "papers_dir": str(args.papers_dir),
        "missing_doi": str(args.missing_doi_path),
        "output_dir": str(args.output_dir),
        "cache": str(args.cache),
        "auth_mode": "api_key" if args.api_key else "anonymous",
        "source": args.source or None,
        "limit": args.limit,
        "year_window": args.year_window,
        "sleep": args.sleep,
        "dry_run": args.dry_run,
        "stopped_reason": stopped_reason,
        "stats": dict(stats),
        "outputs": outputs,
        "samples": samples,
    }
    suffix = "_dry_run" if args.dry_run else ""
    manifest_path = args.output_dir / f"semantic_scholar_missing_doi_search_manifest{suffix}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_progress(force=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
