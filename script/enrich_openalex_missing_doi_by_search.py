#!/usr/bin/env python3
"""Use OpenAlex search to recover DOI/metadata for missing DOI papers.

The script reads data/papers/missing/_missing_doi.json, searches OpenAlex Works
by title, validates candidates locally with title/year/author checks, and then
updates paper records. Records with a recovered DOI move out of _missing_doi:

- DOI + abstract -> data/papers/enriched/{source}.json
- DOI without abstract -> data/papers/missing/{source}.json

Records that still lack DOI remain in _missing_doi, but any safely matched
metadata such as abstract, keywords, and citation_count is preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import requests


CACHE_SCHEMA_VERSION = 3
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
DEFAULT_SELECT = ",".join(
    [
        "id",
        "doi",
        "display_name",
        "publication_year",
        "authorships",
        "abstract_inverted_index",
        "keywords",
        "topics",
        "concepts",
        "cited_by_count",
        "type",
    ]
)


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


def load_dotenv_key(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == key:
                return value.strip().strip("'\"")
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


def work_authors(work: dict[str, Any]) -> list[str]:
    authors: list[str] = []
    authorships = work.get("authorships")
    if not isinstance(authorships, list):
        return authors
    for authorship in authorships:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author")
        if not isinstance(author, dict):
            continue
        name = normalize_text(author.get("display_name"))
        if name:
            authors.append(name)
    return authors


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
    tokens = title.split()
    if len(tokens) <= 2:
        return True
    generic_titles = {
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
    return title in generic_titles


def year_distance(paper: dict[str, Any], work: dict[str, Any]) -> int | None:
    paper_year = normalize_text(paper.get("year"))
    work_year = work.get("publication_year")
    try:
        return abs(int(paper_year) - int(work_year))
    except (TypeError, ValueError):
        return None


def candidate_score(paper: dict[str, Any], work: dict[str, Any]) -> dict[str, Any]:
    display_name = work.get("display_name") or work.get("title")
    title_score = title_similarity(paper.get("title"), display_name)
    authors = work.get("authors") if isinstance(work.get("authors"), list) else work_authors(work)
    overlap = author_overlap(paper.get("authors"), authors)
    distance = year_distance(paper, work)
    has_year_match = distance is not None and distance <= 1

    accepted = False
    reason = "below_threshold"
    generic_title = is_generic_title(paper.get("title"))
    if generic_title and overlap <= 0:
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
        "candidate_title": display_name,
        "candidate_year": work.get("publication_year"),
        "candidate_authors": authors[:10],
    }


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
                key = item.get("key")
                if key:
                    self.items[str(key)] = item

    def get(self, key: str) -> dict[str, Any] | None:
        return self.items.get(key)

    def put(self, key: str, item: dict[str, Any]) -> None:
        if item.get("status") in {"rate_limited", "key_unavailable"}:
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


@dataclass
class OpenAlexKey:
    label: str
    value: str
    requests: int = 0
    exhausted: bool = False
    exhausted_reason: str = ""


@dataclass
class OpenAlexKeyPool:
    keys: list[OpenAlexKey] = field(default_factory=list)
    active_index: int = 0

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "OpenAlexKeyPool":
        keys: list[OpenAlexKey] = []
        seen_values: set[str] = set()

        def add(label: str, value: str) -> None:
            value = normalize_text(value)
            if not value or value in seen_values:
                return
            seen_values.add(value)
            keys.append(OpenAlexKey(label=label, value=value))

        if args.api_key:
            add("cli:--api-key", args.api_key)
        if args.use_env_api_key:
            add("env:OPENALEX_API_KEY", os.environ.get("OPENALEX_API_KEY", ""))
            add("dotenv:OPENALEX_API_KEY", load_dotenv_key(args.env_file, "OPENALEX_API_KEY"))
            for index in range(1, 7):
                name = f"OPENALEX_API_KEY{index}"
                add(f"env:{name}", os.environ.get(name, ""))
                add(f"dotenv:{name}", load_dotenv_key(args.env_file, name))

        if not keys:
            keys.append(OpenAlexKey(label="anonymous", value=""))
        return cls(keys=keys)

    def current(self) -> OpenAlexKey:
        return self.keys[self.active_index]

    def active_label(self) -> str:
        return self.current().label

    def active_value(self) -> str:
        return self.current().value

    def mark_request(self) -> None:
        self.current().requests += 1

    def mark_exhausted(self, reason: str) -> bool:
        key = self.current()
        key.exhausted = True
        key.exhausted_reason = reason
        for index in range(self.active_index + 1, len(self.keys)):
            if not self.keys[index].exhausted:
                self.active_index = index
                return True
        return False

    def manifest(self) -> dict[str, Any]:
        return {
            "active": self.active_label(),
            "available": sum(1 for key in self.keys if not key.exhausted),
            "keys": [
                {
                    "label": key.label,
                    "requests": key.requests,
                    "exhausted": key.exhausted,
                    "exhausted_reason": key.exhausted_reason,
                }
                for key in self.keys
            ],
        }


def work_to_cache_work(work: dict[str, Any]) -> dict[str, Any]:
    abstract = normalize_text(work.get("abstract"))
    if not abstract:
        abstract = abstract_from_inverted_index(work.get("abstract_inverted_index"))
    keywords = extract_names(work.get("keywords"))
    if not keywords:
        keywords = extract_names(work.get("topics"))
    if not keywords:
        keywords = extract_names(work.get("concepts"))
    return {
        "id": work.get("id"),
        "doi": normalize_doi(work.get("doi")),
        "display_name": work.get("display_name") or work.get("title"),
        "publication_year": work.get("publication_year"),
        "authors": work_authors(work),
        "abstract": abstract,
        "keywords": keywords,
        "cited_by_count": work.get("cited_by_count"),
        "type": work.get("type"),
    }


def fetch_candidates(
    paper: dict[str, Any],
    args: argparse.Namespace,
    session: requests.Session,
    api_key: str,
    api_key_label: str,
) -> dict[str, Any]:
    title = normalize_title(paper.get("title"))
    if not title:
        return {"status": "no_query", "error": "", "candidates": []}

    params: dict[str, Any] = {
        "search": title,
        "per_page": args.per_page,
        "select": DEFAULT_SELECT,
    }
    filters: list[str] = []
    year = normalize_text(paper.get("year"))
    if year.isdigit() and args.year_window >= 0:
        year_int = int(year)
        if args.year_window == 0:
            filters.append(f"publication_year:{year_int}")
        else:
            filters.append(f"publication_year:>{year_int - args.year_window - 1}")
            filters.append(f"publication_year:<{year_int + args.year_window + 1}")
    if filters:
        params["filter"] = ",".join(filters)
    if api_key:
        params["api_key"] = api_key
    if args.mailto:
        params["mailto"] = args.mailto

    last_error = ""
    for attempt in range(args.max_retries + 1):
        try:
            response = session.get(OPENALEX_WORKS_URL, params=params, timeout=args.request_timeout)
        except requests.RequestException as exc:
            last_error = str(exc)
        else:
            if response.status_code in {402, 403, 429}:
                retry_after = response.headers.get("Retry-After", "")
                return {
                    "status": "rate_limited" if response.status_code == 429 else "key_unavailable",
                    "error": f"HTTP {response.status_code} on {api_key_label}: {response.text[:300]}",
                    "retry_after": retry_after,
                    "api_key_label": api_key_label,
                    "candidates": [],
                }
            if response.status_code in {500, 503}:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            else:
                try:
                    response.raise_for_status()
                    payload = response.json()
                except Exception as exc:
                    return {"status": "error", "error": str(exc), "candidates": []}
                results = payload.get("results")
                if not isinstance(results, list):
                    return {"status": "error", "error": "Missing results array", "candidates": []}
                return {
                    "status": "ok",
                    "error": "",
                    "count": payload.get("meta", {}).get("count"),
                    "candidates": [work_to_cache_work(work) for work in results if isinstance(work, dict)],
                }

        if attempt < args.max_retries:
            time.sleep(args.retry_backoff * (2**attempt))

    return {"status": "error", "error": last_error, "candidates": []}


def choose_match(paper: dict[str, Any], cache_item: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    candidates = cache_item.get("candidates")
    if not isinstance(candidates, list):
        return None, {"accepted": False, "reason": "no_candidates"}

    scored: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        score = candidate_score(paper, candidate)
        scored.append((candidate, score))
    accepted = [(candidate, score) for candidate, score in scored if score["accepted"]]
    accepted.sort(
        key=lambda item: (
            item[1]["title_similarity"],
            item[1]["author_overlap"],
            -(item[1]["year_distance"] or 99),
        ),
        reverse=True,
    )
    if not accepted:
        best_score = max((score for _, score in scored), key=lambda item: item["title_similarity"], default=None)
        return None, best_score or {"accepted": False, "reason": "no_candidates"}

    best_candidate, best_score = accepted[0]
    if len(accepted) > 1:
        second_score = accepted[1][1]
        if (
            best_score["title_similarity"] - second_score["title_similarity"] < 0.02
            and best_score["author_overlap"] - second_score["author_overlap"] < 0.25
        ):
            details = dict(best_score)
            details["accepted"] = False
            details["reason"] = "ambiguous_candidates"
            return None, details
    return best_candidate, best_score


def apply_work(paper: dict[str, Any], work: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    updated = dict(paper)
    fields: list[str] = []

    doi = normalize_doi(work.get("doi"))
    if doi and not normalize_doi(updated.get("doi")):
        updated["doi"] = doi
        updated["paper_uid"] = f"doi:{doi.casefold()}"
        fields.append("doi")

    abstract = normalize_text(work.get("abstract"))
    if abstract and not normalize_text(updated.get("abstract")):
        updated["abstract"] = abstract
        fields.append("abstract")

    keywords = work.get("keywords")
    if keywords and not updated.get("keywords"):
        updated["keywords"] = keywords
        fields.append("keywords")

    citation = work.get("cited_by_count")
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


def key_for_merge(paper: dict[str, Any]) -> str:
    dblp_key = normalize_text(paper.get("dblp_key"))
    if dblp_key:
        return f"dblp_key:{dblp_key}"
    paper_uid = normalize_text(paper.get("paper_uid"))
    if paper_uid:
        return f"paper_uid:{paper_uid}"
    doi = normalize_doi(paper.get("doi")).casefold()
    if doi:
        return f"doi:{doi}"
    return ""


def merge_records(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    merged = list(existing)
    index: dict[str, int] = {}
    for pos, paper in enumerate(merged):
        key = key_for_merge(paper)
        if key:
            index[key] = pos

    added = 0
    replaced = 0
    for paper in incoming:
        key = key_for_merge(paper)
        if key and key in index:
            merged[index[key]] = paper
            replaced += 1
        else:
            if key:
                index[key] = len(merged)
            merged.append(paper)
            added += 1
    return merged, added, replaced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover missing DOI papers by searching OpenAlex Works by title."
    )
    parser.add_argument("--papers-dir", type=Path, default=Path("data/papers"))
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--use-env-api-key", action="store_true")
    parser.add_argument("--mailto", default="")
    parser.add_argument("--source", default="", help="Optional source name to process.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-page", type=int, default=5)
    parser.add_argument("--year-window", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=20.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Write progress JSON after this many processed papers. Use 0 to disable.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.per_page < 1 or args.per_page > 100:
        parser.error("--per-page must be between 1 and 100")
    if args.sleep < 0:
        parser.error("--sleep must be >= 0")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be > 0")
    if args.cache is None:
        args.cache = args.papers_dir / "cache" / "openalex_missing_doi_search_cache.jsonl"
    args.missing_dir = args.papers_dir / "missing"
    args.enriched_dir = args.papers_dir / "enriched"
    args.missing_doi_path = args.missing_dir / "_missing_doi.json"
    args.progress_path = args.papers_dir / "openalex_missing_doi_search_progress.json"
    return args


def main() -> int:
    args = parse_args()
    cache = SearchCache(args.cache)
    key_pool = OpenAlexKeyPool.from_args(args)
    session = requests.Session()
    missing_doi = load_json_array(args.missing_doi_path)
    remaining_missing_doi: list[dict[str, Any]] = []
    moved_enriched: dict[str, list[dict[str, Any]]] = {}
    moved_missing: dict[str, list[dict[str, Any]]] = {}
    samples: list[dict[str, Any]] = []
    started_at = time.monotonic()
    stopped_reason = ""

    stats = Counter()
    stats["input_missing_doi"] = len(missing_doi)
    last_progress: dict[str, Any] = {
        "index": None,
        "dblp_key": None,
        "paper_uid": None,
        "title": None,
        "source": None,
    }

    def should_process(paper: dict[str, Any]) -> bool:
        if args.source and normalize_text(paper.get("source")) != args.source:
            return False
        return True

    def progress(force: bool = False) -> None:
        if not force and (
            not args.progress_every or stats["processed"] % args.progress_every != 0
        ):
            return
        elapsed = max(time.monotonic() - started_at, 0.001)
        rate = stats["processed"] / elapsed
        print(
            "progress: "
            f"processed={stats['processed']} "
            f"searched={stats['searched']} "
            f"cache_hits={stats['cache_hits']} "
            f"matched={stats['matched']} "
            f"recovered_doi={stats['recovered_doi']} "
            f"moved_enriched={stats['moved_enriched']} "
            f"moved_missing={stats['moved_missing']} "
            f"still_missing_doi={stats['still_missing_doi']} "
            f"errors={stats['errors']} "
            f"key={key_pool.active_label()} "
            f"rate={rate:.2f}/s",
            file=sys.stderr,
        )

    def write_progress(force: bool = False) -> None:
        if not force and (
            not args.checkpoint_every
            or stats["processed"] % args.checkpoint_every != 0
        ):
            return
        payload = {
            "papers_dir": str(args.papers_dir),
            "missing_doi": str(args.missing_doi_path),
            "cache": str(args.cache),
            "dry_run": args.dry_run,
            "source": args.source or None,
            "limit": args.limit,
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "last_progress": last_progress,
            "stats": dict(stats),
            "key_pool": key_pool.manifest(),
        }
        args.progress_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    stop_index: int | None = None
    for index, paper in enumerate(missing_doi):
        if not should_process(paper):
            remaining_missing_doi.append(paper)
            stats["skipped_source"] += 1
            continue
        if args.limit is not None and stats["processed"] >= args.limit:
            remaining_missing_doi.append(paper)
            stats["skipped_limit"] += 1
            continue

        stats["processed"] += 1
        last_progress.update(
            {
                "index": index,
                "dblp_key": paper.get("dblp_key"),
                "paper_uid": paper.get("paper_uid"),
                "title": paper.get("title"),
                "source": paper.get("source"),
            }
        )
        key = search_key(paper)
        cache_item = cache.get(key)
        if cache_item is None:
            while True:
                active_key = key_pool.current()
                cache_item = fetch_candidates(
                    paper,
                    args,
                    session,
                    api_key=active_key.value,
                    api_key_label=active_key.label,
                )
                key_pool.mark_request()
                stats["searched"] += 1
                status = cache_item.get("status")
                if status in {"rate_limited", "key_unavailable"}:
                    stats["key_rotations"] += 1
                    reason = normalize_text(cache_item.get("error")) or str(status)
                    switched = key_pool.mark_exhausted(reason)
                    print(
                        f"OpenAlex key exhausted: {active_key.label}; "
                        f"switched={switched}; reason={reason[:180]}",
                        file=sys.stderr,
                    )
                    write_progress(force=True)
                    if switched:
                        continue
                    stopped_reason = "All OpenAlex API keys are exhausted or unavailable"
                    remaining_missing_doi.append(paper)
                    stop_index = index
                    break
                cache.put(key, cache_item)
                break
            if stop_index is not None:
                break
            if args.sleep:
                time.sleep(args.sleep)
        else:
            stats["cache_hits"] += 1

        status = cache_item.get("status")
        if status in {"rate_limited", "key_unavailable"}:
            stopped_reason = cache_item.get("error") or "OpenAlex rate limited the run"
            remaining_missing_doi.append(paper)
            stop_index = index
            break
        if status not in {"ok", "no_query"}:
            stats["errors"] += 1

        work, match_info = choose_match(paper, cache_item)
        if not work:
            remaining_missing_doi.append(paper)
            stats["still_missing_doi"] += 1
            progress()
            write_progress()
            continue

        stats["matched"] += 1
        updated, fields = apply_work(paper, work)
        if fields:
            for field in fields:
                stats[f"field_{field}"] += 1
        if normalize_doi(updated.get("doi")):
            stats["recovered_doi"] += 1
            source = normalize_text(updated.get("source")) or "Unknown"
            if normalize_text(updated.get("abstract")):
                moved_enriched.setdefault(source, []).append(updated)
                stats["moved_enriched"] += 1
            else:
                moved_missing.setdefault(source, []).append(updated)
                stats["moved_missing"] += 1
        else:
            remaining_missing_doi.append(updated)
            stats["still_missing_doi"] += 1

        if len(samples) < 20:
            samples.append(
                {
                    "title": paper.get("title"),
                    "year": paper.get("year"),
                    "source": paper.get("source"),
                    "dblp_key": paper.get("dblp_key"),
                    "fields": fields,
                    "doi": updated.get("doi"),
                    "match": match_info,
                    "openalex_id": work.get("id"),
                }
            )
        progress()
        write_progress()

    if stop_index is not None:
        for paper in missing_doi[stop_index + 1 :]:
            remaining_missing_doi.append(paper)

    output_stats: dict[str, Any] = {}
    if not args.dry_run:
        write_json_array_atomic(args.missing_doi_path, remaining_missing_doi)
        for source, papers in moved_enriched.items():
            path = args.enriched_dir / safe_filename(source)
            existing = load_json_array(path)
            merged, added, replaced = merge_records(existing, papers)
            write_json_array_atomic(path, merged)
            output_stats[f"enriched/{path.name}"] = {
                "incoming": len(papers),
                "added": added,
                "replaced": replaced,
            }
        for source, papers in moved_missing.items():
            path = args.missing_dir / safe_filename(source)
            existing = load_json_array(path)
            merged, added, replaced = merge_records(existing, papers)
            write_json_array_atomic(path, merged)
            output_stats[f"missing/{path.name}"] = {
                "incoming": len(papers),
                "added": added,
                "replaced": replaced,
            }

    stats["output_missing_doi"] = len(remaining_missing_doi)
    manifest = {
        "papers_dir": str(args.papers_dir),
        "missing_doi": str(args.missing_doi_path),
        "cache": str(args.cache),
        "auth_mode": "api_key" if any(key.value for key in key_pool.keys) else "anonymous",
        "source": args.source or None,
        "limit": args.limit,
        "dry_run": args.dry_run,
        "stopped_reason": stopped_reason,
        "stats": dict(stats),
        "key_pool": key_pool.manifest(),
        "last_progress": last_progress,
        "outputs": output_stats,
        "samples": samples,
    }
    suffix = "_dry_run" if args.dry_run else ""
    manifest_path = args.papers_dir / f"openalex_missing_doi_search_manifest{suffix}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_progress(force=True)
    progress(force=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
