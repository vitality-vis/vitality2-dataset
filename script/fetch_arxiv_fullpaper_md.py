#!/usr/bin/env python3
"""Find arXiv full-paper HTML and convert it to Markdown.

The script writes Markdown files locally and emits one JSONL result per scanned
paper. It does not write to Zilliz directly; the result rows include
``paper_uid`` and ``fullpaper_status`` so they can be used for a later dynamic
field partial upsert.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup, NavigableString, Tag


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
DEFAULT_OUTPUT_DIR = DEFAULT_PAPERS_DIR / "fullpaper"
S2_FIELDS = "paperId,externalIds,title,year,authors"
S2_API_KEY_ENV_NAMES = ("SEMANTIC_SCHOLAR_API_KEY", "S2_API_KEY")
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
STATUS_FIELD = "fullpaper_status"
PROCESS_STATUS = "not_searched"
RETRYABLE_FAILED_STATUSES = {"failed", "failed_s2", "failed_arxivsearch", "failed_openalex"}
SKIP_STATUSES = {"found", "not_found", "not_applicable", *RETRYABLE_FAILED_STATUSES}
ARXIV_LINK_RE = re.compile(
    r"(?:https?://)?arxiv\.org/(?:abs|html|pdf)/"
    r"(?P<id>[A-Za-z.-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_doi(value: Any) -> str:
    doi = normalize_text(value)
    lower = doi.casefold()
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


def normalize_title(value: Any) -> str:
    text = normalize_text(value).casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return normalize_text(text)


def normalize_arxiv_id(value: str) -> str:
    arxiv_id = normalize_text(value)
    arxiv_id = re.sub(r"^arxiv:", "", arxiv_id, flags=re.IGNORECASE)
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id, flags=re.IGNORECASE)
    return arxiv_id


def arxiv_id_from_doi(doi: Any) -> str:
    value = normalize_doi(doi)
    match = re.match(r"10\.48550/arxiv\.(.+)$", value, flags=re.IGNORECASE)
    if not match:
        return ""
    raw = match.group(1)
    if "/" in raw:
        category, identifier = raw.split("/", 1)
        return normalize_arxiv_id(f"{category.casefold()}/{identifier}")
    return normalize_arxiv_id(raw)


def arxiv_id_from_text(*values: Any) -> str:
    for value in values:
        match = ARXIV_LINK_RE.search(str(value or ""))
        if match:
            return normalize_arxiv_id(match.group("id"))
    return ""


def arxiv_id_from_openalex_location(location: dict[str, Any]) -> str:
    location_id = normalize_text(location.get("id"))
    match = re.match(r"pmh:oai:arXiv\.org:(.+)$", location_id, flags=re.IGNORECASE)
    if match:
        return normalize_arxiv_id(match.group(1))
    arxiv_id = arxiv_id_from_text(location.get("landing_page_url"), location.get("pdf_url"))
    if arxiv_id:
        return arxiv_id
    source = location.get("source") or {}
    source_name = normalize_text(source.get("display_name"))
    if "arxiv" in source_name.casefold():
        return arxiv_id_from_text(location_id, location.get("landing_page_url"), location.get("pdf_url"))
    return ""


def arxiv_id_from_openalex_work(data: dict[str, Any]) -> str:
    for key in ("primary_location", "best_oa_location"):
        location = data.get(key)
        if isinstance(location, dict):
            arxiv_id = arxiv_id_from_openalex_location(location)
            if arxiv_id:
                return arxiv_id
    for location in data.get("locations") or []:
        if isinstance(location, dict):
            arxiv_id = arxiv_id_from_openalex_location(location)
            if arxiv_id:
                return arxiv_id
    return ""


def safe_filename(value: Any) -> str:
    safe = re.sub(r"[^A-Za-z0-9._ -]+", "_", normalize_text(value))
    safe = re.sub(r"\s+", " ", safe).strip(" ._")
    return (safe or "unknown")[:180]


def load_json_array(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON array: {path}")
    return data


def load_dotenv_key(path: Path, names: tuple[str, ...]) -> str:
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


def iter_default_input_files(papers_dir: Path) -> Iterable[Path]:
    for folder_name in ("enriched", "missing"):
        folder = papers_dir / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.json")):
            if path.name.startswith("_"):
                continue
            yield path


class JsonlCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.items: dict[str, dict[str, Any]] = {}
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("status_code") not in {200, 404}:
                    continue
                key = normalize_doi(item.get("doi")).casefold()
                if key:
                    self.items[key] = item

    def get(self, doi: Any) -> dict[str, Any] | None:
        return self.items.get(normalize_doi(doi).casefold())

    def append(self, item: dict[str, Any]) -> None:
        key = normalize_doi(item.get("doi")).casefold()
        if key:
            self.items[key] = item
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def semantic_scholar_arxiv_id(
    session: requests.Session,
    doi: str,
    *,
    cache: JsonlCache,
    timeout: float,
    sleep: float,
    max_retries: int,
    retry_backoff: float,
    rate_limit_retry_sleep: float,
    verbose: bool,
) -> tuple[str, str]:
    cached = cache.get(doi)
    if cached is not None:
        return normalize_arxiv_id(cached.get("arxiv_id") or ""), ""

    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{quote(normalize_doi(doi), safe='')}"
    item: dict[str, Any] = {"doi": normalize_doi(doi), "service": "semantic_scholar", "arxiv_id": ""}
    for attempt in range(max_retries + 1):
        try:
            response = session.get(url, params={"fields": S2_FIELDS}, timeout=timeout)
            item["status_code"] = response.status_code
            if response.status_code == 200:
                data = response.json()
                external_ids = data.get("externalIds") or {}
                item["arxiv_id"] = normalize_arxiv_id(external_ids.get("ArXiv") or "")
                item.pop("error", None)
                break
            if response.status_code == 404:
                item["error"] = response.text[:500]
                break
            if response.status_code == 429:
                item["error"] = response.text[:500]
                if attempt < max_retries:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = float(retry_after) if retry_after else rate_limit_retry_sleep
                    except ValueError:
                        delay = rate_limit_retry_sleep
                    if verbose:
                        print(
                            f"Semantic Scholar 429 for DOI {normalize_doi(doi)}; "
                            f"retry {attempt + 1}/{max_retries} after {delay:g}s",
                            file=sys.stderr,
                            flush=True,
                        )
                    time.sleep(max(delay, 0.0))
                    continue
                break
            response.raise_for_status()
        except requests.RequestException as exc:
            item["error"] = str(exc)
            if attempt < max_retries:
                delay = max(retry_backoff * (2**attempt), 0.0)
                if verbose:
                    print(
                        f"Semantic Scholar request failed for DOI {normalize_doi(doi)}: {exc}; "
                        f"retry {attempt + 1}/{max_retries} after {delay:g}s",
                        file=sys.stderr,
                        flush=True,
                    )
                time.sleep(delay)
                continue
            break
    if item.get("status_code") in {200, 404}:
        cache.append(item)
    if sleep > 0:
        time.sleep(sleep)
    error = normalize_text(item.get("error"))
    if item.get("status_code") not in {200, 404} and not error:
        error = f"Semantic Scholar returned HTTP {item.get('status_code')}"
    return normalize_arxiv_id(item.get("arxiv_id") or ""), error


def openalex_arxiv_id(
    session: requests.Session,
    doi: str,
    *,
    cache: JsonlCache,
    timeout: float,
    sleep: float,
    max_retries: int,
    retry_backoff: float,
    rate_limit_retry_sleep: float,
    verbose: bool,
) -> tuple[str, str]:
    cached = cache.get(doi)
    if cached is not None:
        return normalize_arxiv_id(cached.get("arxiv_id") or ""), ""

    url = f"https://api.openalex.org/works/doi:{quote(normalize_doi(doi), safe='')}"
    item: dict[str, Any] = {"doi": normalize_doi(doi), "service": "openalex", "arxiv_id": ""}
    params = {"select": "id,doi,title,primary_location,best_oa_location,locations"}
    for attempt in range(max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            item["status_code"] = response.status_code
            if response.status_code == 200:
                data = response.json()
                item["openalex_id"] = normalize_text(data.get("id"))
                item["arxiv_id"] = arxiv_id_from_openalex_work(data)
                item.pop("error", None)
                break
            if response.status_code == 404:
                item["error"] = response.text[:500]
                break
            if response.status_code == 429:
                item["error"] = response.text[:500]
                if attempt < max_retries:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = float(retry_after) if retry_after else rate_limit_retry_sleep
                    except ValueError:
                        delay = rate_limit_retry_sleep
                    if verbose:
                        print(
                            f"OpenAlex 429 for DOI {normalize_doi(doi)}; "
                            f"retry {attempt + 1}/{max_retries} after {delay:g}s",
                            file=sys.stderr,
                            flush=True,
                        )
                    time.sleep(max(delay, 0.0))
                    continue
                break
            response.raise_for_status()
        except requests.RequestException as exc:
            item["error"] = str(exc)
            if attempt < max_retries:
                delay = max(retry_backoff * (2**attempt), 0.0)
                if verbose:
                    print(
                        f"OpenAlex request failed for DOI {normalize_doi(doi)}: {exc}; "
                        f"retry {attempt + 1}/{max_retries} after {delay:g}s",
                        file=sys.stderr,
                        flush=True,
                    )
                time.sleep(delay)
                continue
            break
    if item.get("status_code") in {200, 404}:
        cache.append(item)
    if sleep > 0:
        time.sleep(sleep)
    error = normalize_text(item.get("error"))
    if item.get("status_code") not in {200, 404} and not error:
        error = f"OpenAlex returned HTTP {item.get('status_code')}"
    return normalize_arxiv_id(item.get("arxiv_id") or ""), error


def arxiv_title_search(
    session: requests.Session,
    record: dict[str, Any],
    *,
    timeout: float,
    min_title_similarity: float,
    sleep: float,
    max_retries: int,
    retry_backoff: float,
    rate_limit_retry_sleep: float,
    verbose: bool,
) -> str:
    title = normalize_text(record.get("title"))
    if not title:
        return ""
    query_words = [word for word in re.findall(r"[A-Za-z0-9]+", title) if len(word) > 2][:10]
    if not query_words:
        return ""
    params = {"search_query": "all:" + " ".join(query_words), "start": 0, "max_results": 5}
    response = None
    for attempt in range(max_retries + 1):
        response = session.get("https://export.arxiv.org/api/query", params=params, timeout=timeout)
        if response.status_code != 429:
            response.raise_for_status()
            break
        if attempt >= max_retries:
            response.raise_for_status()
        retry_after = response.headers.get("Retry-After")
        try:
            delay = float(retry_after) if retry_after else rate_limit_retry_sleep
        except ValueError:
            delay = rate_limit_retry_sleep
        if verbose:
            print(
                f"arXiv title search 429 for title {title[:80]!r}; "
                f"retry {attempt + 1}/{max_retries} after {delay:g}s",
                file=sys.stderr,
                flush=True,
            )
        time.sleep(max(delay, 0.0))
        if retry_backoff > 0:
            rate_limit_retry_sleep *= retry_backoff
    if response is None:
        return ""
    root = ET.fromstring(response.text)
    target_title = normalize_title(title)
    target_year = normalize_text(record.get("year"))
    best_id = ""
    best_score = 0.0
    for entry in root.findall("atom:entry", ARXIV_NS):
        entry_title = normalize_text(entry.findtext("atom:title", default="", namespaces=ARXIV_NS))
        candidate_title = normalize_title(entry_title)
        score = difflib.SequenceMatcher(None, target_title, candidate_title).ratio()
        shorter = min(len(target_title), len(candidate_title))
        if shorter >= 24 and (target_title in candidate_title or candidate_title in target_title):
            score = max(score, 0.95)
        published = entry.findtext("atom:published", default="", namespaces=ARXIV_NS)
        if target_year and published[:4] and target_year != published[:4]:
            score -= 0.05
        if score > best_score:
            entry_id = entry.findtext("atom:id", default="", namespaces=ARXIV_NS)
            best_id = normalize_arxiv_id(entry_id.rsplit("/abs/", 1)[-1]) if "/abs/" in entry_id else ""
            best_score = score
    if sleep > 0:
        time.sleep(sleep)
    return best_id if best_score >= min_title_similarity else ""


def arxiv_doi_search(
    session: requests.Session,
    record: dict[str, Any],
    *,
    timeout: float,
    min_title_similarity: float,
    sleep: float,
    max_retries: int,
    retry_backoff: float,
    rate_limit_retry_sleep: float,
    verbose: bool,
) -> str:
    doi = normalize_doi(record.get("doi"))
    if not doi:
        return ""
    title = normalize_text(record.get("title"))
    target_title = normalize_title(title)
    params = {"search_query": f"doi:{doi}", "start": 0, "max_results": 5}
    response = None
    for attempt in range(max_retries + 1):
        response = session.get("https://export.arxiv.org/api/query", params=params, timeout=timeout)
        if response.status_code != 429:
            response.raise_for_status()
            break
        if attempt >= max_retries:
            response.raise_for_status()
        retry_after = response.headers.get("Retry-After")
        try:
            delay = float(retry_after) if retry_after else rate_limit_retry_sleep
        except ValueError:
            delay = rate_limit_retry_sleep
        if verbose:
            print(
                f"arXiv DOI search 429 for DOI {doi}; retry {attempt + 1}/{max_retries} after {delay:g}s",
                file=sys.stderr,
                flush=True,
            )
        time.sleep(max(delay, 0.0))
        if retry_backoff > 0:
            rate_limit_retry_sleep *= retry_backoff
    if response is None:
        return ""

    root = ET.fromstring(response.text)
    best_id = ""
    best_score = 0.0
    for entry in root.findall("atom:entry", ARXIV_NS):
        entry_id = entry.findtext("atom:id", default="", namespaces=ARXIV_NS)
        if "/abs/" not in entry_id:
            continue
        entry_title = normalize_text(entry.findtext("atom:title", default="", namespaces=ARXIV_NS))
        if target_title:
            candidate_title = normalize_title(entry_title)
            score = difflib.SequenceMatcher(None, target_title, candidate_title).ratio()
            shorter = min(len(target_title), len(candidate_title))
            if shorter >= 24 and (target_title in candidate_title or candidate_title in target_title):
                score = max(score, 0.95)
        else:
            score = 1.0
        if score > best_score:
            best_id = normalize_arxiv_id(entry_id.rsplit("/abs/", 1)[-1])
            best_score = score
    if sleep > 0:
        time.sleep(sleep)
    if not best_id:
        return ""
    return best_id if best_score >= min_title_similarity else ""


def find_arxiv_id(
    session: requests.Session,
    record: dict[str, Any],
    *,
    s2_cache: JsonlCache,
    openalex_cache: JsonlCache,
    use_openalex: bool,
    use_semantic_scholar: bool,
    use_arxiv_doi_search: bool,
    use_title_search: bool,
    timeout: float,
    min_title_similarity: float,
    sleep: float,
    arxiv_search_sleep: float,
    arxiv_max_retries: int,
    arxiv_retry_backoff: float,
    arxiv_rate_limit_retry_sleep: float,
    s2_max_retries: int,
    s2_retry_backoff: float,
    s2_rate_limit_retry_sleep: float,
    openalex_sleep: float,
    openalex_max_retries: int,
    openalex_retry_backoff: float,
    openalex_rate_limit_retry_sleep: float,
    verbose: bool,
) -> tuple[str, str, str]:
    doi = normalize_doi(record.get("doi"))
    arxiv_id = arxiv_id_from_doi(doi)
    if arxiv_id:
        return arxiv_id, "doi", ""

    arxiv_id = arxiv_id_from_text(record.get("abstract"), record.get("title"))
    if arxiv_id:
        return arxiv_id, "local_text", ""

    transient_errors: list[str] = []
    s2_errors: list[str] = []
    arxiv_errors: list[str] = []
    openalex_errors: list[str] = []
    if use_openalex:
        arxiv_id, error = openalex_arxiv_id(
            session,
            doi,
            cache=openalex_cache,
            timeout=timeout,
            sleep=openalex_sleep,
            max_retries=openalex_max_retries,
            retry_backoff=openalex_retry_backoff,
            rate_limit_retry_sleep=openalex_rate_limit_retry_sleep,
            verbose=verbose,
        )
        if arxiv_id:
            return arxiv_id, "openalex", ""
        if error:
            openalex_errors.append(error)

    if use_arxiv_doi_search:
        try:
            arxiv_id = arxiv_doi_search(
                session,
                record,
                timeout=timeout,
                min_title_similarity=min_title_similarity,
                sleep=arxiv_search_sleep,
                max_retries=arxiv_max_retries,
                retry_backoff=arxiv_retry_backoff,
                rate_limit_retry_sleep=arxiv_rate_limit_retry_sleep,
                verbose=verbose,
            )
        except requests.RequestException as exc:
            arxiv_id = ""
            arxiv_errors.append(f"arXiv DOI search failed: {exc}")
        if arxiv_id:
            return arxiv_id, "arxiv_doi_search", ""

    if use_semantic_scholar:
        arxiv_id, error = semantic_scholar_arxiv_id(
            session,
            doi,
            cache=s2_cache,
            timeout=timeout,
            sleep=sleep,
            max_retries=s2_max_retries,
            retry_backoff=s2_retry_backoff,
            rate_limit_retry_sleep=s2_rate_limit_retry_sleep,
            verbose=verbose,
        )
        if arxiv_id:
            return arxiv_id, "semantic_scholar", ""
        if error:
            s2_errors.append(error)

    if use_title_search:
        try:
            arxiv_id = arxiv_title_search(
                session,
                record,
                timeout=timeout,
                min_title_similarity=min_title_similarity,
                sleep=arxiv_search_sleep,
                max_retries=arxiv_max_retries,
                retry_backoff=arxiv_retry_backoff,
                rate_limit_retry_sleep=arxiv_rate_limit_retry_sleep,
                verbose=verbose,
            )
        except requests.RequestException as exc:
            arxiv_id = ""
            arxiv_errors.append(f"arXiv title search failed: {exc}")
        if arxiv_id:
            return arxiv_id, "arxiv_title_search", ""

    if openalex_errors and not s2_errors and not arxiv_errors:
        transient_errors.extend(f"failed_openalex:{error}" for error in openalex_errors)
    elif s2_errors and not openalex_errors and not arxiv_errors:
        transient_errors.extend(f"failed_s2:{error}" for error in s2_errors)
    elif arxiv_errors and not openalex_errors and not s2_errors:
        transient_errors.extend(f"failed_arxivsearch:{error}" for error in arxiv_errors)
    else:
        transient_errors.extend(f"failed_openalex:{error}" for error in openalex_errors)
        transient_errors.extend(f"failed_s2:{error}" for error in s2_errors)
        transient_errors.extend(f"failed_arxivsearch:{error}" for error in arxiv_errors)
    return "", "", "; ".join(transient_errors)


def fetch_arxiv_html(
    session: requests.Session,
    arxiv_id: str,
    *,
    timeout: float,
) -> tuple[str, str]:
    urls = [
        f"https://arxiv.org/html/{arxiv_id}",
        f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
    ]
    last_error = ""
    for url in urls:
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code != 200:
                last_error = f"{url} returned HTTP {response.status_code}"
                continue
            soup = BeautifulSoup(response.text, "lxml")
            if not soup.select_one("article.ltx_document, .ltx_document"):
                last_error = f"{url} did not contain LaTeXML article HTML"
                continue
            return response.text, response.url
        except requests.RequestException as exc:
            last_error = f"{url}: {exc}"
    raise RuntimeError(last_error or f"No HTML found for arXiv ID {arxiv_id}")


def inline_md(node: Tag | NavigableString, base_url: str) -> str:
    if isinstance(node, NavigableString):
        return str(node)
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name.lower()
        if name in {"script", "style", "noscript", "svg"}:
            continue
        text = inline_md(child, base_url)
        if not text:
            continue
        if name in {"em", "i"}:
            parts.append(f"*{text.strip()}*")
        elif name in {"strong", "b"}:
            parts.append(f"**{text.strip()}**")
        elif name == "a":
            href = child.get("href")
            if href and not href.startswith("#"):
                parts.append(f"[{text.strip()}]({urljoin(base_url, href)})")
            else:
                parts.append(text)
        elif name == "br":
            parts.append("\n")
        else:
            parts.append(text)
    return normalize_text("".join(parts))


def markdown_blocks(node: Tag | NavigableString, base_url: str) -> list[str]:
    if not isinstance(node, Tag):
        return []
    name = node.name.lower()
    classes = set(node.get("class") or [])
    if name in {"script", "style", "noscript", "svg"} or "ltx_bibliography" in classes:
        return []
    if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        text = inline_md(node, base_url)
        return [f"{'#' * min(int(name[1]), 6)} {text}"] if text else []
    if name == "p":
        text = inline_md(node, base_url)
        return [text] if text else []
    if name in {"ul", "ol"}:
        out = []
        for index, li in enumerate([c for c in node.children if isinstance(c, Tag) and c.name == "li"], 1):
            text = inline_md(li, base_url)
            if text:
                out.append(f"{index}. {text}" if name == "ol" else f"- {text}")
        return out
    if name == "figure":
        caption_node = node.find("figcaption")
        caption = inline_md(caption_node, base_url) if isinstance(caption_node, Tag) else ""
        out = []
        for image in node.find_all("img"):
            src = image.get("src")
            if src:
                alt = normalize_text(image.get("alt") or caption or "figure")
                out.append(f"![{alt}]({urljoin(base_url, src)})")
        if caption:
            out.append(f"> Figure: {caption}")
        return out
    if name == "table":
        caption_node = node.find(["caption", "figcaption"])
        rows = []
        for tr in node.find_all("tr")[:12]:
            cells = [normalize_text(cell.get_text(" ", strip=True)) for cell in tr.find_all(["th", "td"])]
            if cells:
                rows.append(" | ".join(cells))
        out = []
        if isinstance(caption_node, Tag):
            out.append(f"> Table: {inline_md(caption_node, base_url)}")
        if rows:
            out.append("```text\n" + "\n".join(rows) + "\n```")
        return out

    out: list[str] = []
    for child in node.children:
        out.extend(markdown_blocks(child, base_url))
    return out


def html_to_markdown(html: str, base_url: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    article = soup.select_one("article.ltx_document, .ltx_document")
    if article is None:
        raise ValueError("No LaTeXML article container found")
    for bad in article.select(".ltx_page_footer, .ltx_dates"):
        bad.decompose()
    blocks = [block for block in markdown_blocks(article, base_url) if block]
    return "\n\n".join(blocks).strip() + "\n"


def result_row(
    record: dict[str, Any],
    *,
    status: str,
    arxiv_id: str = "",
    arxiv_method: str = "",
    md_path: Path | None = None,
    error: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "paper_uid": normalize_text(record.get("paper_uid")),
        "doi": normalize_doi(record.get("doi")),
        "title": normalize_text(record.get("title")),
        "fullpaper_status": status,
    }
    if arxiv_id:
        row["arxiv_id"] = arxiv_id
    if arxiv_method:
        row["arxiv_method"] = arxiv_method
    if md_path is not None:
        try:
            row["fullpaper_md_path"] = str(md_path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            row["fullpaper_md_path"] = str(md_path)
    if error:
        row["error"] = error[:1000]
    return row


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Checkpoint must contain a JSON object: {path}")
    return data


def write_checkpoint(
    path: Path,
    *,
    matched: int,
    processed: int,
    paper_uid: str,
    status: str,
    results_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(
            {
                "matched": matched,
                "processed": processed,
                "paper_uid": paper_uid,
                "fullpaper_status": status,
                "results": str(results_path),
                "updated_at": int(time.time()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def current_fullpaper_status(record: dict[str, Any]) -> str:
    return normalize_text(record.get(STATUS_FIELD)) or PROCESS_STATUS


def record_matches(record: dict[str, Any], patterns: list[re.Pattern[str]]) -> bool:
    if not patterns:
        return True
    haystack = "\n".join(
        normalize_text(record.get(key)) for key in ("paper_uid", "doi", "title", "abstract", "dblp_key")
    )
    return any(pattern.search(haystack) for pattern in patterns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch arXiv HTML full papers and convert them to Markdown.")
    parser.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS_DIR)
    parser.add_argument("--input-file", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--openalex-cache", type=Path, default=None)
    parser.add_argument("--match-regex", action="append", default=[], help="Only process records matching regex.")
    parser.add_argument("--skip", type=int, default=0, help="Skip this many matching records before processing.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--min-md-bytes",
        type=int,
        default=1000,
        help="Treat converted Markdown smaller than this as not_found.",
    )
    parser.add_argument("--sleep", type=float, default=3.0, help="Sleep after Semantic Scholar/arXiv API calls.")
    parser.add_argument(
        "--arxiv-search-sleep",
        type=float,
        default=3.2,
        help="Sleep after each arXiv API title-search request. Keep >=3 to respect arXiv guidance.",
    )
    parser.add_argument("--arxiv-max-retries", type=int, default=3)
    parser.add_argument("--arxiv-retry-backoff", type=float, default=2.0)
    parser.add_argument("--arxiv-rate-limit-retry-sleep", type=float, default=60.0)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--api-key", default=None, help="Semantic Scholar API key.")
    parser.add_argument(
        "--use-env-api-key",
        action="store_true",
        help="Read Semantic Scholar API key from SEMANTIC_SCHOLAR_API_KEY or S2_API_KEY.",
    )
    parser.add_argument("--s2-max-retries", type=int, default=3)
    parser.add_argument("--s2-retry-backoff", type=float, default=2.0)
    parser.add_argument("--s2-rate-limit-retry-sleep", type=float, default=60.0)
    parser.add_argument("--openalex", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--openalex-sleep", type=float, default=0.12)
    parser.add_argument("--openalex-max-retries", type=int, default=3)
    parser.add_argument("--openalex-retry-backoff", type=float, default=2.0)
    parser.add_argument("--openalex-rate-limit-retry-sleep", type=float, default=10.0)
    parser.add_argument("--semantic-scholar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--arxiv-doi-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--arxiv-title-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--retry-status",
        action="append",
        choices=sorted(RETRYABLE_FAILED_STATUSES),
        default=[],
        help="Retry records whose current fullpaper_status is this value. Repeatable.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry all failed/failed_s2/failed_arxivsearch records.",
    )
    parser.add_argument(
        "--keep-unresolved-not-searched",
        action="store_true",
        help="If no arXiv ID is found, keep fullpaper_status as not_searched instead of writing not_found.",
    )
    parser.add_argument("--min-title-similarity", type=float, default=0.92)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the checkpoint. This never truncates the existing results file.",
    )
    parser.add_argument("--quiet", action="store_true", help="Do not print one JSON result per processed record.")
    parser.add_argument("--user-agent", default="Vitality2 fullpaper fetcher (mailto:unknown@example.com)")
    parser.add_argument("--verbose", action="store_true", help="Print per-record progress and retry waits.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.skip < 0:
        raise SystemExit("--skip must be >= 0")
    if args.min_md_bytes < 1:
        raise SystemExit("--min-md-bytes must be >= 1")
    if args.checkpoint_every < 1:
        raise SystemExit("--checkpoint-every must be >= 1")
    if args.flush_every < 1:
        raise SystemExit("--flush-every must be >= 1")
    if not 0 <= args.min_title_similarity <= 1:
        raise SystemExit("--min-title-similarity must be between 0 and 1")

    input_files = args.input_file or list(iter_default_input_files(args.papers_dir))
    if not input_files:
        raise SystemExit("No input JSON files found.")

    results_path = args.results or args.output_dir / "fullpaper_results.jsonl"
    cache_path = args.cache or args.output_dir / "semantic_scholar_arxiv_cache.jsonl"
    openalex_cache_path = args.openalex_cache or args.output_dir / "openalex_arxiv_cache.jsonl"
    checkpoint_path = args.checkpoint or args.output_dir / "fullpaper_checkpoint.json"
    md_dir = args.output_dir / "md"
    md_dir.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.overwrite:
        raise SystemExit("--resume cannot be combined with --overwrite")
    checkpoint: dict[str, Any] = {}
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        checkpoint_matched = int(checkpoint.get("matched") or 0)
        if checkpoint_matched > args.skip:
            args.skip = checkpoint_matched
        if args.verbose:
            print(
                f"Resuming from checkpoint matched={checkpoint_matched}; effective skip={args.skip}",
                file=sys.stderr,
                flush=True,
            )
    if args.overwrite and results_path.exists():
        results_path.unlink()
    if args.overwrite and checkpoint_path.exists():
        checkpoint_path.unlink()

    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in args.match_regex]
    s2_cache = JsonlCache(cache_path)
    openalex_cache = JsonlCache(openalex_cache_path)
    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent})
    s2_api_key = args.api_key or ""
    if args.use_env_api_key and not s2_api_key:
        s2_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY") or ""
        if not s2_api_key:
            s2_api_key = load_dotenv_key(args.env_file, S2_API_KEY_ENV_NAMES)
    if s2_api_key:
        session.headers.update({"x-api-key": s2_api_key})

    processed = 0
    matched = 0
    counts: dict[str, int] = {}
    with results_path.open("a", encoding="utf-8") as results:
        def emit(row: dict[str, Any]) -> None:
            counts[row["fullpaper_status"]] = counts.get(row["fullpaper_status"], 0) + 1
            results.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            results.write("\n")
            if processed % args.flush_every == 0:
                results.flush()
            if not args.quiet:
                print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
            if processed % args.checkpoint_every == 0:
                write_checkpoint(
                    checkpoint_path,
                    matched=matched,
                    processed=processed,
                    paper_uid=normalize_text(row.get("paper_uid")),
                    status=normalize_text(row.get("fullpaper_status")),
                    results_path=results_path,
                )

        for path in input_files:
            for record in load_json_array(path):
                if not record_matches(record, patterns):
                    continue
                matched += 1
                if matched <= args.skip:
                    continue
                if args.limit is not None and processed >= args.limit:
                    break

                processed += 1
                if args.verbose:
                    print(
                        f"Processing {processed}: {normalize_text(record.get('paper_uid')) or normalize_doi(record.get('doi'))}",
                        file=sys.stderr,
                        flush=True,
                    )
                current_status = current_fullpaper_status(record)
                retry_statuses = RETRYABLE_FAILED_STATUSES if args.retry_failed else set(args.retry_status)
                should_retry_failed = current_status in retry_statuses
                if current_status in SKIP_STATUSES and not should_retry_failed:
                    row = result_row(record, status=current_status)
                    row["skipped"] = True
                    emit(row)
                    continue
                if current_status != PROCESS_STATUS and not should_retry_failed:
                    row = result_row(record, status="failed", error=f"Unknown {STATUS_FIELD}: {current_status}")
                    emit(row)
                    continue

                doi = normalize_doi(record.get("doi"))
                if not doi:
                    row = result_row(record, status="not_applicable")
                else:
                    try:
                        arxiv_id, method, lookup_error = find_arxiv_id(
                            session,
                            record,
                            s2_cache=s2_cache,
                            openalex_cache=openalex_cache,
                            use_openalex=args.openalex,
                            use_semantic_scholar=args.semantic_scholar,
                            use_arxiv_doi_search=args.arxiv_doi_search,
                            use_title_search=args.arxiv_title_search,
                            timeout=args.timeout,
                            min_title_similarity=args.min_title_similarity,
                            sleep=args.sleep,
                            arxiv_search_sleep=args.arxiv_search_sleep,
                            arxiv_max_retries=args.arxiv_max_retries,
                            arxiv_retry_backoff=args.arxiv_retry_backoff,
                            arxiv_rate_limit_retry_sleep=args.arxiv_rate_limit_retry_sleep,
                            s2_max_retries=args.s2_max_retries,
                            s2_retry_backoff=args.s2_retry_backoff,
                            s2_rate_limit_retry_sleep=args.s2_rate_limit_retry_sleep,
                            openalex_sleep=args.openalex_sleep,
                            openalex_max_retries=args.openalex_max_retries,
                            openalex_retry_backoff=args.openalex_retry_backoff,
                            openalex_rate_limit_retry_sleep=args.openalex_rate_limit_retry_sleep,
                            verbose=args.verbose,
                        )
                        if not arxiv_id:
                            if lookup_error:
                                if (
                                    "failed_openalex:" in lookup_error
                                    and "failed_s2:" not in lookup_error
                                    and "failed_arxivsearch:" not in lookup_error
                                ):
                                    status = "failed_openalex"
                                elif (
                                    "failed_s2:" in lookup_error
                                    and "failed_openalex:" not in lookup_error
                                    and "failed_arxivsearch:" not in lookup_error
                                ):
                                    status = "failed_s2"
                                elif (
                                    "failed_arxivsearch:" in lookup_error
                                    and "failed_openalex:" not in lookup_error
                                    and "failed_s2:" not in lookup_error
                                ):
                                    status = "failed_arxivsearch"
                                else:
                                    status = "failed"
                                clean_error = (
                                    lookup_error.replace("failed_openalex:", "")
                                    .replace("failed_s2:", "")
                                    .replace("failed_arxivsearch:", "")
                                )
                                row = result_row(record, status=status, error=clean_error)
                            elif args.keep_unresolved_not_searched:
                                row = result_row(record, status=PROCESS_STATUS)
                            else:
                                row = result_row(record, status="not_found")
                        else:
                            md_path = md_dir / f"{safe_filename(record.get('paper_uid') or doi)}.md"
                            if md_path.exists() and not args.overwrite:
                                row = result_row(record, status="found", arxiv_id=arxiv_id, arxiv_method=method, md_path=md_path)
                            else:
                                html, html_url = fetch_arxiv_html(session, arxiv_id, timeout=args.timeout)
                                markdown = html_to_markdown(html, html_url)
                                md_bytes = len(markdown.encode("utf-8"))
                                if md_bytes < args.min_md_bytes:
                                    row = result_row(
                                        record,
                                        status="not_found",
                                        arxiv_id=arxiv_id,
                                        arxiv_method=method,
                                        error=f"Converted Markdown too small: {md_bytes} bytes",
                                    )
                                    if md_path.exists():
                                        md_path.unlink()
                                    emit(row)
                                    continue
                                md_path.write_text(markdown, encoding="utf-8")
                                row = result_row(record, status="found", arxiv_id=arxiv_id, arxiv_method=method, md_path=md_path)
                                row["fullpaper_md_bytes"] = md_bytes
                    except Exception as exc:
                        if "No HTML found for arXiv ID" in str(exc) or "did not contain LaTeXML article HTML" in str(exc):
                            row = result_row(record, status="not_found", error=str(exc))
                        else:
                            row = result_row(record, status="failed", error=str(exc))

                emit(row)

            if args.limit is not None and processed >= args.limit:
                break

    if processed:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        write_checkpoint(
            checkpoint_path,
            matched=matched,
            processed=processed,
            paper_uid="",
            status="",
            results_path=results_path,
        )
    print(json.dumps({"processed": processed, "counts": counts, "results": str(results_path)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
