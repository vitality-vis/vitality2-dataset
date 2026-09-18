#!/usr/bin/env python3
"""Download OpenAlex PDF URL candidates and convert PDFs to Markdown."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "script"))

from fetch_arxiv_fullpaper_md import normalize_doi, normalize_text, safe_filename  # noqa: E402


SUCCESS_STATUS_CODES = {200}
RESUME_STATUSES = {"found", "not_found", "failed_download", "failed_pdf_parse"}


def iter_jsonl(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def load_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    for row in iter_jsonl(path):
        paper_uid = normalize_text(row.get("paper_uid"))
        status = normalize_text(row.get("fullpaper_status"))
        if paper_uid and status in RESUME_STATUSES:
            completed.add(paper_uid)
    return completed


def load_found_paper_uids(paths: list[Path]) -> set[str]:
    found: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Skip results file does not exist: {path}")
        for row in iter_jsonl(path):
            paper_uid = normalize_text(row.get("paper_uid"))
            status = normalize_text(row.get("fullpaper_status"))
            if paper_uid and status == "found":
                found.add(paper_uid)
    return found


def first_pdf_urls(row: dict[str, Any], max_urls: int) -> list[str]:
    urls: list[str] = []
    best = normalize_text(row.get("best_pdf_url"))
    if best:
        urls.append(best)
    for item in row.get("pdf_urls") or []:
        if not isinstance(item, dict):
            continue
        url = normalize_text(item.get("pdf_url"))
        if url and url not in urls:
            urls.append(url)
        if len(urls) >= max_urls:
            break
    return urls[:max_urls]


def looks_like_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


def extension_from_url(url: str) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix == ".pdf":
        return ".pdf"
    return ".pdf"


def download_pdf(
    session: requests.Session,
    url: str,
    path: Path,
    *,
    timeout: float,
    max_retries: int,
    retry_backoff: float,
) -> tuple[bool, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(max_retries + 1):
        try:
            with session.get(url, stream=True, timeout=timeout, allow_redirects=True) as response:
                if response.status_code not in SUCCESS_STATUS_CODES:
                    error = f"HTTP {response.status_code}"
                    if attempt < max_retries and response.status_code in {429, 500, 502, 503, 504}:
                        time.sleep(retry_backoff * (2**attempt))
                        continue
                    return False, error
                content_type = response.headers.get("Content-Type", "")
                with tmp_path.open("wb") as out:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            out.write(chunk)
                if not looks_like_pdf(tmp_path):
                    tmp_path.unlink(missing_ok=True)
                    return False, f"not a PDF: {content_type or 'unknown content type'}"
                tmp_path.replace(path)
                return True, ""
        except requests.RequestException as exc:
            if attempt < max_retries:
                time.sleep(retry_backoff * (2**attempt))
                continue
            return False, str(exc)
    return False, "download failed"


def clean_extracted_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pdf_to_markdown(pdf_path: Path, *, title: str, doi: str, source_url: str) -> tuple[str, int]:
    reader = PdfReader(str(pdf_path))
    blocks = []
    clean_title = normalize_text(title)
    if clean_title:
        blocks.append(f"# {clean_title}")
    metadata = []
    if doi:
        metadata.append(f"DOI: {doi}")
    if source_url:
        metadata.append(f"PDF: {source_url}")
    if metadata:
        blocks.append("\n".join(metadata))
    extracted_pages = 0
    for index, page in enumerate(reader.pages, start=1):
        text = clean_extracted_text(page.extract_text() or "")
        if not text:
            continue
        extracted_pages += 1
        blocks.append(f"## Page {index}\n\n{text}")
    markdown = "\n\n".join(blocks).strip() + "\n"
    return markdown, extracted_pages


def result_row(
    source: dict[str, Any],
    *,
    status: str,
    pdf_url: str = "",
    pdf_path: Path | None = None,
    md_path: Path | None = None,
    error: str = "",
    md_bytes: int = 0,
    pdf_bytes: int = 0,
    page_count: int = 0,
    extracted_pages: int = 0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "paper_uid": normalize_text(source.get("paper_uid")),
        "doi": normalize_doi(source.get("doi")),
        "title": normalize_text(source.get("title")),
        "fullpaper_status": status,
        "fullpaper_source": "openalex_pdf",
    }
    if pdf_url:
        row["fullpaper_pdf_url"] = pdf_url
    if pdf_path is not None:
        row["fullpaper_pdf_path"] = relative_path(pdf_path)
    if md_path is not None:
        row["fullpaper_md_path"] = relative_path(md_path)
    if md_bytes:
        row["fullpaper_md_bytes"] = md_bytes
    if pdf_bytes:
        row["fullpaper_pdf_bytes"] = pdf_bytes
    if page_count:
        row["fullpaper_pdf_pages"] = page_count
    if extracted_pages:
        row["fullpaper_pdf_extracted_pages"] = extracted_pages
    if error:
        row["error"] = error[:1000]
    return row


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def make_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1",
        }
    )
    return session


def process_source(
    source: dict[str, Any],
    *,
    pdf_dir: Path,
    md_dir: Path,
    max_urls_per_paper: int,
    timeout: float,
    max_retries: int,
    retry_backoff: float,
    min_md_bytes: int,
    overwrite: bool,
    keep_pdf: bool,
    user_agent: str,
) -> dict[str, Any]:
    paper_uid = normalize_text(source.get("paper_uid"))
    title = normalize_text(source.get("title"))
    doi = normalize_doi(source.get("doi"))
    base_name = safe_filename(paper_uid or doi or hashlib.sha1(title.encode()).hexdigest())
    md_path = md_dir / f"{base_name}.md"
    urls = first_pdf_urls(source, max_urls_per_paper)
    last_error = ""
    session = make_session(user_agent)

    for url_index, pdf_url in enumerate(urls, start=1):
        suffix = extension_from_url(pdf_url)
        pdf_path = pdf_dir / f"{base_name}{'' if url_index == 1 else f'_{url_index}'}{suffix}"
        if not pdf_path.exists() or overwrite:
            ok, error = download_pdf(
                session,
                pdf_url,
                pdf_path,
                timeout=timeout,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
            )
            if not ok:
                last_error = error
                continue
        try:
            markdown, extracted_pages = pdf_to_markdown(pdf_path, title=title, doi=doi, source_url=pdf_url)
            page_count = len(PdfReader(str(pdf_path)).pages)
            if len(markdown.encode("utf-8")) < min_md_bytes or extracted_pages == 0:
                last_error = "extracted text too short"
                continue
            md_path.write_text(markdown, encoding="utf-8")
            pdf_bytes = pdf_path.stat().st_size
            returned_pdf_path = pdf_path if keep_pdf else None
            if not keep_pdf:
                pdf_path.unlink(missing_ok=True)
            return result_row(
                source,
                status="found",
                pdf_url=pdf_url,
                pdf_path=returned_pdf_path,
                md_path=md_path,
                md_bytes=len(markdown.encode("utf-8")),
                pdf_bytes=pdf_bytes,
                page_count=page_count,
                extracted_pages=extracted_pages,
            )
        except Exception as exc:  # pypdf raises several parser-specific exceptions.
            last_error = str(exc)
            continue

    status = "failed_download" if last_error.startswith("HTTP") or "PDF" in last_error else "failed_pdf_parse"
    if not urls:
        status = "not_found"
    return result_row(source, status=status, error=last_error)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OpenAlex PDF URLs and convert PDFs to Markdown.")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument(
        "--skip-existing-results",
        type=Path,
        action="append",
        default=[],
        help="Skip paper_uid values that are already fullpaper_status=found in this JSONL. Repeatable.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max found-PDF records to process.")
    parser.add_argument("--max-urls-per-paper", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=3.0)
    parser.add_argument("--min-md-bytes", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-pending", type=int, default=0, help="Max queued futures; default is workers * 4.")
    parser.add_argument("--keep-pdf", action="store_true", help="Keep downloaded PDFs after Markdown conversion.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--user-agent", default="Vitality2 PDF fullpaper fetcher (research; mailto:unknown@example.com)")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.max_urls_per_paper < 1:
        raise SystemExit("--max-urls-per-paper must be >= 1")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    output_dir = args.output_dir
    pdf_dir = output_dir / "pdf"
    md_dir = output_dir / "md"
    results_path = args.results or output_dir / "fullpaper_pdf_results.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite and results_path.exists():
        results_path.unlink()
    completed = load_completed(results_path)
    skipped_existing = load_found_paper_uids(args.skip_existing_results)

    max_pending = args.max_pending if args.max_pending > 0 else args.workers * 4
    counts = {
        "cached": 0,
        "skipped_existing": 0,
        "found": 0,
        "not_found": 0,
        "failed_download": 0,
        "failed_pdf_parse": 0,
    }
    processed = 0
    scanned = 0

    def submit(executor: ThreadPoolExecutor, source: dict[str, Any]) -> Future[dict[str, Any]]:
        return executor.submit(
            process_source,
            source,
            pdf_dir=pdf_dir,
            md_dir=md_dir,
            max_urls_per_paper=args.max_urls_per_paper,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
            min_md_bytes=args.min_md_bytes,
            overwrite=args.overwrite,
            keep_pdf=args.keep_pdf,
            user_agent=args.user_agent,
        )

    def write_finished(finished: set[Future[dict[str, Any]]], out: Any) -> None:
        for future in finished:
            row = future.result()
            status = normalize_text(row.get("fullpaper_status"))
            counts[status] = counts.get(status, 0) + 1
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            out.flush()
            if args.verbose:
                print(json.dumps({"processed": processed, "counts": counts, "row": row}, ensure_ascii=False), flush=True)

    with results_path.open("a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            pending: set[Future[dict[str, Any]]] = set()
            for source in iter_jsonl(args.input_file):
                if normalize_text(source.get("pdf_status")) != "found":
                    continue
                paper_uid = normalize_text(source.get("paper_uid"))
                if not paper_uid:
                    continue
                if args.limit is not None and processed >= args.limit:
                    break
                scanned += 1
                if paper_uid in skipped_existing:
                    counts["skipped_existing"] += 1
                    continue
                if paper_uid in completed:
                    counts["cached"] += 1
                    continue

                processed += 1
                if args.verbose:
                    print(f"Submit {processed}: {paper_uid}", flush=True)
                pending.add(submit(executor, source))
                if args.sleep > 0:
                    time.sleep(args.sleep)
                if len(pending) >= max_pending:
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    write_finished(finished, out)

            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                write_finished(finished, out)

    print(
        json.dumps(
            {
                "processed": processed,
                "scanned_found_pdf_records": scanned,
                "counts": counts,
                "results": str(results_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
