#!/usr/bin/env python3
"""Interactive, in-process incremental update for Vitality2 papers.

This is the only orchestrator for an incremental update.  It calls the DBLP
and enrichment implementations as Python APIs, not child processes, so the
run owns prompts, state, progress, and the final production promotion.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import download_dblp_dump
import enrich_crossref_missing
import enrich_openalex_by_doi
import enrich_openalex_missing_doi_by_search
import enrich_semantic_scholar_missing
import export_zilliz_paper_update_candidates as zilliz_export
import filter_new_dblp_papers as paper_filter
import split_dblp_by_source
import upload_papers_to_zilliz as paper_upload
import upsert_enriched_papers_to_zilliz as paper_upsert


warnings.filterwarnings(
    "ignore",
    message=r"`.*` is an ORM-style PyMilvus API and will be removed.*",
    category=FutureWarning,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PROD = PROJECT_ROOT / "script_prod"
if str(SCRIPT_PROD) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PROD))
import sync as production_sync  # noqa: E402
import umap_projection  # noqa: E402
from common import EMBEDDING_DIM  # noqa: E402


PAPER_NEW = "paper_new"
PAPER_EXCLUDE = "paper_exclude"
PAPER_PROD = "paper_prod"
METADATA_FIELDS = ["paper_uid", "dblp_key", "doi", "year", "has_doi", "has_abstract"]


class AbortRun(Exception):
    pass


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_records(directory: Path, *, include_missing_doi: bool = True) -> Iterable[tuple[Path, dict[str, Any]]]:
    for folder_name in ("enriched", "missing"):
        folder = directory / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.json")):
            if not include_missing_doi and path.name == "_missing_doi.json":
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                continue
            for record in data:
                if isinstance(record, dict):
                    yield path, record


def count_records(directory: Path) -> dict[str, int]:
    counts = {"total": 0, "with_doi": 0, "with_abstract": 0, "missing_abstract": 0}
    for _, record in read_records(directory):
        counts["total"] += 1
        if zilliz_export.normalize_doi(record.get("doi")):
            counts["with_doi"] += 1
        if zilliz_export.normalize_text(record.get("abstract")):
            counts["with_abstract"] += 1
        else:
            counts["missing_abstract"] += 1
    return counts


def count_doi_missing_abstract(directory: Path) -> int:
    return sum(
        1
        for _, record in read_records(directory, include_missing_doi=False)
        if zilliz_export.normalize_doi(record.get("doi"))
        and not zilliz_export.normalize_text(record.get("abstract"))
    )


def has_split_files(directory: Path) -> bool:
    return any(path.is_file() and path.stat().st_size for path in directory.glob("*.json"))


def read_cached_values(path: Path, normalizer: Callable[[Any], str]) -> set[str]:
    """Read a required newline-delimited cache without silently accepting an empty baseline."""
    if not path.exists():
        raise RuntimeError(f"Required cached file is missing: {path.relative_to(PROJECT_ROOT)}")
    values = {
        value
        for line in path.read_text(encoding="utf-8").splitlines()
        if (value := normalizer(line))
    }
    if not values:
        raise RuntimeError(f"Required cached file is empty: {path.relative_to(PROJECT_ROOT)}")
    return values


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


@dataclass
class UpdateCLI:
    args: argparse.Namespace
    update_date: str
    update_dir: Path
    existing_update_dir: Path
    report: dict[str, str] = field(default_factory=dict)
    changed_uids: set[str] = field(default_factory=set)
    new_uploaded: int = 0
    existing_upserted: int = 0
    production_umap_uids: set[str] = field(default_factory=set)
    production_umap_updated: int = 0
    active_stage: str | None = None
    live_progress_label: str | None = None

    def prompt_yes_skip(self, prompt: str, *, write: bool = False) -> bool:
        if write and self.args.no_write:
            print(f"{prompt} [skipped by --no-write]")
            return False
        if self.args.yes:
            print(f"{prompt} [yes]")
            return True
        while True:
            answer = input(f"{prompt} [yes/skip] ").strip().casefold()
            if answer in {"y", "yes"}:
                return True
            if answer in {"s", "skip", "n", "no"}:
                return False
            print("Please answer yes or skip.")

    def prompt_retry_abort(self, label: str) -> bool:
        while True:
            answer = input(f"{label} failed. [retry/abort] ").strip().casefold()
            if answer in {"r", "retry"}:
                return True
            if answer in {"a", "abort"}:
                raise AbortRun
            print("Please answer retry or abort.")

    def stage(self, number: int, title: str) -> None:
        print(f"\n[{number}/9] {title}")

    def clear_live_progress(self) -> None:
        if self.live_progress_label and sys.stdout.isatty():
            sys.stdout.write("\r\033[2K")
            sys.stdout.flush()
        self.live_progress_label = None

    def live_progress(self, label: str, text: str) -> None:
        line = f"  {label}: {text}"
        if sys.stdout.isatty():
            sys.stdout.write(f"\r\033[2K{line[:180]}")
            sys.stdout.flush()
            self.live_progress_label = label
        else:
            print(line, flush=True)

    def download_progress(self, message: str) -> None:
        """Render byte progress as one terminal line; keep state messages readable."""
        match = re.match(r"^downloaded (.+) / (.+) \(([\d.]+)%\)$", message)
        if match:
            downloaded, total, percent_text = match.groups()
            percent = max(0.0, min(100.0, float(percent_text)))
            width = 24
            filled = round(width * percent / 100)
            bar = "#" * filled + "-" * (width - filled)
            self.live_progress("DBLP download", f"[{bar}] {percent:5.1f}% {downloaded} / {total}")
            return
        decompressed = re.match(r"^decompressed (.+)$", message)
        if decompressed:
            self.live_progress("DBLP download", f"decompress={decompressed.group(1)}")
            return
        self.clear_live_progress()
        print(f"  DBLP download: {message}", flush=True)

    def run_stage(
        self,
        label: str,
        action: Callable[[], Any],
        *,
        nonfatal: bool = False,
    ) -> bool:
        self.active_stage = label
        if label != "DBLP download":
            self.clear_live_progress()
        print(f"  {label}: started", flush=True)
        started = time.monotonic()
        try:
            result = action()
        except KeyboardInterrupt:
            if label == "DBLP download":
                removed = download_dblp_dump.cleanup_partial_downloads(PROJECT_ROOT / "data/dblp/dump")
                print(f"  DBLP download: interrupted; removed {removed} partial download file(s).", flush=True)
            self.report["interrupted stage"] = label
            raise AbortRun
        except Exception as exc:  # noqa: BLE001
            print(f"  {label}: failed ({time.monotonic() - started:.1f}s): {exc}", flush=True)
            if nonfatal:
                return False
            return False
        finally:
            self.clear_live_progress()
            self.active_stage = None
        if isinstance(result, int) and result != 0:
            if result == 130:
                self.report["interrupted stage"] = label
            print(f"  {label}: failed ({time.monotonic() - started:.1f}s), exit={result}", flush=True)
            return False
        print(f"  {label}: done ({time.monotonic() - started:.1f}s)", flush=True)
        return True

    def run_required_openalex(self, label: str, action: Callable[[], Any], manifest: Path) -> None:
        while True:
            if self.run_stage(label, action):
                stopped_reason = load_json(manifest).get("stopped_reason")
                if not stopped_reason:
                    return
                print(f"  {label}: incomplete: {stopped_reason}")
            if not self.prompt_retry_abort(label):
                raise AbortRun

    @staticmethod
    def service_args(module: Any, values: list[str]) -> argparse.Namespace:
        args = module.parse_args(values)
        args.quiet = True
        return args

    def prepare_dblp(self) -> None:
        self.stage(1, "DBLP preparation")
        split_dir = PROJECT_ROOT / "data/dblp/split_source"
        if self.prompt_yes_skip("Download fresh DBLP dump?"):
            download_args = download_dblp_dump.parse_args([])
            download_args.progress_callback = self.download_progress
            if not self.run_stage("DBLP download", lambda: download_dblp_dump.run(download_args)):
                raise AbortRun
            should_split = True
        else:
            should_split = self.prompt_yes_skip("Rebuild selected-source split files?")
        if should_split:
            split_args = split_dblp_by_source.parse_args(
                ["--overwrite", "--max-open-files", "128", "--progress-every", "2000"]
            )
            if not self.run_stage("DBLP split", lambda: split_dblp_by_source.run(split_args)):
                raise AbortRun
        elif not has_split_files(split_dir):
            print("No split DBLP files found. Download or split is required.")
            raise AbortRun

        manifest = load_json(split_dir / "_source_manifest.json")
        papers = sum(int(row.get("count", 0) or 0) for row in manifest.values() if isinstance(row, dict))
        print(f"Split sources: {len(manifest)} | Split papers: {papers}")

    def export_excluded_keys(self) -> set[str]:
        collection = zilliz_export.connect_collection(PAPER_EXCLUDE)
        collection.load(load_fields=["key", "dblp_key", "_exclude_vector"])
        keys: set[str] = set()
        iterator = collection.query_iterator(batch_size=5000, expr="", output_fields=["dblp_key"], timeout=300.0)
        scanned = 0
        try:
            while batch := iterator.next():
                scanned += len(batch)
                keys.update(
                    key for row in batch if (key := zilliz_export.normalize_text(row.get("dblp_key")))
                )
                self.live_progress("paper_exclude export", f"scanned={scanned} keys={len(keys)}")
        finally:
            iterator.close()
        if not keys:
            raise RuntimeError("paper_exclude contains no dblp_key records; refusing to continue without exclusions.")
        return keys

    def export_existing(self) -> tuple[set[str], set[str], set[str]]:
        self.stage(2, "Exporting paper_new metadata")
        keys_path = PROJECT_ROOT / "data/zilliz/paper_new_dblp_keys.txt"
        dois_path = PROJECT_ROOT / "data/zilliz/paper_new_dois.txt"
        excluded_path = PROJECT_ROOT / "data/zilliz/paper_exclude_dblp_keys.txt"
        manifest_path = self.existing_update_dir / "existing_missing_abstract_manifest.json"

        def action() -> None:
            zilliz_export.prepare_candidates_dir(self.existing_update_dir / "split_source", overwrite=True)
            collection = zilliz_export.connect_collection(PAPER_NEW)
            collection.load(load_fields=[*zilliz_export.STATIC_LOAD_FIELDS, zilliz_export.VECTOR_LOAD_FIELD])
            keys: set[str] = set()
            dois: set[str] = set()
            candidates = 0
            metadata_path = PROJECT_ROOT / "data/zilliz/paper_new_update_metadata.jsonl"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            writer = zilliz_export.JsonArrayWriter(self.existing_update_dir / "split_source/Unknown.json")
            scanned = 0
            with metadata_path.open("w", encoding="utf-8") as metadata:
                for scanned, row in enumerate(zilliz_export.iter_rows(collection, 5000, 300.0), start=1):
                    clean = zilliz_export.clean_row(row)
                    if key := zilliz_export.normalize_text(clean.get("dblp_key")):
                        keys.add(key)
                    if doi := zilliz_export.normalize_doi(clean.get("doi")):
                        dois.add(doi)
                    metadata.write(json.dumps(clean, ensure_ascii=False, separators=(",", ":"), default=zilliz_export.json_default) + "\n")
                    if (
                        clean["has_doi"] and not clean["has_abstract"]
                        and clean.get("year") == int(self.update_date[:4])
                    ):
                        writer.write(zilliz_export.minimal_candidate(clean))
                        candidates += 1
                    if scanned % 2000 == 0:
                        self.live_progress(
                            "paper_new export",
                            f"scanned={scanned} keys={len(keys)} dois={len(dois)} old_candidates={candidates}",
                        )
            writer.close()
            keys_path.write_text("\n".join(sorted(keys)) + "\n", encoding="utf-8")
            dois_path.write_text("\n".join(sorted(dois)) + "\n", encoding="utf-8")
            write_json(manifest_path, {
                "collection": PAPER_NEW, "scanned_rows": scanned, "dblp_keys": len(keys), "dois": len(dois),
                "existing_doi_missing_abstract": candidates, "candidate_year": int(self.update_date[:4]),
            })
            self._existing_keys, self._existing_dois = keys, dois

        self._existing_keys: set[str] = set()
        self._existing_dois: set[str] = set()
        excluded: set[str] = set()
        if self.prompt_yes_skip("Export paper_new and paper_exclude metadata?"):
            if not self.run_stage("paper_new export", action):
                raise AbortRun
            if not self.run_stage("paper_exclude export", lambda: excluded.update(self.export_excluded_keys())):
                raise AbortRun
            excluded_path.parent.mkdir(parents=True, exist_ok=True)
            excluded_path.write_text("\n".join(sorted(excluded)) + "\n", encoding="utf-8")
        else:
            try:
                self._existing_keys = read_cached_values(keys_path, zilliz_export.normalize_text)
                self._existing_dois = read_cached_values(dois_path, zilliz_export.normalize_doi)
            except RuntimeError as exc:
                print(f"Cannot skip metadata export: {exc}")
                raise AbortRun from exc
            if excluded_path.exists():
                try:
                    excluded = read_cached_values(excluded_path, zilliz_export.normalize_text)
                except RuntimeError as exc:
                    print(f"Cannot reuse paper_exclude cache: {exc}")
                    raise AbortRun from exc
            else:
                print("paper_exclude cache is unavailable; it is required only when rebuilding the new-paper filter.")
            if not manifest_path.exists():
                print(f"Cannot skip metadata export: required cached file is missing: {manifest_path.relative_to(PROJECT_ROOT)}")
                raise AbortRun
            cached_manifest = load_json(manifest_path)
            cached_year = cached_manifest.get("candidate_year")
            if cached_year != int(self.update_date[:4]):
                print("Cannot skip metadata export: cached old-paper candidates are for a different year.")
                raise AbortRun
            cached_candidates = int(cached_manifest.get("existing_doi_missing_abstract", 0) or 0)
            if cached_candidates and not has_split_files(self.existing_update_dir / "split_source"):
                print("Cannot skip metadata export: cached old-paper candidate output is incomplete.")
                raise AbortRun
            print("Using cached paper_new metadata.")

        manifest = load_json(manifest_path)
        print(f"Rows: {manifest.get('scanned_rows', 0)} | DBLP keys: {manifest.get('dblp_keys', 0)} | DOIs: {manifest.get('dois', 0)}")
        excluded_summary = str(len(excluded)) if excluded else "cache unavailable"
        print(f"Excluded DBLP keys: {excluded_summary} | Existing DOI/no-abstract candidates for {self.update_date[:4]}: {manifest.get('existing_doi_missing_abstract', 0)}")
        return self._existing_keys, self._existing_dois, excluded

    def filter_new(self, existing_keys: set[str], existing_dois: set[str], excluded_keys: set[str]) -> int:
        self.stage(3, "New-paper filter")
        manifest_path = self.update_dir / "filter_manifest.json"
        def action() -> None:
            output = paper_filter.prepare_output_dir(self.update_dir, overwrite=True)
            files = paper_filter.iter_split_files(PROJECT_ROOT / "data/dblp/split_source")
            stats = {"scanned_papers": 0, "excluded_papers": 0, "existing_papers": 0, "existing_doi_papers": 0, "new_papers": 0, "output_files": 0}
            by_source: dict[str, int] = {}
            for path in files:
                retained: list[dict[str, Any]] = []
                for paper in paper_filter.load_json_array(path):
                    stats["scanned_papers"] += 1
                    key = paper_filter.normalize_value(paper.get("dblp_key"))
                    doi = paper_filter.normalize_doi(paper.get("doi"))
                    if key and key in excluded_keys:
                        stats["excluded_papers"] += 1
                    elif key and key in existing_keys:
                        stats["existing_papers"] += 1
                    elif doi and doi in existing_dois:
                        stats["existing_papers"] += 1
                        stats["existing_doi_papers"] += 1
                    else:
                        retained.append(paper)
                    if stats["scanned_papers"] % 2000 == 0:
                        self.live_progress(
                            "New-paper filter",
                            f"scanned={stats['scanned_papers']} excluded={stats['excluded_papers']} "
                            f"existing={stats['existing_papers']} new={stats['new_papers'] + len(retained)}",
                        )
                if retained:
                    paper_filter.write_json_array_atomic(output / path.name, retained)
                    stats["new_papers"] += len(retained)
                    stats["output_files"] += 1
                    by_source[path.stem] = len(retained)
            stats.update({"existing_dblp_keys": len(existing_keys), "existing_dois": len(existing_dois), "excluded_dblp_keys": len(excluded_keys), "by_source": by_source})
            write_json(self.update_dir / "filter_manifest.json", stats)
            self._new_candidates = stats["new_papers"]

        self._new_candidates = 0
        if self.prompt_yes_skip("Filter new DBLP papers?"):
            if not excluded_keys:
                print("Cannot rebuild new-paper filter: paper_exclude cache is missing. Export Stage 2 metadata first.")
                raise AbortRun
            if not self.run_stage("New-paper filter", action):
                raise AbortRun
        else:
            stats = load_json(manifest_path)
            cached_candidates = int(stats.get("new_papers", 0) or 0)
            if not manifest_path.exists() or (cached_candidates and not has_split_files(self.update_dir / "split_source")):
                print("Cannot skip new-paper filter: cached filter output is incomplete.")
                raise AbortRun
            self._new_candidates = cached_candidates
            print("Using cached new-paper filter output.")
        stats = load_json(manifest_path)
        print(f"Scanned: {stats.get('scanned_papers', 0)} | Excluded: {stats.get('excluded_papers', 0)} | Existing key/DOI: {stats.get('existing_papers', 0)} | New: {stats.get('new_papers', 0)}")
        return self._new_candidates

    def run_enrichment(self, directory: Path, *, existing: bool) -> None:
        prefix = "Existing " if existing else ""
        openalex_args = self.service_args(enrich_openalex_by_doi, ["--input-dir", str(directory / "split_source"), "--output-dir", str(directory), "--cache", str(directory / "cache/openalex_doi_cache.jsonl"), "--progress-every", "50", "--overwrite"])
        self.run_required_openalex(f"{prefix}OpenAlex DOI enrichment", lambda: enrich_openalex_by_doi.run(openalex_args), directory / "openalex_manifest.json")
        missing_doi = directory / "missing/_missing_doi.json"
        if not existing and missing_doi.exists() and missing_doi.stat().st_size > 2:
            title_args = self.service_args(enrich_openalex_missing_doi_by_search, ["--papers-dir", str(directory), "--cache", str(directory / "cache/openalex_missing_doi_search_cache.jsonl"), "--use-env-api-key", "--progress-every", "50"])
            self.run_required_openalex(f"{prefix}OpenAlex title search", lambda: enrich_openalex_missing_doi_by_search.run(title_args), directory / "openalex_missing_doi_search_manifest.json")

        for label, module, values in (
            (f"{prefix}Semantic Scholar", enrich_semantic_scholar_missing, ["--papers-dir", str(directory), "--cache", str(directory / "cache/semantic_scholar_doi_cache.jsonl"), "--use-env-api-key", "--progress-every", "50"]),
            (f"{prefix}Crossref", enrich_crossref_missing, ["--papers-dir", str(directory), "--cache", str(directory / "cache/crossref_doi_cache.jsonl"), "--use-env-mailto", "--progress-every", "50"]),
        ):
            eligible = count_doi_missing_abstract(directory)
            if not eligible:
                print(f"  {label}: no eligible records")
                continue
            if not self.prompt_yes_skip(f"Run {label} for {eligible} records?"):
                self.report[label] = "skipped"
                continue
            service_args = self.service_args(module, values)
            self.report[label] = "executed" if self.run_stage(label, lambda m=module, a=service_args: m.run(a), nonfatal=True) else "failed"

    def deduplicate_recovered_dois(self, existing_dois: set[str]) -> dict[str, int]:
        seen: set[str] = set()
        stats = {"removed_existing_doi": 0, "removed_duplicate_doi": 0, "retained_papers": 0, "retained_with_doi": 0}
        for folder in (self.update_dir / "enriched", self.update_dir / "missing"):
            for path in sorted(folder.glob("*.json")):
                if path.name == "_missing_doi.json":
                    continue
                records = json.loads(path.read_text(encoding="utf-8"))
                retained: list[dict[str, Any]] = []
                for record in records:
                    doi = zilliz_export.normalize_doi(record.get("doi"))
                    if doi and doi in existing_dois:
                        stats["removed_existing_doi"] += 1
                    elif doi and doi in seen:
                        stats["removed_duplicate_doi"] += 1
                    else:
                        retained.append(record)
                        if doi:
                            seen.add(doi)
                            stats["retained_with_doi"] += 1
                paper_filter.write_json_array_atomic(path, retained)
                stats["retained_papers"] += len(retained)
        write_json(self.update_dir / "post_doi_dedupe_manifest.json", stats)
        return stats

    def enrich_new(self, candidates: int, existing_dois: set[str]) -> int:
        self.stage(4, "New-paper enrichment and DOI deduplication")
        if not candidates or not self.prompt_yes_skip("Start new-paper enrichment?"):
            self.report["new enrichment"] = "skipped"
            return 0
        self.run_enrichment(self.update_dir, existing=False)
        dedup: dict[str, int] = {}
        if not self.run_stage("Recovered-DOI deduplication", lambda: dedup.update(self.deduplicate_recovered_dois(existing_dois))):
            raise AbortRun
        title_stats = load_json(self.update_dir / "openalex_missing_doi_search_manifest.json").get("stats", {})
        counts = count_records(self.update_dir)
        print(f"Recovered DOI: {title_stats.get('recovered_doi', 0)} | Already in paper_new: {dedup.get('removed_existing_doi', 0)} | Batch duplicates: {dedup.get('removed_duplicate_doi', 0)}")
        print(f"Ready with DOI: {counts['with_doi']} | Still without DOI: {counts['total'] - counts['with_doi']} | With abstract: {counts['with_abstract']}")
        self.report["new enrichment"] = "executed"
        return counts["with_doi"]

    def upload_new(self, candidates: int) -> None:
        self.stage(5, "New-paper upload")
        if not candidates:
            print("No uploadable new records.")
            self.report["new upload"] = "skipped"
            return
        counts = count_records(self.update_dir)
        print(f"Uploadable with DOI: {counts['with_doi']} | Without DOI skipped: {counts['total'] - counts['with_doi']} | With abstract: {counts['with_abstract']} | Missing abstract: {counts['missing_abstract']}")
        if not counts["with_doi"] or not self.prompt_yes_skip("Upload new records?", write=True):
            self.report["new upload"] = "skipped"
            return
        uploaded: set[str] = set()
        def action() -> None:
            collection = paper_upload.connect_collection(PAPER_NEW)
            paper_upload.validate_schema(collection)
            batch: list[dict[str, Any]] = []
            batch_uids: list[str] = []

            def insert_current_batch() -> None:
                if not batch:
                    return
                collection.insert(batch)
                uploaded.update(batch_uids)
                self.changed_uids.update(batch_uids)
                self.new_uploaded = len(uploaded)
                print(f"  new upload: inserted={len(uploaded)}", flush=True)
                batch.clear()
                batch_uids.clear()

            try:
                for _, record in read_records(self.update_dir):
                    entity = paper_upload.to_entity(record)
                    if not entity["doi"]:
                        continue
                    batch.append(entity)
                    batch_uids.append(str(entity["paper_uid"]))
                    if len(batch) >= 500:
                        insert_current_batch()
                insert_current_batch()
            finally:
                collection.flush()
        if not self.run_stage("New-paper upload", action):
            raise AbortRun
        self.changed_uids.update(uploaded)
        self.new_uploaded = len(uploaded)
        self.report["new upload"] = "executed"

    def backfill_existing(self) -> None:
        self.stage(6, "Existing-paper abstract backfill")
        candidates = int(load_json(self.existing_update_dir / "existing_missing_abstract_manifest.json").get("existing_doi_missing_abstract", 0) or 0)
        if not candidates or not self.prompt_yes_skip(f"Run existing-paper backfill for {candidates} records?"):
            self.report["existing backfill"] = "skipped"
            return
        self.run_enrichment(self.existing_update_dir, existing=True)
        counts = count_records(self.existing_update_dir)
        print(f"Existing candidates: {candidates} | Abstracts recovered: {counts['with_abstract']} | Still without abstract: {counts['missing_abstract']}")
        if not counts["with_abstract"] or not self.prompt_yes_skip("Partial-upsert recovered abstracts?", write=True):
            self.report["existing upsert"] = "skipped"
            return
        upserted: set[str] = set()
        def action() -> None:
            client = paper_upsert.connect_client()
            batch: list[dict[str, Any]] = []
            batch_uids: list[str] = []

            def upsert_current_batch() -> None:
                if not batch:
                    return
                client.upsert(collection_name=PAPER_NEW, data=batch, partial_update=True)
                upserted.update(batch_uids)
                self.changed_uids.update(batch_uids)
                self.existing_upserted = len(upserted)
                print(f"  existing upsert: updated={len(upserted)}", flush=True)
                batch.clear()
                batch_uids.clear()

            try:
                for _, record in read_records(self.existing_update_dir):
                    if not paper_upsert.has_abstract(record):
                        continue
                    entity = paper_upsert.to_update_entity(record)
                    batch.append(entity)
                    batch_uids.append(str(entity["paper_uid"]))
                    if len(batch) >= 500:
                        upsert_current_batch()
                upsert_current_batch()
            finally:
                client.flush(PAPER_NEW)
        if not self.run_stage("Existing partial upsert", action):
            raise AbortRun
        self.changed_uids.update(upserted)
        self.existing_upserted = len(upserted)
        self.report["existing backfill"] = "executed"
        self.report["existing upsert"] = "executed"

    def refresh_stats(self) -> None:
        self.stage(7, "paper_stats")
        if not self.changed_uids:
            print("No paper_new writes; skipping paper_stats.")
            self.report["paper_stats"] = "skipped"
            return
        print(f"paper_new changes: new={self.new_uploaded} existing_updates={self.existing_upserted}")
        if not self.prompt_yes_skip("Refresh paper_stats?", write=True):
            self.report["paper_stats"] = "skipped"
            return
        import materialize_paper_stats
        stats_args = materialize_paper_stats.parse_args(
            [
                "--source-collection", PAPER_NEW,
                "--stats-collection", "paper_stats",
                "--read-batch-size", "5000",
                "--write-batch-size", "500",
                "--progress-every", "2000",
                "--replace",
            ]
        )
        if not self.run_stage("paper_stats refresh", lambda: materialize_paper_stats.materialize(stats_args)):
            raise AbortRun
        self.report["paper_stats"] = "executed"

    def sync_changed_to_prod(self) -> None:
        self.stage(8, "Production sync")
        full_sync = False
        if not self.changed_uids:
            if not self.prompt_yes_skip(
                "No paper_new writes. Scan all eligible paper_new records and sync paper_prod?",
                write=True,
            ):
                self.report["paper_prod sync"] = "skipped"
                return
            full_sync = True
        elif not self.prompt_yes_skip(f"Sync {len(self.changed_uids)} changed papers to paper_prod?", write=True):
            self.report["paper_prod sync"] = "skipped"
            return
        stats = production_sync.RunStats()

        def action() -> None:
            client = production_sync.connect_zilliz()
            prod_ready = False
            try:
                production_sync.ensure_prod_collection()
                prod_ready = True
                if full_sync:
                    source_batches = production_sync.iter_eligible_batches(
                        client,
                        PAPER_NEW,
                        timeout=production_sync.QUERY_TIMEOUT_SECONDS,
                    )
                else:
                    uids = sorted(self.changed_uids)

                    def changed_batches() -> Iterable[list[dict[str, Any]]]:
                        found = 0
                        for index, group in enumerate(chunked(uids, production_sync.BATCH_SIZE), start=1):
                            rows = client.query(
                                collection_name=PAPER_NEW,
                                filter=production_sync.uid_in_expr(group),
                                output_fields=production_sync.DEV_OUTPUT_FIELDS,
                                limit=len(group) + 10,
                                timeout=production_sync.QUERY_TIMEOUT_SECONDS,
                            )
                            found += len(rows)
                            eligible = [row for row in rows if production_sync.is_eligible_row(row)]
                            self.live_progress(
                                "paper_new lookup",
                                f"batch={index} requested={min(index * production_sync.BATCH_SIZE, len(uids))}/{len(uids)} "
                                f"found={found} eligible={len(eligible)}",
                            )
                            if eligible:
                                yield eligible

                    source_batches = changed_batches()

                embedder: production_sync.AzureEmbedder | None = None
                eligible_total = 0
                for index, group in enumerate(source_batches, start=1):
                    eligible_total += len(group)
                    if embedder is None:
                        embedder = production_sync.AzureEmbedder()
                    prod = production_sync.lookup_prod_rows(client, PAPER_PROD, [str(row["paper_uid"]) for row in group])
                    classified = production_sync.classify_batch(group, prod, embedding_model=production_sync.EMBEDDING_MODEL)
                    embed_rows = classified.new + classified.embed_input_change
                    vectors, failures = production_sync.embed_rows(embedder, embed_rows)
                    self.production_umap_uids.update(
                        uid for uid, vector in vectors.items() if vector is not None
                    )
                    production_sync.upsert_entities(client, PAPER_PROD, production_sync.build_full_upsert_entities(embed_rows, embeddings_by_uid=vectors, embedding_model=production_sync.EMBEDDING_MODEL))
                    production_sync.upsert_entities(client, PAPER_PROD, production_sync.build_metadata_partial_entities(classified.metadata_only_change), partial_update=True)
                    stats.new += len(classified.new); stats.embed_input_change += len(classified.embed_input_change); stats.metadata_only_change += len(classified.metadata_only_change); stats.unchanged += len(classified.unchanged); stats.embedding_failures += failures; stats.upserted += len(embed_rows) + len(classified.metadata_only_change)
                    self.live_progress(
                        "production sync",
                        f"batch={index} eligible={eligible_total} new={stats.new} reembed={stats.embed_input_change} "
                        f"metadata={stats.metadata_only_change} unchanged={stats.unchanged} failures={stats.embedding_failures}",
                    )
            finally:
                if prod_ready:
                    client.flush(PAPER_PROD)
        label = "Full paper_prod sync" if full_sync else "Incremental paper_prod sync"
        if not self.run_stage(label, action):
            raise AbortRun
        print(f"Production synced: new={stats.new} reembed={stats.embed_input_change} metadata={stats.metadata_only_change} unchanged={stats.unchanged} failures={stats.embedding_failures}")
        self.report["paper_prod sync"] = "executed"

    def update_production_umap(self) -> None:
        self.stage(9, "Production UMAP")
        if self.report.get("paper_prod sync") != "executed":
            print("Production sync was not executed; skipping UMAP.")
            self.report["paper_prod umap"] = "skipped"
            return
        if not self.production_umap_uids:
            print("No successful embedding changes; skipping UMAP.")
            self.report["paper_prod umap"] = "skipped"
            return
        if not umap_projection.DEFAULT_MODEL_PATH.exists():
            print(f"UMAP model is unavailable; skipping ({umap_projection.DEFAULT_MODEL_PATH.relative_to(PROJECT_ROOT)}).")
            self.report["paper_prod umap"] = "skipped"
            return
        count = len(self.production_umap_uids)
        if not self.prompt_yes_skip(f"Update UMAP for {count} embedded papers?", write=True):
            self.report["paper_prod umap"] = "skipped"
            return

        def action() -> None:
            client = production_sync.connect_zilliz()
            self.production_umap_updated = umap_projection.transform_umap_uids(
                client,
                collection_name=PAPER_PROD,
                paper_uids=sorted(self.production_umap_uids),
                batch_size=production_sync.BATCH_SIZE,
                query_timeout=production_sync.QUERY_TIMEOUT_SECONDS,
                model_path=umap_projection.DEFAULT_MODEL_PATH,
                embedding_dim=EMBEDDING_DIM,
                upsert_batch_size=production_sync.BATCH_SIZE,
                execute=True,
            )
            print(f"UMAP updated: {self.production_umap_updated}", flush=True)

        if not self.run_stage("Incremental production UMAP", action):
            raise AbortRun
        self.report["paper_prod umap"] = "executed"

    def print_report(self) -> None:
        manifest = self.update_dir / "update_cli_manifest.json"
        write_json(manifest, {"date": self.update_date, "new_uploaded": self.new_uploaded, "existing_upserted": self.existing_upserted, "production_umap_updated": self.production_umap_updated, "changed_uids": sorted(self.changed_uids), "stages": self.report})
        print("\n=== Update report ===")
        print(f"Run manifest: {manifest.relative_to(PROJECT_ROOT)}")
        for name in ("new enrichment", "new upload", "existing backfill", "existing upsert", "paper_stats", "paper_prod sync", "paper_prod umap"):
            print(f"{name}: {self.report.get(name, 'skipped')}")

    def run_all(self) -> None:
        self.prepare_dblp()
        keys, dois, excluded = self.export_existing()
        candidates = self.filter_new(keys, dois, excluded)
        self.upload_new(self.enrich_new(candidates, dois))
        self.backfill_existing()
        self.refresh_stats()
        self.sync_changed_to_prod()
        self.update_production_umap()
        self.print_report()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the in-process incremental paper update workflow.")
    parser.add_argument("--date", default=time.strftime("%Y%m%d"), help="Update date in YYYYMMDD format.")
    parser.add_argument("--yes", action="store_true", help="Answer yes to update CLI confirmations.")
    parser.add_argument("--no-write", action="store_true", help="Skip all Zilliz writes, stats, and production sync.")
    args = parser.parse_args()
    if len(args.date) != 8 or not args.date.isdigit():
        parser.error("--date must use YYYYMMDD")
    return args


def main() -> int:
    args = parse_args()
    cli = UpdateCLI(args, args.date, PROJECT_ROOT / f"data/papers/update{args.date}", PROJECT_ROOT / f"data/papers/update_missing{args.date}")
    try:
        cli.run_all()
    except (AbortRun, KeyboardInterrupt):
        print("\nUpdate aborted.")
        cli.print_report()
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
