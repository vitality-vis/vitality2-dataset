#!/usr/bin/env python3
"""Filter split DBLP papers against existing DBLP keys.

The output is a per-source split-source directory suitable as the OpenAlex
input for an update batch:

    python3 script/enrich_openalex_by_doi.py \
      --input-dir data/papers/updateYYYYMMDD/split_source \
      --output-dir data/papers/updateYYYYMMDD
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


DEFAULT_SPLIT_DIR = Path("data/dblp/split_source")
DEFAULT_EXISTING_FILE = Path("data/zilliz/paper_new_dblp_keys.txt")
DEFAULT_EXCLUDE_TITLE_FILE = Path("data/dblp/exclude_title.txt")


def normalize_value(value: Any) -> str:
    return str(value or "").strip()


def normalize_doi(value: Any) -> str:
    doi = normalize_value(value)
    lower = doi.lower()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ):
        if lower.startswith(prefix):
            return doi[len(prefix) :].strip().casefold()
    return doi.casefold()


def normalize_title(value: Any) -> str:
    return " ".join(normalize_value(value).casefold().split())


def default_update_dir() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d")
    return Path("data/papers") / f"update{stamp}"


def load_existing_values(path: Path, field: str) -> set[str]:
    if not path.exists():
        raise SystemExit(f"Existing {field} file does not exist: {path}")

    values: set[str] = set()
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON array")
        for item in data:
            if isinstance(item, dict):
                value = normalize_value(item.get(field))
            else:
                value = normalize_value(item)
            if value:
                values.add(value)
        return values

    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("{") or line.startswith('"'):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON line {line_no} in {path}: {exc}") from exc
                if isinstance(item, dict):
                    value = normalize_value(item.get(field))
                else:
                    value = normalize_value(item)
            else:
                value = line
            if value:
                values.add(value)
    return values


def load_excluded_titles(path: Path) -> set[str]:
    if not path.exists():
        raise SystemExit(f"Exclude title file does not exist: {path}")
    return {
        title
        for line in path.read_text(encoding="utf-8").splitlines()
        if (title := normalize_title(line)) and not title.startswith("#")
    }


def load_json_array(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
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


def iter_split_files(split_dir: Path) -> list[Path]:
    return sorted(path for path in split_dir.glob("*.json") if not path.name.startswith("_"))


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    split_output_dir = output_dir / "split_source"
    if split_output_dir.exists() and any(split_output_dir.iterdir()) and not overwrite:
        raise SystemExit(f"Output split_source directory is not empty: {split_output_dir}. Use --overwrite.")
    if overwrite and split_output_dir.exists():
        for path in split_output_dir.glob("*.json"):
            path.unlink()
    split_output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "enriched").mkdir(parents=True, exist_ok=True)
    (output_dir / "missing").mkdir(parents=True, exist_ok=True)
    return split_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter split DBLP papers to a new-paper update batch.")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--existing-file", type=Path, default=DEFAULT_EXISTING_FILE)
    parser.add_argument(
        "--exclude-title-file",
        type=Path,
        default=DEFAULT_EXCLUDE_TITLE_FILE,
        help="One title per line. Matching records are excluded from the update batch.",
    )
    parser.add_argument(
        "--existing-doi-file",
        type=Path,
        default=None,
        help="Optional DOI file. Papers matching either existing key or existing DOI are filtered out.",
    )
    parser.add_argument(
        "--field",
        choices=("dblp_key", "paper_uid"),
        default="dblp_key",
        help="Existing-record key to compare. Use dblp_key for DBLP incremental updates.",
    )
    parser.add_argument(
        "--uid-file",
        type=Path,
        default=None,
        help="Deprecated alias for --existing-file; kept for old paper_uid-based tests.",
    )
    parser.add_argument("--output-dir", type=Path, default=default_update_dir())
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional output limit for testing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.uid_file is not None:
        args.existing_file = args.uid_file
        args.field = "paper_uid"

    existing_values = load_existing_values(args.existing_file, args.field)
    excluded_titles = load_excluded_titles(args.exclude_title_file)
    existing_dois = (
        {normalize_doi(value) for value in load_existing_values(args.existing_doi_file, "doi")}
        if args.existing_doi_file is not None
        else set()
    )
    split_output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    files = iter_split_files(args.split_dir)
    if not files:
        raise SystemExit(f"No split JSON files found in {args.split_dir}")

    stats = {
        "split_files": len(files),
        "filter_field": args.field,
        "existing_values": len(existing_values),
        "existing_dois": len(existing_dois),
        "excluded_titles": len(excluded_titles),
        "excluded_title_papers": 0,
        "scanned_papers": 0,
        "existing_papers": 0,
        "existing_doi_papers": 0,
        "new_papers": 0,
        "missing_filter_field": 0,
        "output_files": 0,
    }
    by_source: dict[str, int] = {}

    stopped = False
    for path in files:
        papers = load_json_array(path)
        new_papers: list[dict[str, Any]] = []
        for paper in papers:
            stats["scanned_papers"] += 1
            value = normalize_value(paper.get(args.field))
            doi = normalize_doi(paper.get("doi"))
            if normalize_title(paper.get("title")) in excluded_titles:
                stats["excluded_title_papers"] += 1
            elif not value:
                stats["missing_filter_field"] += 1
                if doi and doi in existing_dois:
                    stats["existing_doi_papers"] += 1
                    stats["existing_papers"] += 1
                else:
                    new_papers.append(paper)
            elif value in existing_values:
                stats["existing_papers"] += 1
            elif doi and doi in existing_dois:
                stats["existing_doi_papers"] += 1
                stats["existing_papers"] += 1
            else:
                new_papers.append(paper)

            if args.limit is not None and stats["new_papers"] + len(new_papers) >= args.limit:
                keep = args.limit - stats["new_papers"]
                new_papers = new_papers[:keep]
                stopped = True
                break

        if new_papers:
            write_json_array_atomic(split_output_dir / path.name, new_papers)
            stats["new_papers"] += len(new_papers)
            stats["output_files"] += 1
            by_source[path.stem] = len(new_papers)

        if stopped:
            break

    manifest = {
        **stats,
        "split_dir": str(args.split_dir),
        "existing_file": str(args.existing_file),
        "existing_doi_file": str(args.existing_doi_file) if args.existing_doi_file else None,
        "exclude_title_file": str(args.exclude_title_file),
        "output_dir": str(args.output_dir),
        "split_source_dir": str(split_output_dir),
        "by_source": dict(sorted(by_source.items())),
    }
    manifest_path = args.output_dir / "filter_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
