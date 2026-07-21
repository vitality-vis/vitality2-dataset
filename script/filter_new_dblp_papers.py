#!/usr/bin/env python3
"""Filter split DBLP papers against existing paper_uid values.

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
DEFAULT_UID_FILE = Path("data/zilliz/paper_new_paper_uids.txt")


def normalize_uid(value: Any) -> str:
    return str(value or "").strip()


def default_update_dir() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d")
    return Path("data/papers") / f"update{stamp}"


def load_existing_uids(path: Path) -> set[str]:
    if not path.exists():
        raise SystemExit(f"UID file does not exist: {path}")

    uids: set[str] = set()
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON array")
        for item in data:
            if isinstance(item, dict):
                uid = normalize_uid(item.get("paper_uid"))
            else:
                uid = normalize_uid(item)
            if uid:
                uids.add(uid)
        return uids

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
                    uid = normalize_uid(item.get("paper_uid"))
                else:
                    uid = normalize_uid(item)
            else:
                uid = line
            if uid:
                uids.add(uid)
    return uids


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
    parser.add_argument("--uid-file", type=Path, default=DEFAULT_UID_FILE)
    parser.add_argument("--output-dir", type=Path, default=default_update_dir())
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional output limit for testing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    existing_uids = load_existing_uids(args.uid_file)
    split_output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    files = iter_split_files(args.split_dir)
    if not files:
        raise SystemExit(f"No split JSON files found in {args.split_dir}")

    stats = {
        "split_files": len(files),
        "existing_uids": len(existing_uids),
        "scanned_papers": 0,
        "existing_papers": 0,
        "new_papers": 0,
        "missing_paper_uid": 0,
        "output_files": 0,
    }
    by_source: dict[str, int] = {}

    stopped = False
    for path in files:
        papers = load_json_array(path)
        new_papers: list[dict[str, Any]] = []
        for paper in papers:
            stats["scanned_papers"] += 1
            uid = normalize_uid(paper.get("paper_uid"))
            if not uid:
                stats["missing_paper_uid"] += 1
                new_papers.append(paper)
            elif uid in existing_uids:
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
        "uid_file": str(args.uid_file),
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
