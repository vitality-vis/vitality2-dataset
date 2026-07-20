#!/usr/bin/env python3
"""Split DBLP XML dump into per-source JSON files.

Only <article> and <inproceedings> records are exported.  The parser is SAX
based so the multi-GB DBLP dump is streamed instead of loaded into memory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import xml.sax
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


RECORD_TAGS = {"article", "inproceedings"}
FIELD_TAGS = {"author", "title", "journal", "booktitle", "year", "ee"}
SOURCE_FIELDS = {"article": "journal", "inproceedings": "booktitle"}
DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
)


@dataclass(frozen=True)
class SourceMappingEntry:
    source: str
    fullpaper: bool


def first(values: list[str]) -> str:
    return values[0] if values else ""


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def extract_doi(ees: list[str]) -> str:
    for ee in ees:
        value = ee.strip()
        lower = value.lower()
        for prefix in DOI_PREFIXES:
            if lower.startswith(prefix):
                return value[len(prefix) :].strip()
        if lower.startswith("10."):
            return value
    return ""


def parse_bool(value: str) -> bool:
    normalized = normalize_text(value).casefold()
    if normalized in {"yes", "true", "1", "y"}:
        return True
    if normalized in {"no", "false", "0", "n"}:
        return False
    raise SystemExit(f"Invalid Full paper value: {value!r}")


def load_source_mapping(path: Path) -> dict[str, SourceMappingEntry]:
    """Load dblp_source -> source/fullpaper mapping from CSV."""

    mapping: dict[str, SourceMappingEntry] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source", "dblp_source", "Full paper"}
        if not required.issubset(reader.fieldnames or []):
            raise SystemExit(f"{path} must contain columns: source,dblp_source,Full paper")

        for row in reader:
            source = normalize_text(row.get("source", ""))
            dblp_source = normalize_text(row.get("dblp_source", ""))
            fullpaper = parse_bool(row.get("Full paper", "yes"))
            if not source or not dblp_source:
                continue

            existing = mapping.get(dblp_source)
            if existing is not None and existing.source != source:
                raise SystemExit(
                    f"Conflicting mapping for DBLP source {dblp_source!r}: "
                    f"{existing.source!r} vs {source!r}"
                )
            if existing is not None and existing.fullpaper != fullpaper:
                raise SystemExit(
                    f"Conflicting Full paper value for DBLP source {dblp_source!r}: "
                    f"{existing.fullpaper!r} vs {fullpaper!r}"
                )
            mapping[dblp_source] = SourceMappingEntry(source=source, fullpaper=fullpaper)

    if not mapping:
        raise SystemExit(f"No source mappings loaded from {path}")
    return mapping


def safe_filename(source: str, used: dict[str, str]) -> str:
    """Return a stable, filesystem-safe filename for a source string."""

    normalized = normalize_text(source) or "__unknown_source__"
    safe = re.sub(r"[^A-Za-z0-9._ -]+", "_", normalized)
    safe = re.sub(r"\s+", " ", safe).strip(" ._")
    safe = safe[:160] or "__unknown_source__"

    filename = f"{safe}.json"
    filename_key = filename.casefold()
    existing = used.get(filename_key)
    if existing is None or existing == normalized:
        used[filename_key] = normalized
        return filename

    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    safe_with_hash = f"{safe[:149]}_{digest}.json"
    used[safe_with_hash.casefold()] = normalized
    return safe_with_hash


class SourceJsonWriter:
    def __init__(self, output_dir: Path, max_open_files: int) -> None:
        self.output_dir = output_dir
        self.max_open_files = max_open_files
        self.source_to_file: dict[str, str] = {}
        self.safe_to_source: dict[str, str] = {}
        self.has_items: set[str] = set()
        self.handles: OrderedDict[str, TextIO] = OrderedDict()
        self.counts: dict[str, int] = {}

    def _open(self, filename: str) -> TextIO:
        handle = self.handles.get(filename)
        if handle is not None:
            self.handles.move_to_end(filename)
            return handle

        if len(self.handles) >= self.max_open_files:
            _, old_handle = self.handles.popitem(last=False)
            old_handle.close()

        path = self.output_dir / filename
        if filename in self.has_items:
            handle = path.open("a", encoding="utf-8")
        else:
            handle = path.open("w", encoding="utf-8")
            handle.write("[")
        self.handles[filename] = handle
        return handle

    def write(self, source: str, paper: dict[str, object]) -> None:
        source = normalize_text(source) or "__unknown_source__"
        filename = self.source_to_file.get(source)
        if filename is None:
            filename = safe_filename(source, self.safe_to_source)
            self.source_to_file[source] = filename

        handle = self._open(filename)
        if filename in self.has_items:
            handle.write(",\n")
        else:
            handle.write("\n")
            self.has_items.add(filename)
        json.dump(paper, handle, ensure_ascii=False, separators=(",", ":"))
        self.counts[filename] = self.counts.get(filename, 0) + 1

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

        for filename in self.has_items:
            with (self.output_dir / filename).open("a", encoding="utf-8") as handle:
                handle.write("\n]\n")

        manifest = {
            source: {
                "file": filename,
                "count": self.counts.get(filename, 0),
            }
            for source, filename in sorted(self.source_to_file.items())
        }
        with (self.output_dir / "_source_manifest.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


class DblpHandler(xml.sax.handler.ContentHandler):
    def __init__(
        self,
        writer: SourceJsonWriter,
        source_mapping: dict[str, SourceMappingEntry],
        limit: int | None = None,
    ) -> None:
        super().__init__()
        self.writer = writer
        self.source_mapping = source_mapping
        self.limit = limit
        self.current_record_type: str | None = None
        self.current_field: str | None = None
        self.current_chars: list[str] = []
        self.source_field: str | None = None
        self.title = ""
        self.authors: list[str] = []
        self.dblp_source = ""
        self.year = ""
        self.ees: list[str] = []
        self.exported = 0
        self.seen_target_records = 0
        self.skipped_unmapped = 0

    def startElement(self, name: str, attrs: xml.sax.xmlreader.AttributesImpl) -> None:
        if name in RECORD_TAGS:
            self.current_record_type = name
            self.source_field = SOURCE_FIELDS[name]
            self.title = ""
            self.authors = []
            self.dblp_source = ""
            self.year = ""
            self.ees = []
            return

        if self.current_record_type and name in FIELD_TAGS:
            self.current_field = name
            self.current_chars = []

    def characters(self, content: str) -> None:
        if self.current_field:
            self.current_chars.append(content)

    def endElement(self, name: str) -> None:
        if self.current_record_type and name == self.current_field:
            value = normalize_text("".join(self.current_chars))
            if value:
                if name == "author":
                    self.authors.append(value)
                elif name == "title" and not self.title:
                    self.title = value
                elif name == self.source_field and not self.dblp_source:
                    self.dblp_source = value
                elif name == "year" and not self.year:
                    self.year = value
                elif name == "ee":
                    self.ees.append(value)
            self.current_field = None
            self.current_chars = []
            return

        if name == self.current_record_type:
            self.seen_target_records += 1
            mapping_entry = self.source_mapping.get(self.dblp_source)
            if mapping_entry is None:
                self.skipped_unmapped += 1
            else:
                paper = self._build_paper(mapping_entry)
                self.writer.write(mapping_entry.source, paper)
                self.exported += 1
            self.current_record_type = None
            self.current_field = None
            self.source_field = None

            if self.limit is not None and self.exported >= self.limit:
                raise StopParsing()

    def _build_paper(self, mapping_entry: SourceMappingEntry) -> dict[str, object]:
        return {
            "title": self.title,
            "authors": self.authors,
            "source": mapping_entry.source,
            "dblp_source": self.dblp_source,
            "year": self.year,
            "doi": extract_doi(self.ees),
            "abstract": "",
            "keywords": [],
            "citationCounts": None,
            "fullpaper": mapping_entry.fullpaper,
        }


class StopParsing(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split DBLP article/inproceedings records into per-source JSON files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/dblp/dump/dblp.xml"),
        help="Path to DBLP XML dump.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/dblp/split_source"),
        help="Directory for per-source JSON files.",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("data/dblp/source_mapping.csv"),
        help="CSV with source,dblp_source,Full paper columns.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output directory before writing.",
    )
    parser.add_argument(
        "--max-open-files",
        type=int,
        default=128,
        help="Maximum number of source JSON files kept open at once.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for testing; stops after exporting this many records.",
    )
    return parser.parse_args()


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_json = list(output_dir.glob("*.json"))
    if existing_json:
        raise SystemExit(
            f"{output_dir} already contains JSON files. "
            "Use --overwrite to replace them."
        )


def main() -> int:
    args = parse_args()
    source_mapping = load_source_mapping(args.mapping)
    prepare_output_dir(args.output_dir, args.overwrite)

    writer = SourceJsonWriter(args.output_dir, args.max_open_files)
    handler = DblpHandler(writer, source_mapping, limit=args.limit)
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)

    try:
        parser.parse(str(args.input))
    except StopParsing:
        pass
    finally:
        writer.close()

    print(f"exported records: {handler.exported}", file=sys.stderr)
    print(f"target records seen: {handler.seen_target_records}", file=sys.stderr)
    print(f"skipped unmapped records: {handler.skipped_unmapped}", file=sys.stderr)
    print(f"mapped DBLP sources: {len(source_mapping)}", file=sys.stderr)
    print(f"sources: {len(writer.source_to_file)}", file=sys.stderr)
    print(f"output: {args.output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
