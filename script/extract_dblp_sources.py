#!/usr/bin/env python3
"""Extract source names, paper counts, and DOI examples from DBLP XML dump.

For DBLP records, the source is:
- <journal> for <article>
- <booktitle> for <inproceedings>

The dump is parsed with SAX so the multi-GB XML file is streamed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import re
import sys
import xml.sax
from pathlib import Path


TARGET_RECORDS = {"article": "journal", "inproceedings": "booktitle"}
DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def extract_doi(value: str) -> str:
    lower = value.lower()
    for prefix in DOI_PREFIXES:
        if lower.startswith(prefix):
            return value[len(prefix) :].strip()
    if lower.startswith("10."):
        return value
    return ""


class DblpSourceHandler(xml.sax.handler.ContentHandler):
    def __init__(self) -> None:
        super().__init__()
        self.current_record: str | None = None
        self.source_field: str | None = None
        self.current_field: str | None = None
        self.chars: list[str] = []
        self.current_source = ""
        self.current_dois: list[str] = []
        self.source_counts: Counter[str] = Counter()
        self.source_doi_examples: dict[str, list[str]] = {}
        self.target_records = 0
        self.records_with_source = 0

    def startElement(self, name: str, attrs: xml.sax.xmlreader.AttributesImpl) -> None:
        if name in TARGET_RECORDS:
            self.current_record = name
            self.source_field = TARGET_RECORDS[name]
            self.current_source = ""
            self.current_dois = []
            self.target_records += 1
            return

        if self.current_record and name == self.source_field:
            self.current_field = name
            self.chars = []
            return

        if self.current_record and name == "ee":
            self.current_field = name
            self.chars = []

    def characters(self, content: str) -> None:
        if self.current_field:
            self.chars.append(content)

    def endElement(self, name: str) -> None:
        if self.current_record and name == self.current_field:
            value = normalize_text("".join(self.chars))
            if name == self.source_field:
                self.current_source = value
            elif name == "ee":
                doi = extract_doi(value)
                if doi:
                    self.current_dois.append(doi)
            self.current_field = None
            self.chars = []
            return

        if name == self.current_record:
            if self.current_source:
                self.source_counts[self.current_source] += 1
                examples = self.source_doi_examples.setdefault(self.current_source, [])
                for doi in self.current_dois:
                    if doi not in examples and len(examples) < 3:
                        examples.append(doi)
                self.records_with_source += 1
            self.current_record = None
            self.source_field = None
            self.current_source = ""
            self.current_dois = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract unique article/inproceedings source names from DBLP."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/dblp/dump/dblp.xml"),
        help="Path to DBLP XML dump.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dblp/dblp_source_list.txt"),
        help=(
            "Output text file, one source per line as: "
            "source<TAB>paper_count<TAB>doi_examples."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    handler = DblpSourceHandler()
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)
    parser.parse(str(args.input))

    with args.output.open("w", encoding="utf-8") as handle:
        for source, count in sorted(
            handler.source_counts.items(), key=lambda item: item[0].casefold()
        ):
            doi_examples = "; ".join(handler.source_doi_examples.get(source, []))
            handle.write(f"{source}\t{count}\t{doi_examples}\n")

    print(f"target records: {handler.target_records}", file=sys.stderr)
    print(f"records with source: {handler.records_with_source}", file=sys.stderr)
    print(f"unique sources: {len(handler.source_counts)}", file=sys.stderr)
    print(f"output: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
