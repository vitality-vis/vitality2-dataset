#!/usr/bin/env python3
"""Generate a DBLP split-source data report.

The script reads one split JSON file at a time, so memory use does not scale
with the total number of papers.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SourceStats:
    source: str
    total: int = 0
    with_doi: int = 0
    without_doi: int = 0
    with_abstract: int = 0
    without_abstract: int = 0
    complete: int = 0
    with_doi_without_abstract: int = 0
    dblp_sources: Counter[str] = field(default_factory=Counter)
    years: Counter[str] = field(default_factory=Counter)


def percent(numerator: int, denominator: int) -> float:
    return (numerator / denominator * 100.0) if denominator else 0.0


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._ -]+", "_", value)
    safe = re.sub(r"\s+", "_", safe).strip("._")
    safe = safe.replace(" ", "_")
    return safe or "unknown"


def normalize_year(value: object) -> str:
    text = str(value or "").strip()
    return text if text.isdigit() else "Unknown"


def read_stats(input_dirs: list[Path]) -> tuple[dict[str, SourceStats], Counter[str]]:
    stats_by_source: dict[str, SourceStats] = {}
    overall_years: Counter[str] = Counter()

    for input_dir in input_dirs:
        for path in sorted(input_dir.glob("*.json")):
            if path.name == "_source_manifest.json":
                continue

            with path.open("r", encoding="utf-8") as handle:
                papers = json.load(handle)

            for paper in papers:
                source = str(paper.get("source") or "Unknown").strip() or "Unknown"
                stats = stats_by_source.setdefault(source, SourceStats(source=source))
                stats.total += 1

                dblp_source = str(paper.get("dblp_source") or "Unknown").strip() or "Unknown"
                stats.dblp_sources[dblp_source] += 1

                has_doi = bool(str(paper.get("doi") or "").strip())
                has_abstract = bool(str(paper.get("abstract") or "").strip())

                if has_doi:
                    stats.with_doi += 1
                else:
                    stats.without_doi += 1

                if has_abstract:
                    stats.with_abstract += 1
                else:
                    stats.without_abstract += 1

                if has_doi and has_abstract:
                    stats.complete += 1
                elif has_doi and not has_abstract:
                    stats.with_doi_without_abstract += 1

                year = normalize_year(paper.get("year"))
                stats.years[year] += 1
                overall_years[year] += 1

    return stats_by_source, overall_years


def write_summary_csv(path: Path, stats_by_source: dict[str, SourceStats]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "total",
                "with_doi",
                "without_doi",
                "doi_percent",
                "with_abstract",
                "without_abstract",
                "abstract_percent",
                "complete",
                "complete_percent",
                "with_doi_without_abstract",
                "with_doi_without_abstract_percent",
                "dblp_sources",
                "year_min",
                "year_max",
            ],
        )
        writer.writeheader()
        for stats in sorted(stats_by_source.values(), key=lambda item: item.source.casefold()):
            numeric_years = sorted(int(year) for year in stats.years if year.isdigit())
            writer.writerow(
                {
                    "source": stats.source,
                    "total": stats.total,
                    "with_doi": stats.with_doi,
                    "without_doi": stats.without_doi,
                    "doi_percent": f"{percent(stats.with_doi, stats.total):.2f}",
                    "with_abstract": stats.with_abstract,
                    "without_abstract": stats.without_abstract,
                    "abstract_percent": f"{percent(stats.with_abstract, stats.total):.2f}",
                    "complete": stats.complete,
                    "complete_percent": f"{percent(stats.complete, stats.total):.2f}",
                    "with_doi_without_abstract": stats.with_doi_without_abstract,
                    "with_doi_without_abstract_percent": f"{percent(stats.with_doi_without_abstract, stats.total):.2f}",
                    "dblp_sources": "; ".join(
                        f"{source}:{count}"
                        for source, count in sorted(
                            stats.dblp_sources.items(), key=lambda item: (-item[1], item[0].casefold())
                        )
                    ),
                    "year_min": numeric_years[0] if numeric_years else "",
                    "year_max": numeric_years[-1] if numeric_years else "",
                }
            )


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="#202124">{html.escape(text)}</text>'
    )


def write_doi_chart(path: Path, stats_by_source: dict[str, SourceStats]) -> None:
    rows = sorted(
        stats_by_source.values(),
        key=lambda item: (-percent(item.with_doi, item.total), -item.total, item.source.casefold()),
    )
    width = 1200
    left = 230
    right = 180
    top = 44
    row_h = 24
    bar_h = 14
    chart_w = width - left - right
    height = top + row_h * len(rows) + 34
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(20, 24, "DOI coverage by source", 18),
        svg_text(left, 42, "0%", 11, "middle"),
        svg_text(left + chart_w / 2, 42, "50%", 11, "middle"),
        svg_text(left + chart_w, 42, "100%", 11, "middle"),
    ]
    for i, stats in enumerate(rows):
        y = top + i * row_h
        doi_w = chart_w * percent(stats.with_doi, stats.total) / 100.0
        no_doi_w = chart_w - doi_w
        parts.append(svg_text(12, y + 13, stats.source, 12))
        parts.append(f'<rect x="{left}" y="{y}" width="{chart_w}" height="{bar_h}" fill="#e8eaed"/>')
        parts.append(f'<rect x="{left}" y="{y}" width="{doi_w:.1f}" height="{bar_h}" fill="#1a73e8"/>')
        if no_doi_w > 0:
            parts.append(
                f'<rect x="{left + doi_w:.1f}" y="{y}" width="{no_doi_w:.1f}" '
                f'height="{bar_h}" fill="#fbbc04"/>'
            )
        label = (
            f"{fmt_pct(percent(stats.with_doi, stats.total))} DOI "
            f"({stats.with_doi:,}/{stats.total:,})"
        )
        parts.append(svg_text(left + chart_w + 12, y + 12, label, 11))
    parts.append(svg_text(left, height - 10, "blue = with DOI, yellow = missing DOI", 11))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_completeness_chart(path: Path, stats_by_source: dict[str, SourceStats]) -> None:
    rows = sorted(
        stats_by_source.values(),
        key=lambda item: (-percent(item.complete, item.total), -item.total, item.source.casefold()),
    )
    width = 1200
    left = 230
    right = 230
    top = 44
    row_h = 24
    bar_h = 14
    chart_w = width - left - right
    height = top + row_h * len(rows) + 34
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(20, 24, "Paper completeness by source", 18),
        svg_text(left, 42, "0%", 11, "middle"),
        svg_text(left + chart_w / 2, 42, "50%", 11, "middle"),
        svg_text(left + chart_w, 42, "100%", 11, "middle"),
    ]
    for i, stats in enumerate(rows):
        y = top + i * row_h
        complete_w = chart_w * percent(stats.complete, stats.total) / 100.0
        no_abstract_w = chart_w * percent(stats.with_doi_without_abstract, stats.total) / 100.0
        no_doi_w = chart_w - complete_w - no_abstract_w
        parts.append(svg_text(12, y + 13, stats.source, 12))
        parts.append(f'<rect x="{left}" y="{y}" width="{chart_w}" height="{bar_h}" fill="#e8eaed"/>')
        parts.append(f'<rect x="{left}" y="{y}" width="{complete_w:.1f}" height="{bar_h}" fill="#188038"/>')
        if no_abstract_w > 0:
            parts.append(
                f'<rect x="{left + complete_w:.1f}" y="{y}" width="{no_abstract_w:.1f}" '
                f'height="{bar_h}" fill="#d93025"/>'
            )
        if no_doi_w > 0:
            parts.append(
                f'<rect x="{left + complete_w + no_abstract_w:.1f}" y="{y}" width="{no_doi_w:.1f}" '
                f'height="{bar_h}" fill="#fbbc04"/>'
            )
        label = (
            f"{fmt_pct(percent(stats.complete, stats.total))} complete "
            f"({stats.complete:,}/{stats.total:,})"
        )
        parts.append(svg_text(left + chart_w + 12, y + 12, label, 11))
    parts.append(svg_text(left, height - 10, "green = complete, red = missing abstract, yellow = missing DOI", 11))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_year_chart(path: Path, years_counter: Counter[str], title: str) -> None:
    numeric_years = sorted(int(year) for year in years_counter if year.isdigit())
    years = [str(year) for year in range(numeric_years[0], numeric_years[-1] + 1)] if numeric_years else []
    counts = [years_counter[year] for year in years]
    max_count = max(counts) if counts else 1

    width = max(900, min(1800, 38 * max(len(years), 1)))
    height = 520
    left = 70
    right = 28
    top = 44
    bottom = 92
    chart_w = width - left - right
    chart_h = height - top - bottom
    bar_gap = 2
    bar_w = max(4, (chart_w - bar_gap * max(len(years) - 1, 0)) / max(len(years), 1))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(20, 26, title, 18),
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#5f6368"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#5f6368"/>',
    ]

    for tick in range(0, 5):
        value = max_count * tick / 4
        y = top + chart_h - chart_h * tick / 4
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#f1f3f4"/>')
        parts.append(svg_text(left - 8, y + 4, f"{int(value):,}", 10, "end"))

    for idx, (year, count) in enumerate(zip(years, counts)):
        x = left + idx * (bar_w + bar_gap)
        h = chart_h * count / max_count
        y = top + chart_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#188038"/>')
        if idx % 5 == 0 or idx == len(years) - 1:
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{top + chart_h + 18}" '
                f'font-family="Arial, sans-serif" font-size="10" text-anchor="middle" '
                f'fill="#202124" transform="rotate(45 {x + bar_w / 2:.1f} {top + chart_h + 18})">'
                f'{html.escape(year)}</text>'
            )

    unknown = years_counter.get("Unknown", 0)
    if unknown:
        parts.append(svg_text(left, height - 12, f"Unknown year: {unknown:,}", 11))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_source_year_charts(output_dir: Path, stats_by_source: dict[str, SourceStats]) -> dict[str, str]:
    charts_dir = output_dir / "source_year_distribution"
    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_paths: dict[str, str] = {}
    used_names: set[str] = set()

    for stats in sorted(stats_by_source.values(), key=lambda item: item.source.casefold()):
        base = safe_filename(stats.source)
        filename = f"{base}.svg"
        if filename.casefold() in used_names:
            filename = f"{base}_{abs(hash(stats.source)) & 0xffffffff:08x}.svg"
        used_names.add(filename.casefold())
        path = charts_dir / filename
        write_year_chart(path, stats.years, f"{stats.source} year distribution")
        chart_paths[stats.source] = f"{charts_dir.name}/{filename}"

    return chart_paths


def write_markdown_report(
    path: Path,
    stats_by_source: dict[str, SourceStats],
    overall_years: Counter[str],
    doi_chart_name: str,
    completeness_chart_name: str,
    year_chart_name: str,
    source_year_charts: dict[str, str],
    summary_csv_name: str,
) -> None:
    total = sum(stats.total for stats in stats_by_source.values())
    with_doi = sum(stats.with_doi for stats in stats_by_source.values())
    without_doi = sum(stats.without_doi for stats in stats_by_source.values())
    with_abstract = sum(stats.with_abstract for stats in stats_by_source.values())
    without_abstract = sum(stats.without_abstract for stats in stats_by_source.values())
    complete = sum(stats.complete for stats in stats_by_source.values())
    with_doi_without_abstract = sum(stats.with_doi_without_abstract for stats in stats_by_source.values())
    numeric_years = sorted(int(year) for year in overall_years if year.isdigit())

    lines = [
        "# DBLP Split Source Data Report",
        "",
        "## Overview",
        "",
        f"- Sources: {len(stats_by_source):,}",
        f"- Papers: {total:,}",
        f"- With DOI: {with_doi:,} ({fmt_pct(percent(with_doi, total))})",
        f"- Missing DOI: {without_doi:,} ({fmt_pct(percent(without_doi, total))})",
        f"- With abstract: {with_abstract:,} ({fmt_pct(percent(with_abstract, total))})",
        f"- Missing abstract: {without_abstract:,} ({fmt_pct(percent(without_abstract, total))})",
        f"- Complete: {complete:,} ({fmt_pct(percent(complete, total))})",
        f"- With DOI but missing abstract: {with_doi_without_abstract:,} ({fmt_pct(percent(with_doi_without_abstract, total))})",
    ]
    if numeric_years:
        lines.append(f"- Year range: {numeric_years[0]}-{numeric_years[-1]}")
    if overall_years.get("Unknown", 0):
        lines.append(f"- Unknown year: {overall_years['Unknown']:,}")

    lines += [
        "",
        "## DOI Coverage",
        "",
        f"![DOI coverage]({doi_chart_name})",
        "",
        "## Paper Completeness",
        "",
        f"![Paper completeness]({completeness_chart_name})",
        "",
        "## Year Distribution",
        "",
        f"![Year distribution]({year_chart_name})",
        "",
        "## Source Details",
        "",
        f"Machine-readable summary: [{summary_csv_name}]({summary_csv_name})",
        "",
        "| Source | Papers | DBLP sources | Complete | Missing abstract | Missing DOI | Year range |",
        "|---|---:|---|---:|---:|---:|---|",
    ]

    sorted_stats = sorted(stats_by_source.values(), key=lambda item: (-item.total, item.source.casefold()))
    detail_blocks: list[str] = ["", "## Source Year Distributions", ""]

    for stats in sorted_stats:
        dblp_sources = "<br>".join(
            f"{html.escape(source)} ({count:,})"
            for source, count in sorted(stats.dblp_sources.items(), key=lambda item: (-item[1], item[0].casefold()))
        )
        numeric = sorted(int(year) for year in stats.years if year.isdigit())
        year_range = f"{numeric[0]}-{numeric[-1]}" if numeric else "Unknown"
        doi_summary = f"{stats.with_doi:,} ({fmt_pct(percent(stats.with_doi, stats.total))})"
        missing_doi_summary = f"{stats.without_doi:,} ({fmt_pct(percent(stats.without_doi, stats.total))})"
        abstract_summary = f"{stats.with_abstract:,} ({fmt_pct(percent(stats.with_abstract, stats.total))})"
        missing_abstract_summary = (
            f"{stats.without_abstract:,} ({fmt_pct(percent(stats.without_abstract, stats.total))})"
        )
        complete_summary = f"{stats.complete:,} ({fmt_pct(percent(stats.complete, stats.total))})"
        doi_no_abstract_summary = (
            f"{stats.with_doi_without_abstract:,} "
            f"({fmt_pct(percent(stats.with_doi_without_abstract, stats.total))})"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    stats.source,
                    f"{stats.total:,}",
                    dblp_sources,
                    complete_summary,
                    doi_no_abstract_summary,
                    missing_doi_summary,
                    year_range,
                ]
            )
            + " |"
        )

        detail_blocks.extend(
            [
                f"### {stats.source}",
                "",
                f"- Papers: {stats.total:,}",
                f"- DBLP sources: "
                + "; ".join(
                    f"{source} ({count:,})"
                    for source, count in sorted(
                        stats.dblp_sources.items(), key=lambda item: (-item[1], item[0].casefold())
                    )
                ),
                f"- With DOI: {doi_summary}",
                f"- Missing DOI: {missing_doi_summary}",
                f"- With abstract: {abstract_summary}",
                f"- Missing abstract: {missing_abstract_summary}",
                f"- Complete: {complete_summary}",
                f"- With DOI but missing abstract: {doi_no_abstract_summary}",
                f"- Year range: {year_range}",
                "",
                f"![{stats.source} year distribution]({source_year_charts[stats.source]})",
                "",
            ]
        )

    lines.extend(detail_blocks)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DBLP split-source data report.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing source JSON files. Defaults to data/papers/enriched and data/papers/missing.",
    )
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=Path("data/papers"),
        help="Directory containing enriched/ and missing/ paper state directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/report/data_report"),
        help="Directory where this report bundle is written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_dirs = [args.input_dir] if args.input_dir else [args.papers_dir / "enriched", args.papers_dir / "missing"]
    stats_by_source, overall_years = read_stats(input_dirs)
    if not stats_by_source:
        raise SystemExit(
            "No source JSON files found in "
            + ", ".join(str(path) for path in input_dirs)
        )

    doi_chart = args.output_dir / "doi_coverage.svg"
    completeness_chart = args.output_dir / "paper_completeness.svg"
    year_chart = args.output_dir / "year_distribution.svg"
    summary_csv = args.output_dir / "source_summary.csv"
    report_md = args.output_dir / "data_report.md"

    write_doi_chart(doi_chart, stats_by_source)
    write_completeness_chart(completeness_chart, stats_by_source)
    write_year_chart(year_chart, overall_years, "Overall year distribution")
    source_year_charts = write_source_year_charts(args.output_dir, stats_by_source)
    write_summary_csv(summary_csv, stats_by_source)
    write_markdown_report(
        report_md,
        stats_by_source,
        overall_years,
        doi_chart.name,
        completeness_chart.name,
        year_chart.name,
        source_year_charts,
        summary_csv.name,
    )

    print(f"sources: {len(stats_by_source)}")
    print(f"papers: {sum(stats.total for stats in stats_by_source.values())}")
    print(f"report: {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
