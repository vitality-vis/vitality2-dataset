#!/usr/bin/env python3
"""Upload full-paper Markdown chunks and update paper collection chunk mappings."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    from create_zilliz_collection import PROJECT_ROOT
    from upsert_enriched_papers_to_zilliz import connect_client, upsert_batch
    from upload_papers_to_zilliz import DEFAULT_BATCH_SIZE, DEFAULT_COLLECTION, connect_collection, normalize_text
except ModuleNotFoundError:
    from script.create_zilliz_collection import PROJECT_ROOT
    from script.upsert_enriched_papers_to_zilliz import connect_client, upsert_batch
    from script.upload_papers_to_zilliz import DEFAULT_BATCH_SIZE, DEFAULT_COLLECTION, connect_collection, normalize_text


DEFAULT_FULL_COLLECTION = "paper_full"
DEFAULT_PAPER_COLLECTION = "paper_prod"
SCRIPT_PROD = PROJECT_ROOT / "script_prod"
if str(SCRIPT_PROD) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PROD))

from azure_embeddings import AzureEmbedder  # noqa: E402
from common import BATCH_SIZE, EMBEDDING_MODEL, uid_in_expr  # noqa: E402


def upsert_full_batch(client, collection_name: str, batch: list[dict[str, Any]]) -> int:
    if not batch:
        return 0
    client.upsert(collection_name=collection_name, data=batch)
    upserted = len(batch)
    batch.clear()
    return upserted


def embed_chunk_batch(
    embedder: AzureEmbedder | None,
    batch: list[dict[str, Any]],
    *,
    embed_batch_size: int,
    embed_max_chars: int,
) -> int:
    if embedder is None or not batch:
        return 0
    failures = 0
    for start in range(0, len(batch), embed_batch_size):
        group = batch[start : start + embed_batch_size]
        texts = [normalize_text(row.get("text"))[:embed_max_chars] for row in group]
        vectors = embedder.embed_texts(texts)
        for row, vector in zip(group, vectors):
            if vector is None:
                row["embedding"] = None
                row["embedding_model"] = None
                row["has_embedding"] = False
                failures += 1
            else:
                row["embedding"] = vector
                row["embedding_model"] = EMBEDDING_MODEL
                row["has_embedding"] = True
    return failures


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            yield row


def load_existing_paper_uids(client, collection_name: str, results_path: Path) -> set[str]:
    wanted: list[str] = []
    seen: set[str] = set()
    for row in iter_jsonl(results_path):
        if row.get("fullpaper_status") != "found":
            continue
        paper_uid = normalize_text(row.get("paper_uid"))
        if paper_uid and paper_uid not in seen:
            seen.add(paper_uid)
            wanted.append(paper_uid)
    existing: set[str] = set()
    for start in range(0, len(wanted), BATCH_SIZE):
        group = wanted[start : start + BATCH_SIZE]
        rows = client.query(
            collection_name=collection_name,
            filter=uid_in_expr(group),
            output_fields=["paper_uid"],
            limit=len(group) + 10,
        )
        existing.update(normalize_text(row.get("paper_uid")) for row in rows if row.get("paper_uid"))
    return existing


def resolve_path(value: Any) -> Path:
    text = normalize_text(value)
    if not text:
        raise ValueError("Missing fullpaper_md_path")
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def safe_id_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._:-]+", "_", normalize_text(value))
    return safe.strip("._:")[:850] or "unknown"


def utf8_len(value: str) -> int:
    return len(value.encode("utf-8"))


def split_long_text(text: str, max_bytes: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_bytes = 0
    paragraphs = re.split(r"(\n{2,})", text)
    units: list[str] = []
    for index in range(0, len(paragraphs), 2):
        paragraph = paragraphs[index]
        sep = paragraphs[index + 1] if index + 1 < len(paragraphs) else ""
        unit = paragraph + sep
        if unit:
            units.append(unit)
    for unit in units:
        unit_bytes = utf8_len(unit)
        if unit_bytes > max_bytes:
            if current:
                chunks.append("".join(current).strip())
                current = []
                current_bytes = 0
            piece: list[str] = []
            piece_bytes = 0
            for char in unit:
                char_bytes = utf8_len(char)
                if piece and piece_bytes + char_bytes > max_bytes:
                    chunks.append("".join(piece).strip())
                    piece = []
                    piece_bytes = 0
                piece.append(char)
                piece_bytes += char_bytes
            if piece:
                current = piece
                current_bytes = piece_bytes
            continue
        if current and current_bytes + unit_bytes > max_bytes:
            chunks.append("".join(current).strip())
            current = []
            current_bytes = 0
        current.append(unit)
        current_bytes += unit_bytes
    if current:
        chunks.append("".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def chunk_entities(row: dict[str, Any], *, max_text_bytes: int) -> tuple[list[dict[str, Any]], dict[str, str]]:
    paper_uid = normalize_text(row.get("paper_uid"))
    if not paper_uid:
        raise ValueError("Found row is missing paper_uid")
    md_path = resolve_path(row.get("fullpaper_md_path"))
    markdown = md_path.read_text(encoding="utf-8")
    chunks = split_long_text(markdown, max_text_bytes)
    id_prefix = safe_id_part(paper_uid)
    entities: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    for offset, chunk in enumerate(chunks, start=1):
        chunk_id = f"{id_prefix}:{offset:03d}"
        chunk_bytes = utf8_len(chunk)
        if chunk_bytes > max_text_bytes:
            raise ValueError(f"Chunk too large for {paper_uid} #{offset}: {chunk_bytes} bytes")
        entity: dict[str, Any] = {
            "id": chunk_id,
            "paper_uid": paper_uid,
            "chunk_index": offset,
            "text": chunk,
            "_chunk_vector": [0.0, 0.0],
        }
        for source_key, target_key in (
            ("doi", "doi"),
            ("title", "title"),
            ("arxiv_id", "arxiv_id"),
            ("arxiv_method", "arxiv_method"),
        ):
            value = normalize_text(row.get(source_key))
            if value:
                entity[target_key] = value
        entities.append(entity)
        mapping[str(offset)] = chunk_id
    return entities, mapping


def paper_update_entity(row: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    entity: dict[str, Any] = {
        "paper_uid": normalize_text(row.get("paper_uid")),
        "fullpaper_chunks": mapping,
        "fullpaper_chunk_count": len(mapping),
    }
    if row.get("fullpaper_md_bytes") is not None:
        entity["fullpaper_md_bytes"] = int(row["fullpaper_md_bytes"])
    for source_key, target_key in (
        ("fullpaper_status", "fullpaper_status"),
        ("fullpaper_source", "fullpaper_source"),
        ("fullpaper_pdf_url", "fullpaper_pdf_url"),
        ("fullpaper_pdf_path", "fullpaper_pdf_path"),
        ("fullpaper_pdf_bytes", "fullpaper_pdf_bytes"),
        ("fullpaper_pdf_pages", "fullpaper_pdf_pages"),
        ("fullpaper_pdf_extracted_pages", "fullpaper_pdf_extracted_pages"),
        ("arxiv_id", "fullpaper_arxiv_id"),
        ("arxiv_method", "fullpaper_arxiv_method"),
    ):
        value = row.get(source_key)
        if isinstance(value, int):
            entity[target_key] = value
            continue
        text = normalize_text(value)
        if text:
            entity[target_key] = text
    return entity


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload fullpaper Markdown chunks to Zilliz.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--full-collection", default=DEFAULT_FULL_COLLECTION)
    parser.add_argument("--paper-collection", default=DEFAULT_PAPER_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--paper-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-text-bytes", type=int, default=60000)
    parser.add_argument("--embed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--embed-batch-size", type=int, default=100)
    parser.add_argument(
        "--embed-max-chars",
        type=int,
        default=6000,
        help="Maximum leading characters from each full-paper chunk sent to the embedding API.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-existing-paper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1 or args.paper_batch_size < 1:
        raise SystemExit("batch sizes must be >= 1")
    if not 1 <= args.max_text_bytes <= 65535:
        raise SystemExit("--max-text-bytes must be between 1 and 65535")
    if args.embed_batch_size < 1:
        raise SystemExit("--embed-batch-size must be >= 1")
    if args.embed_max_chars < 1:
        raise SystemExit("--embed-max-chars must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    connect_collection(args.full_collection)
    connect_collection(args.paper_collection)
    client = None if args.dry_run else connect_client()
    if client is not None:
        client.load_collection(args.full_collection)
    embedder = None if args.dry_run or not args.embed else AzureEmbedder()
    existing_paper_uids = (
        load_existing_paper_uids(client, args.paper_collection, args.results)
        if client is not None and args.require_existing_paper
        else None
    )

    scanned = 0
    found_rows = 0
    chunk_count = 0
    max_chunk_bytes = 0
    chunk_batch: list[dict[str, Any]] = []
    paper_batch: list[dict[str, Any]] = []
    uploaded_chunks = 0
    updated_papers = 0
    embedding_failures = 0
    skipped_missing_paper = 0

    for row in iter_jsonl(args.results):
        scanned += 1
        if row.get("fullpaper_status") != "found":
            continue
        found_rows += 1
        if args.limit is not None and found_rows > args.limit:
            break
        paper_uid = normalize_text(row.get("paper_uid"))
        if existing_paper_uids is not None and paper_uid not in existing_paper_uids:
            skipped_missing_paper += 1
            continue
        chunks, mapping = chunk_entities(row, max_text_bytes=args.max_text_bytes)
        chunk_count += len(chunks)
        for chunk in chunks:
            max_chunk_bytes = max(max_chunk_bytes, utf8_len(chunk["text"]))
        if args.dry_run:
            continue
        chunk_batch.extend(chunks)
        paper_batch.append(paper_update_entity(row, mapping))
        if len(chunk_batch) >= args.batch_size:
            embedding_failures += embed_chunk_batch(
                embedder,
                chunk_batch,
                embed_batch_size=args.embed_batch_size,
                embed_max_chars=args.embed_max_chars,
            )
            uploaded_chunks += upsert_full_batch(client, args.full_collection, chunk_batch)
            print(f"Uploaded chunks: {uploaded_chunks}; found papers: {found_rows}", flush=True)
        if len(paper_batch) >= args.paper_batch_size:
            updated_papers += upsert_batch(client, args.paper_collection, paper_batch)
            print(f"Updated papers: {updated_papers}; found papers: {found_rows}", flush=True)

    if not args.dry_run:
        embedding_failures += embed_chunk_batch(
            embedder,
            chunk_batch,
            embed_batch_size=args.embed_batch_size,
            embed_max_chars=args.embed_max_chars,
        )
        uploaded_chunks += upsert_full_batch(client, args.full_collection, chunk_batch)
        updated_papers += upsert_batch(client, args.paper_collection, paper_batch)

    summary = {
        "dry_run": args.dry_run,
        "results": str(args.results),
        "scanned_rows": scanned,
        "found_rows": found_rows if args.limit is None else min(found_rows, args.limit),
        "chunk_count": chunk_count,
        "max_chunk_bytes": max_chunk_bytes,
        "uploaded_chunks": uploaded_chunks,
        "updated_papers": updated_papers,
        "embedding_failures": embedding_failures,
        "skipped_missing_paper": skipped_missing_paper,
        "embed": args.embed,
        "embed_batch_size": args.embed_batch_size,
        "embed_max_chars": args.embed_max_chars,
        "full_collection": args.full_collection,
        "paper_collection": args.paper_collection,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
