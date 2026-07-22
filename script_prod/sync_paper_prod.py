#!/usr/bin/env python3
"""
Promote eligible papers from `paper_new` to `paper_prod` with embeddings.

Flow:
  1. Count eligible rows (config eligibility_fields), confirm.
  2. Stream batches: preview classification → embed as needed → upsert.
  3. Backfill any remaining null embeddings; print a final report.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from azure_embeddings import AzureEmbedder  # noqa: E402
from common import (  # noqa: E402
    BATCH_SIZE,
    DEV_COLLECTION,
    DEV_OUTPUT_FIELDS,
    ELIGIBLE_EXPR,
    EMBED_BATCH_SIZE,
    EMBEDDING_FIELDS,
    EMBEDDING_MODEL,
    MISSING_EMBEDDING_EXPR,
    PROD_COLLECTION,
    PROD_LOOKUP_FIELDS,
    SCALAR_FIELDS,
    ask_confirm,
    build_embed_text,
    connect_zilliz,
    count_by_expr,
    embed_input_equal,
    has_collection,
    is_eligible_row,
    list_collections,
    normalize_embed_field,
    uid_in_expr,
    values_equal,
)
from create_paper_prod_collection import ensure_paper_prod_collection  # noqa: E402


@dataclass
class BatchClassified:
    new: list[dict[str, Any]] = field(default_factory=list)
    embed_input_change: list[dict[str, Any]] = field(default_factory=list)
    metadata_only_change: list[dict[str, Any]] = field(default_factory=list)
    unchanged: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunStats:
    new: int = 0
    embed_input_change: int = 0
    metadata_only_change: int = 0
    unchanged: int = 0
    embedding_failures: int = 0
    upserted: int = 0
    backfilled: int = 0
    backfill_failures: int = 0


def metadata_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    for key in SCALAR_FIELDS:
        if key == "paper_uid":
            continue
        if not values_equal(left.get(key), right.get(key)):
            return False
    return True


def classify_batch(
    dev_rows: list[dict[str, Any]],
    prod_by_uid: dict[str, dict[str, Any]],
    *,
    embedding_model: str,
) -> BatchClassified:
    classified = BatchClassified()
    for row in dev_rows:
        uid = str(row.get("paper_uid") or "")
        if not uid:
            continue
        prod = prod_by_uid.get(uid)
        if prod is None:
            classified.new.append(row)
            continue

        model_mismatch = normalize_embed_field(prod.get("embedding_model")) != normalize_embed_field(
            embedding_model
        )
        # Strict: only True means a successful dense vector was recorded.
        missing_embedding = prod.get("has_embedding") is not True
        if (not embed_input_equal(row, prod)) or model_mismatch or missing_embedding:
            classified.embed_input_change.append(row)
            continue

        if metadata_equal(row, prod):
            classified.unchanged.append(row)
        else:
            classified.metadata_only_change.append(row)
    return classified


def example_uids(rows: list[dict[str, Any]], limit: int = 3) -> str:
    uids = [str(row.get("paper_uid") or "") for row in rows[:limit]]
    uids = [uid for uid in uids if uid]
    if not uids:
        return "-"
    return ", ".join(uids)


def print_batch_preview(batch_index: int, classified: BatchClassified, *, progress=None) -> None:
    lines = [
        f"\n--- Batch {batch_index} preview ---",
        f"  new:                  {len(classified.new)}  e.g. {example_uids(classified.new)}",
        (
            f"  embed-input change:   {len(classified.embed_input_change)}  "
            f"e.g. {example_uids(classified.embed_input_change)}"
        ),
        (
            f"  metadata-only change: {len(classified.metadata_only_change)}  "
            f"e.g. {example_uids(classified.metadata_only_change)}"
        ),
        (
            f"  unchanged (skip):     {len(classified.unchanged)}  "
            f"e.g. {example_uids(classified.unchanged)}"
        ),
    ]
    text = "\n".join(lines)
    if progress is not None and hasattr(progress, "write"):
        progress.write(text)
    else:
        print(text, flush=True)


def update_batch_tqdm(progress, classified: BatchClassified, *, failures: int | None = None) -> None:
    if progress is None or not hasattr(progress, "set_postfix"):
        return
    postfix = {
        "new": len(classified.new),
        "embed": len(classified.embed_input_change),
        "meta": len(classified.metadata_only_change),
        "skip": len(classified.unchanged),
    }
    if failures is not None:
        postfix["fail"] = failures
    progress.set_postfix(postfix, refresh=True)


def scalar_entity(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in SCALAR_FIELDS}


def lookup_prod_rows(client, collection_name: str, uids: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch prod scalars (no dense embedding) for classification."""
    if not uids:
        return {}
    by_uid: dict[str, dict[str, Any]] = {}
    for start in range(0, len(uids), BATCH_SIZE):
        chunk = uids[start : start + BATCH_SIZE]
        rows = client.query(
            collection_name=collection_name,
            filter=uid_in_expr(chunk),
            output_fields=PROD_LOOKUP_FIELDS,
            limit=len(chunk) + 10,
        )
        for row in rows:
            uid = row.get("paper_uid")
            if uid:
                by_uid[str(uid)] = row
    return by_uid


def iter_eligible_batches(client, collection_name: str) -> Iterator[list[dict[str, Any]]]:
    iterator = client.query_iterator(
        collection_name=collection_name,
        batch_size=BATCH_SIZE,
        filter=ELIGIBLE_EXPR,
        output_fields=DEV_OUTPUT_FIELDS,
    )
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            rows = [row for row in batch if is_eligible_row(row)]
            if rows:
                yield rows
    finally:
        iterator.close()


def embed_rows(
    embedder: AzureEmbedder,
    rows: list[dict[str, Any]],
    *,
    progress=None,
) -> tuple[dict[str, list[float] | None], int]:
    """Return uid → embedding (or None) and failure count."""
    results: dict[str, list[float] | None] = {}
    failures = 0
    n_requests = (len(rows) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE if rows else 0
    if progress is not None:
        progress.reset(total=n_requests)
        progress.set_description("embed requests")
        if n_requests == 0:
            progress.refresh()

    for start in range(0, len(rows), EMBED_BATCH_SIZE):
        chunk = rows[start : start + EMBED_BATCH_SIZE]
        texts = [build_embed_text(row) for row in chunk]
        vectors = embedder.embed_texts(texts)
        for row, vector in zip(chunk, vectors):
            uid = str(row["paper_uid"])
            results[uid] = vector
            if vector is None:
                failures += 1
        if progress is not None:
            progress.update(1)
    return results, failures


def build_full_upsert_entities(
    rows: list[dict[str, Any]],
    *,
    embeddings_by_uid: dict[str, list[float] | None],
    embedding_model: str,
) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for row in rows:
        entity = scalar_entity(row)
        uid = str(row["paper_uid"])
        vector = embeddings_by_uid.get(uid)
        ok = vector is not None
        entity["embedding"] = vector
        entity["embedding_model"] = embedding_model if ok else None
        entity["has_embedding"] = ok
        entities.append(entity)
    return entities


def build_metadata_partial_entities(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Scalars only — partial_update keeps embedding / has_embedding untouched."""
    return [scalar_entity(row) for row in rows]


def upsert_entities(
    client,
    collection_name: str,
    entities: list[dict[str, Any]],
    *,
    partial_update: bool = False,
) -> None:
    if not entities:
        return
    # Chunk so dense vectors in full upserts stay under the ~4MB gRPC cap.
    chunk_size = 64 if not partial_update else BATCH_SIZE
    for start in range(0, len(entities), chunk_size):
        client.upsert(
            collection_name=collection_name,
            data=entities[start : start + chunk_size],
            partial_update=partial_update,
        )


def iter_missing_embedding_batches(client, collection_name: str) -> Iterator[list[dict[str, Any]]]:
    output_fields = ["paper_uid", "embedding_model", *EMBEDDING_FIELDS]
    # Deduplicate while preserving order.
    seen: set[str] = set()
    ordered_fields: list[str] = []
    for name in output_fields:
        if name not in seen:
            seen.add(name)
            ordered_fields.append(name)
    iterator = client.query_iterator(
        collection_name=collection_name,
        batch_size=BATCH_SIZE,
        filter=MISSING_EMBEDDING_EXPR,
        output_fields=ordered_fields,
    )
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            yield batch
    finally:
        iterator.close()


def backfill_missing_embeddings(
    *,
    client,
    collection_name: str,
    embedder: AzureEmbedder,
    stats: RunStats,
    embed_progress=None,
) -> list[str]:
    """Return paper_uids that still have no embedding after backfill."""
    failed_uids: list[str] = []
    try:
        missing_count = count_by_expr(client, collection_name, MISSING_EMBEDDING_EXPR)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not count missing embeddings via expr ({exc}); scanning instead.", flush=True)
        missing_count = -1

    if missing_count == 0:
        print("No missing embeddings to backfill.", flush=True)
        return []

    print(
        "Backfilling missing embeddings"
        + (f" (count={missing_count})" if missing_count >= 0 else "")
        + "...",
        flush=True,
    )

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    batches: Iterable[list[dict[str, Any]]] = iter_missing_embedding_batches(client, collection_name)
    if tqdm is not None:
        total_batches = (
            (missing_count + BATCH_SIZE - 1) // BATCH_SIZE if missing_count > 0 else None
        )
        batches = tqdm(
            batches,
            desc="backfill batches",
            total=total_batches,
            unit="batch",
            position=0,
            leave=True,
        )

    for batch in batches:
        need = [row for row in batch if row.get("paper_uid") and is_eligible_row(row)]
        if not need:
            need = [
                row
                for row in batch
                if row.get("paper_uid") and all(str(row.get(field) or "").strip() for field in EMBEDDING_FIELDS)
            ]
        if not need:
            continue

        embeddings_by_uid, failures = embed_rows(embedder, need, progress=embed_progress)
        stats.embedding_failures += failures

        partial_rows: list[dict[str, Any]] = []
        for row in need:
            uid = str(row["paper_uid"])
            vector = embeddings_by_uid.get(uid)
            if vector is None:
                failed_uids.append(uid)
                stats.backfill_failures += 1
                continue
            partial_rows.append(
                {
                    "paper_uid": uid,
                    "embedding": vector,
                    "embedding_model": EMBEDDING_MODEL,
                    "has_embedding": True,
                }
            )
            stats.backfilled += 1

        if partial_rows:
            client.upsert(collection_name=collection_name, data=partial_rows, partial_update=True)

    return failed_uids


def main() -> None:
    client = connect_zilliz()

    if not has_collection(DEV_COLLECTION):
        existing = ", ".join(list_collections()) or "(none)"
        raise SystemExit(
            f"Development collection does not exist: {DEV_COLLECTION}\n"
            f"Collections visible with current ZILLIZ_URI/TOKEN: {existing}\n"
            "Check vitality2-dataset/.env points at the cluster that has paper_new "
            "(not leftover placeholders from .env.example)."
        )

    ensure_paper_prod_collection()

    # paper_new often has no dense embedding index; scalar query/iterator still works
    # without load_collection. Only load paper_prod (needed after create / for upserts).
    try:
        client.load_collection(PROD_COLLECTION)
    except Exception as exc:  # noqa: BLE001
        print(f"Note: could not load {PROD_COLLECTION} ({exc})", flush=True)

    eligible_count = count_by_expr(client, DEV_COLLECTION, ELIGIBLE_EXPR)
    print(f"Eligible papers in {DEV_COLLECTION}: {eligible_count}", flush=True)
    if eligible_count == 0:
        print("Nothing to sync.", flush=True)
        return

    decision = ask_confirm("Proceed with batch sync?", yes_all=False)
    if decision == "n":
        print("Aborted.", flush=True)
        return
    yes_all = decision == "all"

    embedder = AzureEmbedder()
    stats = RunStats()

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    batch_iter = iter_eligible_batches(client, DEV_COLLECTION)
    total_batches = (eligible_count + BATCH_SIZE - 1) // BATCH_SIZE
    progress = None
    embed_progress = None
    if tqdm is not None:
        # Two fixed lines: outer batches + inner Azure embed requests (reuse via reset).
        progress = tqdm(
            batch_iter,
            desc="eligible batches",
            total=total_batches,
            unit="batch",
            position=0,
            leave=True,
        )
        embed_progress = tqdm(
            total=0,
            desc="embed requests",
            unit="req",
            position=1,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
    else:
        progress = batch_iter

    try:
        for batch_index, batch in enumerate(progress, start=1):
            uids = [str(row["paper_uid"]) for row in batch if row.get("paper_uid")]
            prod_by_uid = lookup_prod_rows(client, PROD_COLLECTION, uids)
            classified = classify_batch(
                batch,
                prod_by_uid,
                embedding_model=EMBEDDING_MODEL,
            )
            work_count = (
                len(classified.new)
                + len(classified.embed_input_change)
                + len(classified.metadata_only_change)
            )
            if work_count == 0:
                stats.unchanged += len(classified.unchanged)
                if yes_all:
                    update_batch_tqdm(progress, classified)
                continue

            if yes_all:
                update_batch_tqdm(progress, classified)
            else:
                print_batch_preview(batch_index, classified, progress=progress)

            decision = ask_confirm("Upsert this batch?", yes_all=yes_all)
            if decision == "n":
                if progress is not None and hasattr(progress, "write"):
                    progress.write("Skipping remaining batches.")
                else:
                    print("Skipping remaining batches.", flush=True)
                break
            if decision == "all":
                yes_all = True

            to_embed = classified.new + classified.embed_input_change
            embeddings_by_uid, failures = embed_rows(
                embedder,
                to_embed,
                progress=embed_progress,
            )
            stats.embedding_failures += failures
            full_entities = build_full_upsert_entities(
                to_embed,
                embeddings_by_uid=embeddings_by_uid,
                embedding_model=EMBEDDING_MODEL,
            )
            meta_entities = build_metadata_partial_entities(classified.metadata_only_change)
            upsert_entities(client, PROD_COLLECTION, full_entities)
            upsert_entities(client, PROD_COLLECTION, meta_entities, partial_update=True)
            stats.new += len(classified.new)
            stats.embed_input_change += len(classified.embed_input_change)
            stats.metadata_only_change += len(classified.metadata_only_change)
            stats.unchanged += len(classified.unchanged)
            stats.upserted += len(full_entities) + len(meta_entities)
            if yes_all:
                update_batch_tqdm(progress, classified, failures=failures)
            else:
                msg = (
                    f"  Upserted {len(full_entities)} full + {len(meta_entities)} metadata-only "
                    f"(embed failures this batch: {failures})."
                )
                if progress is not None and hasattr(progress, "write"):
                    progress.write(msg)
                else:
                    print(msg, flush=True)
    finally:
        if embed_progress is not None:
            embed_progress.close()
        if progress is not None and hasattr(progress, "close"):
            progress.close()

    client.flush(PROD_COLLECTION)

    # Reuse a single second-line embed bar during backfill.
    backfill_embed = None
    if tqdm is not None:
        backfill_embed = tqdm(
            total=0,
            desc="embed requests",
            unit="req",
            position=1,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
    try:
        failed_after_backfill = backfill_missing_embeddings(
            client=client,
            collection_name=PROD_COLLECTION,
            embedder=embedder,
            stats=stats,
            embed_progress=backfill_embed,
        )
    finally:
        if backfill_embed is not None:
            backfill_embed.close()

    client.flush(PROD_COLLECTION)

    try:
        stats_info = client.get_collection_stats(PROD_COLLECTION)
        prod_size = int(stats_info.get("row_count", stats_info.get("rowCount", -1)))
    except Exception:
        prod_size = -1

    print("\n=== Sync report ===", flush=True)
    print(f"new:                    {stats.new}", flush=True)
    print(f"embed-input changed:    {stats.embed_input_change}", flush=True)
    print(f"metadata-only changed:  {stats.metadata_only_change}", flush=True)
    print(f"unchanged:              {stats.unchanged}", flush=True)
    print(f"upserted:               {stats.upserted}", flush=True)
    print(f"embedding failures:     {stats.embedding_failures}", flush=True)
    print(f"backfilled:             {stats.backfilled}", flush=True)
    print(f"backfill failures:      {stats.backfill_failures}", flush=True)
    print(f"production size:        {prod_size}", flush=True)
    if failed_after_backfill:
        sample = ", ".join(failed_after_backfill[:10])
        more = "" if len(failed_after_backfill) <= 10 else f" (+{len(failed_after_backfill) - 10} more)"
        print(f"still missing embedding: {len(failed_after_backfill)}  e.g. {sample}{more}", flush=True)


if __name__ == "__main__":
    main()
