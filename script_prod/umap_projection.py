#!/usr/bin/env python3
"""Generate and upsert UMAP coordinates for `paper_prod` embeddings.

Full run:
  python3 script_prod/umap_projection.py fit --execute

Incremental run:
  python3 script_prod/umap_projection.py transform --execute

Resume a failed fit upsert after quota/network recovery:
  python3 script_prod/umap_projection.py resume-upsert --execute

Resume a failed overwrite without scanning Zilliz:
  python3 script_prod/umap_projection.py upsert-saved --start-index 89000 --execute
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import BATCH_SIZE, PROD_COLLECTION, connect_zilliz  # noqa: E402

DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "data" / "zilliz" / "umap"
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACT_DIR / "paper_prod_umap.joblib"
DEFAULT_EMBEDDING_CACHE_PATH = DEFAULT_ARTIFACT_DIR / "paper_prod_embeddings.float32.npy"
DEFAULT_TRAINING_PATH = DEFAULT_ARTIFACT_DIR / "paper_prod_umap_training.npz"
DEFAULT_MANIFEST_PATH = DEFAULT_ARTIFACT_DIR / "paper_prod_umap_manifest.json"
DEFAULT_FILTER = "has_embedding == true"
DEFAULT_BATCH_SIZE = BATCH_SIZE
DEFAULT_UPSERT_BATCH_SIZE = 500


@dataclass
class PaperEmbedding:
    paper_uid: str
    embedding_model: str | None = None
    existing_umap: Any = None


@dataclass
class EmbeddingDataset:
    rows: list[PaperEmbedding]
    matrix: Any
    cache_path: Path
    embedding_dim: int


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def collection_row_count(client, collection_name: str, expr: str) -> int | None:
    try:
        rows = client.query(
            collection_name=collection_name,
            filter=expr,
            output_fields=["count(*)"],
        )
        if rows:
            return int(rows[0].get("count(*)") or 0)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not count rows with filter {expr!r}: {exc}", flush=True)
    return None


def iter_embedding_batches(
    client,
    *,
    collection_name: str,
    batch_size: int,
    query_filter: str,
    query_timeout: float | None,
) -> Iterator[list[dict[str, Any]]]:
    iterator = client.query_iterator(
        collection_name=collection_name,
        batch_size=batch_size,
        filter=query_filter,
        output_fields=["paper_uid", "embedding", "embedding_model", "umap"],
        timeout=query_timeout,
    )
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            yield batch
    finally:
        iterator.close()


def collect_embedding_dataset(
    client,
    *,
    collection_name: str,
    batch_size: int,
    query_filter: str,
    query_timeout: float | None,
    skip_existing_umap: bool,
    limit: int | None,
    cache_path: Path,
    embedding_dim: int,
) -> EmbeddingDataset:
    import numpy as np
    from numpy.lib.format import open_memmap

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    total = collection_row_count(client, collection_name, query_filter)
    if total is None and limit is None:
        raise SystemExit("Could not count matching rows; use --limit for a bounded test run.")
    capacity = int(limit if limit is not None else total or 0)
    if capacity < 1:
        return EmbeddingDataset(rows=[], matrix=np.empty((0, embedding_dim), dtype="float32"), cache_path=cache_path, embedding_dim=embedding_dim)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = open_memmap(cache_path, mode="w+", dtype="float32", shape=(capacity, embedding_dim))

    total_batches = (total + batch_size - 1) // batch_size if total else None
    batches: Iterable[list[dict[str, Any]]] = iter_embedding_batches(
        client,
        collection_name=collection_name,
        batch_size=batch_size,
        query_filter=query_filter,
        query_timeout=query_timeout,
    )
    if tqdm is not None:
        batches = tqdm(batches, desc="download embeddings", total=total_batches, unit="batch")

    rows: list[PaperEmbedding] = []
    skipped_existing = 0
    for batch in batches:
        for row in batch:
            uid = str(row.get("paper_uid") or "").strip()
            embedding = row.get("embedding")
            if not uid or embedding is None:
                continue
            if skip_existing_umap and row.get("umap") not in (None, {}, []):
                skipped_existing += 1
                continue
            vector = np.asarray(embedding, dtype="float32")
            if vector.size == 0:
                continue
            if vector.shape[0] != embedding_dim:
                raise SystemExit(
                    f"Embedding dimension mismatch for {uid}: expected {embedding_dim}, got {vector.shape[0]}"
                )
            index = len(rows)
            if index >= capacity:
                raise SystemExit(
                    f"Downloaded more rows than allocated capacity ({capacity}). "
                    "Re-run after checking collection count stability."
                )
            matrix[index, :] = vector
            rows.append(
                PaperEmbedding(
                    paper_uid=uid,
                    embedding_model=(
                        str(row.get("embedding_model"))
                        if row.get("embedding_model") is not None
                        else None
                    ),
                    existing_umap=row.get("umap"),
                )
            )
            if limit is not None and len(rows) >= limit:
                matrix.flush()
                if skipped_existing:
                    print(f"Skipped rows with existing umap: {skipped_existing}", flush=True)
                return EmbeddingDataset(
                    rows=rows,
                    matrix=matrix[: len(rows)],
                    cache_path=cache_path,
                    embedding_dim=embedding_dim,
                )
    matrix.flush()
    if skipped_existing:
        print(f"Skipped rows with existing umap: {skipped_existing}", flush=True)
    return EmbeddingDataset(
        rows=rows,
        matrix=matrix[: len(rows)],
        cache_path=cache_path,
        embedding_dim=embedding_dim,
    )


def build_reducer(args: argparse.Namespace):
    import umap

    return umap.UMAP(
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
        transform_seed=args.transform_seed,
        low_memory=True,
        verbose=args.verbose_umap,
    )


def fit_umap(dataset: EmbeddingDataset, args: argparse.Namespace):
    matrix = dataset.matrix
    reducer = build_reducer(args)
    coords = reducer.fit_transform(matrix)
    return reducer, coords


def transform_umap(dataset: EmbeddingDataset, model_path: Path):
    import joblib

    reducer = joblib.load(model_path)
    coords = reducer.transform(dataset.matrix)
    return reducer, coords


def transform_missing_umap(
    client,
    *,
    collection_name: str,
    query_filter: str,
    batch_size: int,
    query_timeout: float | None,
    model_path: Path,
    cache_path: Path,
    embedding_dim: int,
    upsert_batch_size: int,
    model_version: str | None = None,
    execute: bool = False,
) -> int:
    """Project only matching embedded rows that do not already have UMAP."""
    if not model_path.exists():
        raise SystemExit(f"Missing UMAP model: {model_path}. Run fit first.")
    dataset = collect_embedding_dataset(
        client,
        collection_name=collection_name,
        batch_size=batch_size,
        query_filter=query_filter,
        query_timeout=query_timeout,
        skip_existing_umap=True,
        limit=None,
        cache_path=cache_path,
        embedding_dim=embedding_dim,
    )
    if not dataset.rows:
        print("No missing UMAP rows to transform.", flush=True)
        return 0
    _, coords = transform_umap(dataset, model_path)
    rows = build_upsert_rows(
        dataset.rows,
        coords,
        mode="transform",
        model_version=model_version,
        generated_at=now_iso(),
    )
    print(f"Prepared incremental UMAP rows: {len(rows)}", flush=True)
    if not execute:
        return 0
    return upsert_umap_rows(
        client,
        collection_name=collection_name,
        rows=rows,
        upsert_batch_size=upsert_batch_size,
    )


def transform_umap_uids(
    client,
    *,
    collection_name: str,
    paper_uids: list[str],
    batch_size: int,
    query_timeout: float | None,
    model_path: Path,
    embedding_dim: int,
    upsert_batch_size: int,
    model_version: str | None = None,
    execute: bool = False,
) -> int:
    """Reproject only the supplied production UIDs after embedding changes."""
    import numpy as np

    from common import uid_in_expr

    if not paper_uids:
        print("No newly embedded papers require UMAP.", flush=True)
        return 0
    if not model_path.exists():
        raise SystemExit(f"Missing UMAP model: {model_path}. Run fit first.")

    rows: list[PaperEmbedding] = []
    vectors: list[Any] = []
    for start in range(0, len(paper_uids), batch_size):
        chunk = paper_uids[start : start + batch_size]
        fetched = client.query(
            collection_name=collection_name,
            filter=f"has_embedding == true and ({uid_in_expr(chunk)})",
            output_fields=["paper_uid", "embedding", "embedding_model"],
            limit=len(chunk) + 10,
            timeout=query_timeout,
        )
        for row in fetched:
            uid = str(row.get("paper_uid") or "").strip()
            vector = row.get("embedding")
            if not uid or vector is None:
                continue
            vector_array = np.asarray(vector, dtype="float32")
            if vector_array.shape != (embedding_dim,):
                raise SystemExit(
                    f"Embedding dimension mismatch for {uid}: expected {embedding_dim}, got {vector_array.size}"
                )
            rows.append(PaperEmbedding(uid, str(row.get("embedding_model") or "") or None))
            vectors.append(vector_array)

    if not rows:
        print("No successful embeddings available for UMAP.", flush=True)
        return 0
    dataset = EmbeddingDataset(
        rows=rows,
        matrix=np.stack(vectors),
        cache_path=Path(),
        embedding_dim=embedding_dim,
    )
    _, coords = transform_umap(dataset, model_path)
    upserts = build_upsert_rows(
        dataset.rows,
        coords,
        mode="transform",
        model_version=model_version,
        generated_at=now_iso(),
    )
    print(f"Prepared UMAP updates: {len(upserts)}", flush=True)
    if not execute:
        return 0
    return upsert_umap_rows(
        client,
        collection_name=collection_name,
        rows=upserts,
        upsert_batch_size=upsert_batch_size,
    )


def umap_payload(
    *,
    x: float,
    y: float,
    mode: str,
    model_version: str | None,
    embedding_model: str | None,
    generated_at: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "x": float(x),
        "y": float(y),
        "algorithm": "umap-learn",
        "mode": mode,
        "generated_at": generated_at,
    }
    if model_version:
        payload["model_version"] = model_version
    if embedding_model:
        payload["embedding_model"] = embedding_model
    return payload


def build_upsert_rows(
    rows: list[PaperEmbedding],
    coords,
    *,
    mode: str,
    model_version: str | None,
    generated_at: str,
) -> list[dict[str, Any]]:
    return [
        {
            "paper_uid": row.paper_uid,
            "umap": umap_payload(
                x=float(coords[index][0]),
                y=float(coords[index][1]),
                mode=mode,
                model_version=model_version,
                embedding_model=row.embedding_model,
                generated_at=generated_at,
            ),
        }
        for index, row in enumerate(rows)
    ]


def iter_umap_metadata_batches(
    client,
    *,
    collection_name: str,
    batch_size: int,
    query_filter: str,
    query_timeout: float | None,
) -> Iterator[list[dict[str, Any]]]:
    iterator = client.query_iterator(
        collection_name=collection_name,
        batch_size=batch_size,
        filter=query_filter,
        output_fields=["paper_uid", "embedding_model", "umap"],
        timeout=query_timeout,
    )
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            yield batch
    finally:
        iterator.close()


def build_resume_upsert_rows(
    client,
    *,
    collection_name: str,
    batch_size: int,
    query_filter: str,
    query_timeout: float | None,
    training_path: Path,
    model_version: str | None,
    generated_at: str,
    overwrite_existing_umap: bool,
) -> list[dict[str, Any]]:
    import numpy as np

    if not training_path.exists():
        raise SystemExit(f"Missing saved UMAP coordinates: {training_path}. Run fit first.")

    archive = np.load(training_path, allow_pickle=True)
    paper_uids = [str(uid) for uid in archive["paper_uids"].tolist()]
    coords = archive["umap"]
    coord_by_uid = {uid: coords[index] for index, uid in enumerate(paper_uids)}

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    total = collection_row_count(client, collection_name, query_filter)
    total_batches = (total + batch_size - 1) // batch_size if total else None
    batches: Iterable[list[dict[str, Any]]] = iter_umap_metadata_batches(
        client,
        collection_name=collection_name,
        batch_size=batch_size,
        query_filter=query_filter,
        query_timeout=query_timeout,
    )
    if tqdm is not None:
        batches = tqdm(batches, desc="scan missing umap", total=total_batches, unit="batch")

    rows: list[dict[str, Any]] = []
    for batch in batches:
        for row in batch:
            uid = str(row.get("paper_uid") or "").strip()
            if not uid:
                continue
            if not overwrite_existing_umap and row.get("umap") not in (None, {}, []):
                continue
            coord = coord_by_uid.get(uid)
            if coord is None:
                continue
            embedding_model = (
                str(row.get("embedding_model"))
                if row.get("embedding_model") is not None
                else None
            )
            rows.append(
                {
                    "paper_uid": uid,
                    "umap": umap_payload(
                        x=float(coord[0]),
                        y=float(coord[1]),
                        mode="fit",
                        model_version=model_version,
                        embedding_model=embedding_model,
                        generated_at=generated_at,
                    ),
                }
            )
    return rows


def build_saved_upsert_rows(
    *,
    training_path: Path,
    model_version: str | None,
    generated_at: str,
    start_index: int,
    end_index: int | None,
    compact_umap: bool,
) -> list[dict[str, Any]]:
    import numpy as np

    if not training_path.exists():
        raise SystemExit(f"Missing saved UMAP coordinates: {training_path}. Run fit first.")

    archive = np.load(training_path, allow_pickle=True)
    paper_uids = [str(uid) for uid in archive["paper_uids"].tolist()]
    coords = archive["umap"]
    total = len(paper_uids)
    if start_index < 0:
        raise SystemExit("--start-index must be >= 0")
    if start_index > total:
        raise SystemExit(f"--start-index {start_index} is beyond saved coordinate count {total}")
    stop = total if end_index is None else end_index
    if stop < start_index:
        raise SystemExit("--end-index must be >= --start-index")
    stop = min(stop, total)

    rows: list[dict[str, Any]] = []
    for index in range(start_index, stop):
        x = float(coords[index][0])
        y = float(coords[index][1])
        if compact_umap:
            umap_value: dict[str, Any] = {"x": x, "y": y}
        else:
            umap_value = umap_payload(
                x=x,
                y=y,
                mode="fit",
                model_version=model_version,
                embedding_model=None,
                generated_at=generated_at,
            )
        rows.append({"paper_uid": paper_uids[index], "umap": umap_value})
    return rows


def upsert_umap_rows(
    client,
    *,
    collection_name: str,
    rows: list[dict[str, Any]],
    upsert_batch_size: int,
) -> int:
    if not rows:
        return 0
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    ranges = range(0, len(rows), upsert_batch_size)
    if tqdm is not None:
        ranges = tqdm(ranges, desc="upsert umap", unit="batch")  # type: ignore[assignment]
    upserted = 0
    for start in ranges:
        chunk = rows[start : start + upsert_batch_size]
        client.upsert(collection_name=collection_name, data=chunk, partial_update=True)
        upserted += len(chunk)
    client.flush(collection_name)
    return upserted


def write_artifacts(
    *,
    reducer,
    rows: list[PaperEmbedding],
    coords,
    model_path: Path,
    training_path: Path,
    manifest_path: Path,
    args: argparse.Namespace,
    mode: str,
    generated_at: str,
) -> None:
    import joblib
    import numpy as np
    import umap

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(reducer, model_path)
    np.savez_compressed(
        training_path,
        paper_uids=np.asarray([row.paper_uid for row in rows], dtype=object),
        umap=np.asarray(coords, dtype="float32"),
    )
    manifest = {
        "collection": args.collection,
        "mode": mode,
        "generated_at": generated_at,
        "row_count": len(rows),
        "model_path": str(model_path),
        "embedding_cache_path": str(args.embedding_cache_path),
        "training_path": str(training_path),
        "umap_learn_version": getattr(umap, "__version__", None),
        "params": {
            "n_components": args.n_components,
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": args.metric,
            "random_state": args.random_state,
            "transform_seed": args.transform_seed,
            "embedding_dim": args.embedding_dim,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=json_default) + "\n", encoding="utf-8")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--collection", default=PROD_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--upsert-batch-size", type=int, default=DEFAULT_UPSERT_BATCH_SIZE)
    parser.add_argument("--query-timeout", type=float, default=600.0)
    parser.add_argument("--filter", default=DEFAULT_FILTER, help="Zilliz filter for rows to download.")
    parser.add_argument("--limit", type=int, default=None, help="Testing limit after local filtering.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--embedding-cache-path", type=Path, default=DEFAULT_EMBEDDING_CACHE_PATH)
    parser.add_argument("--training-path", type=Path, default=DEFAULT_TRAINING_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--model-version", default=None, help="Optional label stored in each umap JSON value.")
    parser.add_argument("--embedding-dim", type=int, default=1536)
    parser.add_argument("--execute", action="store_true", help="Actually partial-upsert UMAP JSON values.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit/transform UMAP coordinates from paper_prod embeddings.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Fit a new UMAP model from all selected embeddings.")
    add_common_args(fit_parser)
    fit_parser.add_argument("--n-components", type=int, default=2)
    fit_parser.add_argument("--n-neighbors", type=int, default=30)
    fit_parser.add_argument("--min-dist", type=float, default=0.05)
    fit_parser.add_argument("--metric", default="cosine")
    fit_parser.add_argument("--random-state", type=int, default=42)
    fit_parser.add_argument("--transform-seed", type=int, default=42)
    fit_parser.add_argument("--verbose-umap", action="store_true")

    transform_parser = subparsers.add_parser(
        "transform",
        help="Load an existing UMAP model and project selected embeddings into that fitted space.",
    )
    add_common_args(transform_parser)
    transform_parser.add_argument(
        "--include-existing-umap",
        action="store_true",
        help="Transform all rows matched by --filter instead of locally skipping rows that already have umap.",
    )

    resume_parser = subparsers.add_parser(
        "resume-upsert",
        help="Upsert missing UMAP values from saved fit coordinates without rerunning UMAP.",
    )
    add_common_args(resume_parser)
    resume_parser.add_argument(
        "--overwrite-existing-umap",
        action="store_true",
        help="Overwrite existing UMAP values from saved fit coordinates instead of only filling missing values.",
    )

    saved_parser = subparsers.add_parser(
        "upsert-saved",
        help="Upsert saved fit coordinates by training-file index without scanning Zilliz.",
    )
    add_common_args(saved_parser)
    saved_parser.add_argument("--start-index", type=int, default=0)
    saved_parser.add_argument("--end-index", type=int, default=None)
    saved_parser.add_argument(
        "--compact-umap",
        action="store_true",
        help="Store only {'x', 'y'} in the UMAP JSON to minimize disk usage.",
    )

    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.upsert_batch_size < 1:
        raise SystemExit("--upsert-batch-size must be >= 1")
    if args.query_timeout is not None and args.query_timeout <= 0:
        raise SystemExit("--query-timeout must be > 0")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.embedding_dim < 1:
        raise SystemExit("--embedding-dim must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    if args.command == "transform" and not args.model_path.exists():
        raise SystemExit(f"Missing UMAP model: {args.model_path}. Run fit first.")

    client = connect_zilliz()
    if not client.has_collection(args.collection):
        raise SystemExit(f"Collection does not exist: {args.collection}")

    try:
        client.load_collection(args.collection)
    except Exception as exc:  # noqa: BLE001
        print(f"Note: could not load {args.collection}: {exc}", flush=True)

    if args.command == "resume-upsert":
        upsert_rows = build_resume_upsert_rows(
            client,
            collection_name=args.collection,
            batch_size=args.batch_size,
            query_filter=args.filter,
            query_timeout=args.query_timeout,
            training_path=args.training_path,
            model_version=args.model_version,
            generated_at=now_iso(),
            overwrite_existing_umap=args.overwrite_existing_umap,
        )
        print(f"Prepared missing UMAP rows from saved coordinates: {len(upsert_rows)}", flush=True)
        if not args.execute:
            print("Dry run only. Re-run with --execute to partial-upsert UMAP values.", flush=True)
            return 0
        count = upsert_umap_rows(
            client,
            collection_name=args.collection,
            rows=upsert_rows,
            upsert_batch_size=args.upsert_batch_size,
        )
        print(f"Upserted UMAP rows: {count}", flush=True)
        return 0

    if args.command == "upsert-saved":
        upsert_rows = build_saved_upsert_rows(
            training_path=args.training_path,
            model_version=args.model_version,
            generated_at=now_iso(),
            start_index=args.start_index,
            end_index=args.end_index,
            compact_umap=args.compact_umap,
        )
        print(f"Prepared saved UMAP rows: {len(upsert_rows)}", flush=True)
        if not args.execute:
            print("Dry run only. Re-run with --execute to partial-upsert UMAP values.", flush=True)
            return 0
        count = upsert_umap_rows(
            client,
            collection_name=args.collection,
            rows=upsert_rows,
            upsert_batch_size=args.upsert_batch_size,
        )
        print(f"Upserted UMAP rows: {count}", flush=True)
        return 0

    skip_existing = args.command == "transform" and not args.include_existing_umap
    dataset = collect_embedding_dataset(
        client,
        collection_name=args.collection,
        batch_size=args.batch_size,
        query_filter=args.filter,
        query_timeout=args.query_timeout,
        skip_existing_umap=skip_existing,
        limit=args.limit,
        cache_path=args.embedding_cache_path,
        embedding_dim=args.embedding_dim,
    )
    if not dataset.rows:
        print("No rows to process.", flush=True)
        return 0
    print(
        f"Prepared embedding cache: {dataset.cache_path} "
        f"({len(dataset.rows)} x {dataset.embedding_dim} float32)",
        flush=True,
    )

    generated_at = now_iso()
    if args.command == "fit":
        reducer, coords = fit_umap(dataset, args)
        write_artifacts(
            reducer=reducer,
            rows=dataset.rows,
            coords=coords,
            model_path=args.model_path,
            training_path=args.training_path,
            manifest_path=args.manifest_path,
            args=args,
            mode="fit",
            generated_at=generated_at,
        )
        print(f"Saved UMAP model: {args.model_path}", flush=True)
        print(f"Saved UMAP coordinates: {args.training_path}", flush=True)
        print(f"Saved manifest: {args.manifest_path}", flush=True)
    else:
        reducer, coords = transform_umap(dataset, args.model_path)

    upsert_rows = build_upsert_rows(
        dataset.rows,
        coords,
        mode=args.command,
        model_version=args.model_version,
        generated_at=generated_at,
    )
    print(f"Prepared UMAP rows: {len(upsert_rows)}", flush=True)
    if not args.execute:
        print("Dry run only. Re-run with --execute to partial-upsert UMAP values.", flush=True)
        return 0

    count = upsert_umap_rows(
        client,
        collection_name=args.collection,
        rows=upsert_rows,
        upsert_batch_size=args.upsert_batch_size,
    )
    print(f"Upserted UMAP rows: {count}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
