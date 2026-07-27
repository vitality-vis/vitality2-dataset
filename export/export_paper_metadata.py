#!/usr/bin/env python3
"""Export all paper metadata from a Zilliz collection (default: paper_new).

Omits heavy / derived fields: embedding, search_text, and search_sparse.
Writes one JSON object per line (JSONL).

Supports timeout retries and --resume to append without overwriting prior rows.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLLECTION = "paper_new"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "zilliz" / "paper_new_metadata.jsonl"
DEFAULT_BATCH_SIZE = 200
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BACKOFF = 2.0
# Zilliz rejects scalar-only load_fields, so load the sparse vector field and
# avoid pulling the dense embedding.
VECTOR_LOAD_FIELD = "search_sparse"
PRIMARY_KEY_FIELD = "paper_uid"
EXCLUDE_FIELDS = frozenset({"embedding", "search_text", "search_sparse"})
RETRYABLE_MARKERS = (
    "timeout",
    "timed out",
    "deadlineexceeded",
    "unavailable",
    "connection reset",
    "connection refused",
    "temporarily unavailable",
    "goaway",
    "broken pipe",
    "statuscode.unavailable",
    "statuscode.deadline_exceeded",
    "rpc error",
)


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def connect_client():
    from pymilvus import MilvusClient

    load_dotenv_file(PROJECT_ROOT / ".env")
    uri = os.environ.get("ZILLIZ_URI")
    token = os.environ.get("ZILLIZ_TOKEN")
    if not uri or not token:
        raise SystemExit("Missing ZILLIZ_URI or ZILLIZ_TOKEN in environment or project .env.")
    return MilvusClient(uri=uri, token=token)


def metadata_fields(client, collection_name: str) -> list[str]:
    desc = client.describe_collection(collection_name)
    names: list[str] = []
    for field in desc.get("fields") or []:
        if isinstance(field, dict):
            name = field.get("name")
        else:
            name = getattr(field, "name", None)
        if name and name not in EXCLUDE_FIELDS:
            names.append(str(name))
    return names


def json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    # pymilvus/protobuf returns array fields as RepeatedScalarContainer
    if type(value).__name__ in {"RepeatedScalarContainer", "RepeatedCompositeContainer"}:
        return list(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def is_retryable_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    if any(marker in message for marker in RETRYABLE_MARKERS):
        return True
    name = type(exc).__name__.lower()
    return any(marker in name for marker in ("timeout", "unavailable", "deadline"))


def load_existing_uids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            uid = str(row.get(PRIMARY_KEY_FIELD) or "").strip()
            if uid:
                seen.add(uid)
    return seen


def merge_tmp_into_output(output: Path, tmp_path: Path) -> int:
    """Append unique tmp rows into output. Returns number of rows merged."""
    if not tmp_path.exists():
        return 0
    existing = load_existing_uids(output)
    merged = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("r", encoding="utf-8") as src, output.open("a", encoding="utf-8") as dest:
        for line in src:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            uid = str(row.get(PRIMARY_KEY_FIELD) or "").strip()
            if not uid or uid in existing:
                continue
            dest.write(text)
            dest.write("\n")
            existing.add(uid)
            merged += 1
    if merged:
        dest_size = output.stat().st_size if output.exists() else 0
        print(f"Merged {merged} unique rows from {tmp_path.name} into output ({dest_size} bytes).", flush=True)
    return merged


def open_query_iterator(client, collection_name: str, fields: list[str], batch_size: int, timeout: float | None):
    return client.query_iterator(
        collection_name=collection_name,
        batch_size=batch_size,
        filter="",
        output_fields=fields,
        timeout=timeout,
    )


def next_batch_with_retry(iterator, *, max_retries: int, retry_backoff: float):
    attempt = 0
    while True:
        try:
            return iterator.next()
        except Exception as exc:
            if not is_retryable_error(exc) or attempt >= max_retries:
                raise
            delay = retry_backoff * (2**attempt)
            attempt += 1
            print(
                f"Query batch failed ({type(exc).__name__}: {exc}); "
                f"retry {attempt}/{max_retries} in {delay:g}s",
                flush=True,
            )
            time.sleep(delay)


def iter_batches(
    client,
    collection_name: str,
    fields: list[str],
    batch_size: int,
    timeout: float | None,
    *,
    max_retries: int,
    retry_backoff: float,
) -> Iterable[list[dict[str, Any]]]:
    """Yield batches with per-batch retries; recreate the iterator after exhausted retries."""
    recreate_attempts = 0
    while True:
        iterator = open_query_iterator(client, collection_name, fields, batch_size, timeout)
        recreate = False
        try:
            while True:
                try:
                    batch = next_batch_with_retry(
                        iterator,
                        max_retries=max_retries,
                        retry_backoff=retry_backoff,
                    )
                except Exception as exc:
                    if not is_retryable_error(exc):
                        raise
                    recreate_attempts += 1
                    delay = retry_backoff * (2 ** min(recreate_attempts - 1, 4))
                    print(
                        f"Iterator failed after retries ({type(exc).__name__}: {exc}); "
                        f"recreating iterator (attempt {recreate_attempts}) in {delay:g}s",
                        flush=True,
                    )
                    time.sleep(delay)
                    recreate = True
                    break
                if not batch:
                    return
                recreate_attempts = 0
                yield batch
        finally:
            try:
                iterator.close()
            except Exception:
                pass
        if not recreate:
            return

def load_for_export(client, fields: list[str], collection_name: str) -> None:
    load_fields = list(dict.fromkeys([PRIMARY_KEY_FIELD, *fields, VECTOR_LOAD_FIELD]))
    print(f"Loading fields for export: {', '.join(load_fields)}", flush=True)
    try:
        client.load_collection(collection_name, load_fields=load_fields)
    except TypeError:
        raise SystemExit(
            "Installed pymilvus does not support load_fields. "
            "Create/load the sparse index first with: "
            f"python3 script/index_zilliz_collection.py --only search_sparse --collection {collection_name}"
        )
    except Exception as exc:
        message = str(exc)
        if "there is no vector index" in message or "does not contain vector field" in message:
            raise SystemExit(
                "Could not load the fields needed for metadata export. "
                "Create/load the sparse index first with: "
                f"python3 script/index_zilliz_collection.py --only search_sparse --collection {collection_name}"
            ) from exc
        raise
    print("Collection load request completed.", flush=True)


def collection_row_count(client, collection_name: str) -> int:
    try:
        stats = client.get_collection_stats(collection_name)
        return int(stats.get("row_count", stats.get("rowCount", 0)) or 0)
    except Exception:
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export paper metadata from Zilliz (all scalar/JSON fields except "
            "embedding, search_text, and search_sparse)."
        )
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--query-timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-backoff", type=float, default=DEFAULT_RETRY_BACKOFF)
    parser.add_argument("--limit", type=int, default=None, help="Optional new-row limit for testing.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Append to an existing output file instead of overwriting. "
            "Skips paper_uids already present (also merges any leftover .tmp file)."
        ),
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help=(
            "Load metadata fields plus search_sparse before querying "
            "(needed if the collection is not already loaded)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.query_timeout <= 0:
        raise SystemExit("--query-timeout must be > 0")
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be >= 0")
    if args.retry_backoff <= 0:
        raise SystemExit("--retry-backoff must be > 0")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    client = connect_client()
    if not client.has_collection(args.collection):
        raise SystemExit(f"Collection does not exist: {args.collection}")

    fields = metadata_fields(client, args.collection)
    if not fields:
        raise SystemExit("No metadata fields left after exclusions.")
    if PRIMARY_KEY_FIELD not in fields:
        raise SystemExit(f"Schema is missing primary key field: {PRIMARY_KEY_FIELD}")

    if args.load:
        load_for_export(client, fields, args.collection)

    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise SystemExit("tqdm is required. Install with: python3 -m pip install tqdm") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(args.output) + ".tmp")

    if args.resume:
        merge_tmp_into_output(args.output, tmp_path)
        if tmp_path.exists():
            tmp_path.unlink()
        seen = load_existing_uids(args.output)
        print(f"Resume mode: {len(seen)} paper_uids already in {args.output}", flush=True)
        file_mode = "a"
        write_path = args.output
    else:
        seen = set()
        file_mode = "w"
        write_path = tmp_path

    total_entities = collection_row_count(client, args.collection)
    expected = total_entities
    if args.limit is not None and not args.resume:
        expected = min(args.limit, total_entities) if total_entities else args.limit
    elif args.resume and total_entities:
        expected = total_entities

    scanned = 0
    written = 0
    skipped = 0
    print(
        f"Exporting metadata from {args.collection} → {args.output} "
        f"(fields={', '.join(fields)}; batch_size={args.batch_size}; "
        f"timeout={args.query_timeout:g}s; max_retries={args.max_retries}; "
        f"~{total_entities} entities; resume={args.resume})",
        flush=True,
    )

    with write_path.open(file_mode, encoding="utf-8") as handle:
        batches = iter_batches(
            client,
            args.collection,
            fields,
            args.batch_size,
            args.query_timeout,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
        )
        # Progress tracks collection scan. Resume may re-scan already-written rows (skipped).
        with tqdm(total=expected or None, unit="paper", desc="export") as bar:
            stop = False
            for batch in batches:
                for row in batch:
                    scanned += 1
                    uid = str(row.get(PRIMARY_KEY_FIELD) or "").strip()
                    if not uid:
                        bar.update(1)
                        continue
                    if uid in seen:
                        skipped += 1
                        bar.update(1)
                        continue

                    record = {field: row.get(field) for field in fields}
                    handle.write(json.dumps(record, ensure_ascii=False, default=json_default))
                    handle.write("\n")
                    seen.add(uid)
                    written += 1
                    bar.update(1)

                    if args.limit is not None and written >= args.limit:
                        stop = True
                        break
                handle.flush()
                if stop:
                    break
    if not args.resume:
        write_path.replace(args.output)
    elif tmp_path.exists():
        # Should not remain in resume mode, but clean up if present.
        tmp_path.unlink()

    manifest = {
        "collection": args.collection,
        "output": str(args.output),
        "fields": fields,
        "excluded_fields": sorted(EXCLUDE_FIELDS),
        "scanned_rows": scanned,
        "written_rows": written,
        "skipped_rows": skipped,
        "total_rows_in_file": len(seen),
        "batch_size": args.batch_size,
        "resume": args.resume,
        "max_retries": args.max_retries,
        "retry_backoff": args.retry_backoff,
    }
    manifest_path = Path(str(args.output) + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
