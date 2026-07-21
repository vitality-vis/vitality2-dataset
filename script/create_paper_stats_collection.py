#!/usr/bin/env python3
"""Create the materialized Zilliz stats collection for dashboard queries."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, TypedDict

try:
    from create_zilliz_collection import PROJECT_ROOT, load_dotenv_file
except ModuleNotFoundError:
    from script.create_zilliz_collection import PROJECT_ROOT, load_dotenv_file


DEFAULT_COLLECTION = "paper_stats"


class FieldPlan(TypedDict, total=False):
    name: str
    dtype: str
    primary: bool
    params: dict[str, object]


def schema_plan() -> list[FieldPlan]:
    return [
        {"name": "stat_key", "dtype": "VARCHAR", "primary": True, "params": {"max_length": 1024}},
        {"name": "stat_type", "dtype": "VARCHAR", "params": {"max_length": 64}},
        {"name": "source_collection", "dtype": "VARCHAR", "params": {"max_length": 256}},
        {"name": "source", "dtype": "VARCHAR", "params": {"max_length": 1024, "nullable": True}},
        {"name": "dblp_source", "dtype": "VARCHAR", "params": {"max_length": 1024, "nullable": True}},
        {"name": "year", "dtype": "INT64", "params": {"nullable": True}},
        {"name": "paper_count", "dtype": "INT64"},
        {"name": "missing_doi_count", "dtype": "INT64", "params": {"nullable": True}},
        {"name": "missing_abstract_count", "dtype": "INT64", "params": {"nullable": True}},
        {"name": "complete_count", "dtype": "INT64", "params": {"nullable": True}},
        {"name": "generated_at", "dtype": "INT64"},
        {"name": "_stats_vector", "dtype": "FLOAT_VECTOR", "params": {"dim": 2}},
    ]


def create_schema():
    from pymilvus import CollectionSchema, DataType, FieldSchema

    fields = [
        FieldSchema(name="stat_key", dtype=DataType.VARCHAR, max_length=1024, is_primary=True),
        FieldSchema(name="stat_type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="source_collection", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024, nullable=True),
        FieldSchema(name="dblp_source", dtype=DataType.VARCHAR, max_length=1024, nullable=True),
        FieldSchema(name="year", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="paper_count", dtype=DataType.INT64),
        FieldSchema(name="missing_doi_count", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="missing_abstract_count", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="complete_count", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="generated_at", dtype=DataType.INT64),
        FieldSchema(name="_stats_vector", dtype=DataType.FLOAT_VECTOR, dim=2),
    ]
    return CollectionSchema(
        fields=fields,
        description="Materialized paper statistics for dashboard queries.",
        enable_dynamic_field=False,
    )


def iter_schema_lines(fields: Iterable[FieldPlan]) -> Iterable[str]:
    for field in fields:
        parts = [field["name"], field["dtype"]]
        if field.get("params"):
            parts.append(str(field["params"]))
        if field.get("primary"):
            parts.append("primary")
        yield " - " + " | ".join(parts)


def create_collection(args: argparse.Namespace) -> None:
    from pymilvus import Collection, connections, utility

    load_dotenv_file(PROJECT_ROOT / ".env")
    uri = os.environ.get("ZILLIZ_URI")
    token = os.environ.get("ZILLIZ_TOKEN")
    if not uri or not token:
        raise SystemExit("Missing ZILLIZ_URI or ZILLIZ_TOKEN.")

    connections.connect(uri=uri, token=token)

    if utility.has_collection(args.collection):
        if args.drop_existing:
            utility.drop_collection(args.collection)
            print(f"Dropped existing collection: {args.collection}")
        else:
            print(f"Collection already exists, unchanged: {args.collection}")
            return

    collection = Collection(name=args.collection, schema=create_schema())
    collection.create_index(
        field_name="_stats_vector",
        index_params={"index_type": "AUTOINDEX", "metric_type": "L2", "params": {}},
    )
    print(f"Created collection: {args.collection}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the paper_stats Zilliz collection.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--drop-existing", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Zilliz paper stats collection plan")
    print(f"Collection: {args.collection}")
    print(f"Drop existing: {args.drop_existing}")
    print("Fields:")
    print("\n".join(iter_schema_lines(schema_plan())))
    if not args.execute:
        print("\nDry run only. Re-run with --execute to create the collection.")
        return
    create_collection(args)


if __name__ == "__main__":
    main()
