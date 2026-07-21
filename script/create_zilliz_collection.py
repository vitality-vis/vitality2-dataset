#!/usr/bin/env python3
"""
Create the Zilliz collection schema for the current Vitality2 paper dataset.

This script does not upload paper rows or embeddings. By default it runs in
dry-run mode and prints the schema plan. Pass --execute to connect to Zilliz
Cloud and create the collection.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, TypedDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLLECTION = "paper_new"
DEFAULT_EMBEDDING_DIM = 1536


class FieldPlan(TypedDict, total=False):
    name: str
    dtype: str
    primary: bool
    params: dict[str, object]


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


def schema_plan(embedding_dim: int) -> list[FieldPlan]:
    return [
        {"name": "paper_uid", "dtype": "VARCHAR", "primary": True, "params": {"max_length": 1024}},
        {"name": "dblp_key", "dtype": "VARCHAR", "params": {"max_length": 1024, "nullable": True}},
        {"name": "doi", "dtype": "VARCHAR", "params": {"max_length": 512, "nullable": True}},
        {
            "name": "embedding",
            "dtype": "FLOAT_VECTOR",
            "params": {"dim": embedding_dim, "nullable": True},
        },
        {"name": "umap", "dtype": "JSON", "params": {"nullable": True}},
        {
            "name": "search_text",
            "dtype": "VARCHAR",
            "params": {
                "max_length": 65535,
                "nullable": True,
                "enable_analyzer": True,
                "enable_match": True,
            },
        },
        {"name": "search_sparse", "dtype": "SPARSE_FLOAT_VECTOR", "params": {"function": "search_text_bm25"}},
        {"name": "title", "dtype": "VARCHAR", "params": {"max_length": 4096}},
        {
            "name": "abstract",
            "dtype": "VARCHAR",
            "params": {"max_length": 65535, "nullable": True},
        },
        {
            "name": "authors",
            "dtype": "ARRAY<VARCHAR>",
            "params": {"max_capacity": 256, "max_length": 512},
        },
        {
            "name": "keywords",
            "dtype": "ARRAY<VARCHAR>",
            "params": {"max_capacity": 256, "max_length": 512, "nullable": True},
        },
        {"name": "source", "dtype": "VARCHAR", "params": {"max_length": 1024}},
        {"name": "dblp_source", "dtype": "VARCHAR", "params": {"max_length": 1024}},
        {"name": "year", "dtype": "INT64"},
        {"name": "citationCounts", "dtype": "INT64", "params": {"nullable": True}},
        {"name": "fullpaper", "dtype": "BOOL"},
    ]


def create_schema(embedding_dim: int):
    from pymilvus import CollectionSchema, DataType, FieldSchema, Function, FunctionType

    fields = [
        FieldSchema(name="paper_uid", dtype=DataType.VARCHAR, max_length=1024, is_primary=True),
        FieldSchema(name="dblp_key", dtype=DataType.VARCHAR, max_length=1024, nullable=True),
        FieldSchema(name="doi", dtype=DataType.VARCHAR, max_length=512, nullable=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim, nullable=True),
        FieldSchema(name="umap", dtype=DataType.JSON, nullable=True),
        FieldSchema(
            name="search_text",
            dtype=DataType.VARCHAR,
            max_length=65535,
            nullable=True,
            enable_analyzer=True,
            enable_match=True,
        ),
        FieldSchema(name="search_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="abstract", dtype=DataType.VARCHAR, max_length=65535, nullable=True),
        FieldSchema(
            name="authors",
            dtype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=256,
            max_length=512,
        ),
        FieldSchema(
            name="keywords",
            dtype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=256,
            max_length=512,
            nullable=True,
        ),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="dblp_source", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="year", dtype=DataType.INT64),
        FieldSchema(name="citationCounts", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="fullpaper", dtype=DataType.BOOL),
    ]
    bm25_function = Function(
        name="search_text_bm25",
        input_field_names=["search_text"],
        output_field_names=["search_sparse"],
        function_type=FunctionType.BM25,
    )
    return CollectionSchema(
        fields=fields,
        functions=[bm25_function],
        description="Vitality2 papers with dense embeddings plus BM25 full-text search over search_text.",
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
    from pymilvus import Collection, connections, db, utility

    load_dotenv_file(PROJECT_ROOT / ".env")
    uri = os.environ.get("ZILLIZ_URI")
    token = os.environ.get("ZILLIZ_TOKEN")
    if not uri or not token:
        raise SystemExit(
            "Missing ZILLIZ_URI or ZILLIZ_TOKEN. Add them to environment variables "
            "or to the project .env file before running with --execute."
        )

    connections.connect(uri=uri, token=token)

    if args.database:
        if args.create_database:
            existing = {item.name for item in db.list_database()}
            if args.database not in existing:
                db.create_database(args.database)
        db.using_database(args.database)

    if utility.has_collection(args.collection):
        if args.drop_existing:
            utility.drop_collection(args.collection)
            print(f"Dropped existing collection: {args.collection}")
        elif not args.keep_existing:
            raise SystemExit(
                f"Collection already exists: {args.collection}. "
                "Use --keep-existing to leave it unchanged or --drop-existing to recreate it."
            )
        else:
            print(f"Collection already exists, unchanged: {args.collection}")
            return

    schema = create_schema(args.embedding_dim)
    collection = Collection(name=args.collection, schema=schema)

    if not args.defer_index:
        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": args.index_type,
                "metric_type": args.metric_type,
                "params": {},
            },
        )
        collection.create_index(
            field_name="search_sparse",
            index_params={
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25",
                "params": {
                    "inverted_index_algo": "DAAT_MAXSCORE",
                },
            },
        )

    if args.load and not args.defer_index:
        collection.load()

    print(f"Created collection: {args.collection}")
    if args.defer_index:
        print("Indexes deferred. Run script/index_zilliz_collection.py after uploading.")
    else:
        print(f"Embedding index: {args.index_type} / {args.metric_type}")
        print("BM25 index: SPARSE_INVERTED_INDEX / BM25")
    if args.load and not args.defer_index:
        print("Collection loaded.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the Vitality2 Zilliz collection schema without uploading data."
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--database", default=None, help="Optional Zilliz database name.")
    parser.add_argument(
        "--create-database",
        action="store_true",
        help="Create --database first if it does not exist. Dedicated clusters only.",
    )
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--metric-type", choices=("COSINE", "L2", "IP"), default="COSINE")
    parser.add_argument("--index-type", default="AUTOINDEX")
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load the collection after creating the embedding index.",
    )
    parser.add_argument(
        "--defer-index",
        action="store_true",
        help="Create only the schema. Build indexes later after bulk upload.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not fail if the collection already exists.",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop and recreate the collection if it already exists.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually connect to Zilliz and create the collection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fields = schema_plan(args.embedding_dim)

    print("Zilliz collection plan")
    print(f"Collection: {args.collection}")
    print(f"Database: {args.database or '(current/default)'}")
    print(f"Create database: {args.create_database}")
    print(f"Drop existing: {args.drop_existing}")
    print(f"Embedding index: {args.index_type} / {args.metric_type}")
    print("Fields:")
    print("\n".join(iter_schema_lines(fields)))

    if not args.execute:
        print("\nDry run only. Re-run with --execute to create the collection.")
        return

    create_collection(args)


if __name__ == "__main__":
    main()
