#!/usr/bin/env python3
"""Create the paper_full collection for Markdown full-paper chunks."""

from __future__ import annotations

import argparse

try:
    from create_zilliz_collection import PROJECT_ROOT, load_dotenv_file
except ModuleNotFoundError:
    from script.create_zilliz_collection import PROJECT_ROOT, load_dotenv_file


DEFAULT_COLLECTION = "paper_full"


def create_collection(args: argparse.Namespace) -> None:
    import os

    from pymilvus import CollectionSchema, DataType, FieldSchema, Function, FunctionType, MilvusClient

    load_dotenv_file(PROJECT_ROOT / ".env")
    uri = os.environ.get("ZILLIZ_URI")
    token = os.environ.get("ZILLIZ_TOKEN")
    if not uri or not token:
        raise SystemExit("Missing ZILLIZ_URI or ZILLIZ_TOKEN in environment or project .env.")

    client = MilvusClient(uri=uri, token=token)
    if client.has_collection(args.collection):
        if args.drop_existing:
            client.drop_collection(args.collection, timeout=args.timeout)
            print(f"Dropped existing collection: {args.collection}")
        elif args.keep_existing:
            print(f"Collection already exists, unchanged: {args.collection}")
            return
        else:
            raise SystemExit(
                f"Collection already exists: {args.collection}. "
                "Use --keep-existing or --drop-existing."
            )

    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=1024, is_primary=True),
            FieldSchema(name="paper_uid", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=args.text_max_length,
                enable_analyzer=args.enable_bm25,
                enable_match=args.enable_bm25,
            ),
            FieldSchema(name="search_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="_chunk_vector", dtype=DataType.FLOAT_VECTOR, dim=2),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=args.embedding_dim, nullable=True),
        ],
        functions=[
            Function(
                name="text_bm25",
                input_field_names=["text"],
                output_field_names=["search_sparse"],
                function_type=FunctionType.BM25,
            )
        ]
        if args.enable_bm25
        else [],
        description="Vitality2 full-paper Markdown chunks.",
        enable_dynamic_field=True,
    )
    client.create_collection(collection_name=args.collection, schema=schema, timeout=args.timeout)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="_chunk_vector", index_type="AUTOINDEX", metric_type="COSINE")
    index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
    if args.enable_bm25:
        index_params.add_index(
            field_name="search_sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"drop_ratio_build": 0.2},
        )
    client.create_index(collection_name=args.collection, index_params=index_params, timeout=args.timeout)
    print(f"Created collection: {args.collection}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper_full chunk collection.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--text-max-length", type=int, default=65535)
    parser.add_argument("--embedding-dim", type=int, default=1536)
    parser.add_argument("--enable-bm25", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--drop-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.text_max_length <= 65535:
        raise SystemExit("--text-max-length must be between 1 and 65535")
    if args.embedding_dim < 1:
        raise SystemExit("--embedding-dim must be >= 1")
    if args.keep_existing and args.drop_existing:
        raise SystemExit("--keep-existing cannot be combined with --drop-existing")
    print(
        {
            "collection": args.collection,
            "fields": [
                "id VARCHAR(1024) primary",
                "paper_uid VARCHAR(1024)",
                "chunk_index INT64",
                f"text VARCHAR({args.text_max_length}, analyzer={args.enable_bm25})",
                "search_sparse SPARSE_FLOAT_VECTOR (BM25 output)" if args.enable_bm25 else None,
                "_chunk_vector FLOAT_VECTOR(2)",
                f"embedding FLOAT_VECTOR({args.embedding_dim}) nullable",
            ],
            "functions": ["text_bm25: text -> search_sparse"] if args.enable_bm25 else [],
            "enable_dynamic_field": True,
            "execute": args.execute,
        }
    )
    if args.execute:
        create_collection(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
