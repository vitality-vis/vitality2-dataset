#!/usr/bin/env python3
"""Create Zilliz indexes for an existing paper collection and load it."""

from __future__ import annotations

import argparse
import os

from create_zilliz_collection import PROJECT_ROOT, load_dotenv_file


DEFAULT_COLLECTION = "paper_new"
SPARSE_LOAD_FIELDS = [
    "paper_uid",
    "search_text",
    "search_sparse",
    "title",
    "source",
    "dblp_source",
    "year",
    "doi",
    "abstract",
]


def connect_collection(collection_name: str):
    from pymilvus import Collection, connections, utility

    load_dotenv_file(PROJECT_ROOT / ".env")
    uri = os.environ.get("ZILLIZ_URI")
    token = os.environ.get("ZILLIZ_TOKEN")
    if not uri or not token:
        raise SystemExit("Missing ZILLIZ_URI or ZILLIZ_TOKEN in environment or project .env.")

    connections.connect(uri=uri, token=token)
    if not utility.has_collection(collection_name):
        raise SystemExit(f"Collection does not exist: {collection_name}")
    return Collection(collection_name)


def has_index(collection, field_name: str) -> bool:
    return any(index.field_name == field_name for index in collection.indexes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create indexes and load a Zilliz paper collection.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--metric-type", choices=("COSINE", "L2", "IP"), default="COSINE")
    parser.add_argument("--index-type", default="AUTOINDEX")
    parser.add_argument(
        "--only",
        choices=("all", "embedding", "search_sparse"),
        default="all",
        help="Limit index creation to one field.",
    )
    parser.add_argument(
        "--no-load",
        action="store_true",
        help="Do not load the collection after creating indexes.",
    )
    args = parser.parse_args()

    collection = connect_collection(args.collection)
    collection.flush()

    if args.only in {"all", "embedding"} and not has_index(collection, "embedding"):
        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": args.index_type,
                "metric_type": args.metric_type,
                "params": {},
            },
        )
        print(f"Created embedding index: {args.index_type} / {args.metric_type}")
    elif args.only in {"all", "embedding"}:
        print("Embedding index already exists.")

    if args.only in {"all", "search_sparse"} and not has_index(collection, "search_sparse"):
        collection.create_index(
            field_name="search_sparse",
            index_params={
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25",
                "params": {"inverted_index_algo": "DAAT_MAXSCORE"},
            },
        )
        print("Created BM25 index: SPARSE_INVERTED_INDEX / BM25")
    elif args.only in {"all", "search_sparse"}:
        print("BM25 index already exists.")

    if not args.no_load:
        if args.only == "search_sparse":
            collection.load(load_fields=SPARSE_LOAD_FIELDS)
        else:
            collection.load()
        print(f"Loaded collection: {args.collection}")
    print(f"Collection entities: {collection.num_entities}")


if __name__ == "__main__":
    main()
