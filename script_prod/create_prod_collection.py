"""
Schema and create helper for the production Zilliz collection `paper_prod`.

Used by sync.py — not a standalone CLI entry point.
Schema matches `paper_new`, plus:
  - `embedding_model` — logical embedding model id from config.toml
  - `has_embedding` — BOOL set strictly when a dense vector was written successfully
    (Milvus cannot filter FLOAT_VECTOR with `is null`, so this is the searchable marker)
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    EMBEDDING_DIM,
    PROD_COLLECTION,
    connect_zilliz,
)


def create_schema(embedding_dim: int = EMBEDDING_DIM):
    from pymilvus import DataType, Function, FunctionType, MilvusClient

    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
        description="Vitality2 production papers with dense embeddings and BM25 search.",
    )
    schema.add_field(field_name="paper_uid", datatype=DataType.VARCHAR, max_length=1024, is_primary=True)
    schema.add_field(field_name="dblp_key", datatype=DataType.VARCHAR, max_length=1024, nullable=True)
    schema.add_field(field_name="doi", datatype=DataType.VARCHAR, max_length=512, nullable=True)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=embedding_dim, nullable=True)
    schema.add_field(field_name="embedding_model", datatype=DataType.VARCHAR, max_length=256, nullable=True)
    schema.add_field(
        field_name="has_embedding",
        datatype=DataType.BOOL,
        nullable=False,
        default_value=False,
    )
    schema.add_field(field_name="umap", datatype=DataType.JSON, nullable=True)
    schema.add_field(
        field_name="search_text",
        datatype=DataType.VARCHAR,
        max_length=65535,
        nullable=True,
        enable_analyzer=True,
        enable_match=True,
    )
    schema.add_field(field_name="search_sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=4096)
    schema.add_field(field_name="abstract", datatype=DataType.VARCHAR, max_length=65535, nullable=True)
    schema.add_field(
        field_name="authors",
        datatype=DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=256,
        max_length=512,
    )
    schema.add_field(
        field_name="keywords",
        datatype=DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=256,
        max_length=512,
        nullable=True,
    )
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="dblp_source", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="year", datatype=DataType.INT64)
    schema.add_field(field_name="citation_count", datatype=DataType.INT64, nullable=True)
    schema.add_field(field_name="full_paper", datatype=DataType.BOOL)

    schema.add_function(
        Function(
            name="search_text_bm25",
            input_field_names=["search_text"],
            output_field_names=["search_sparse"],
            function_type=FunctionType.BM25,
        )
    )
    return schema


def _collection_field_names(client, collection_name: str) -> set[str]:
    desc = client.describe_collection(collection_name)
    fields = desc.get("fields") or []
    names: set[str] = set()
    for field in fields:
        if isinstance(field, dict):
            name = field.get("name")
        else:
            name = getattr(field, "name", None)
        if name:
            names.add(str(name))
    return names


def ensure_has_embedding_field(client=None) -> bool:
    """Add has_embedding to an existing collection if missing. Returns True if added."""
    from pymilvus import DataType

    client = client or connect_zilliz()
    if "has_embedding" in _collection_field_names(client, PROD_COLLECTION):
        return False

    client.add_collection_field(
        collection_name=PROD_COLLECTION,
        field_name="has_embedding",
        data_type=DataType.BOOL,
        desc="True iff dense embedding was successfully written",
        nullable=True,
        default_value=False,
    )
    print(f"Added field has_embedding to existing collection: {PROD_COLLECTION}", flush=True)
    return True


def ensure_prod_collection() -> bool:
    """Create paper_prod if needed; ensure has_embedding exists. Returns True if created."""
    client = connect_zilliz()

    if client.has_collection(PROD_COLLECTION):
        print(f"Using existing production collection: {PROD_COLLECTION}", flush=True)
        ensure_has_embedding_field(client)
        return False

    schema = create_schema()
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    index_params.add_index(
        field_name="search_sparse",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )
    client.create_collection(
        collection_name=PROD_COLLECTION,
        schema=schema,
        index_params=index_params,
    )
    client.load_collection(PROD_COLLECTION)
    print(f"Created collection: {PROD_COLLECTION}", flush=True)
    return True
