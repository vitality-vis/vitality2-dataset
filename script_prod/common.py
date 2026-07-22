"""Shared helpers for production paper sync scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = SCRIPT_DIR / "config.toml"


def _load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Missing config file: {CONFIG_PATH}")
    with CONFIG_PATH.open("rb") as handle:
        data = tomllib.load(handle)
    required = {
        "dev_collection",
        "prod_collection",
        "embedding_dim",
        "embedding_model",
        "embedding_fields",
        "eligibility_fields",
        "batch_size",
        "embed_batch_size",
    }
    missing = sorted(required - set(data))
    if missing:
        raise SystemExit(f"config.toml missing keys: {', '.join(missing)}")

    for key in ("embedding_fields", "eligibility_fields"):
        value = data[key]
        if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
            raise SystemExit(f"config.toml {key} must be a non-empty list of field names")
    return data


_CONFIG = _load_config()

DEV_COLLECTION = str(_CONFIG["dev_collection"])
PROD_COLLECTION = str(_CONFIG["prod_collection"])
EMBEDDING_DIM = int(_CONFIG["embedding_dim"])
EMBEDDING_MODEL = str(_CONFIG["embedding_model"])
EMBEDDING_FIELDS = [str(name) for name in _CONFIG["embedding_fields"]]
ELIGIBILITY_FIELDS = [str(name) for name in _CONFIG["eligibility_fields"]]
BATCH_SIZE = int(_CONFIG["batch_size"])
EMBED_BATCH_SIZE = int(_CONFIG["embed_batch_size"])

# Scalar fields copied from development → production (excluding vectors / derived fields).
SCALAR_FIELDS = [
    "paper_uid",
    "dblp_key",
    "doi",
    "umap",
    "search_text",
    "title",
    "abstract",
    "authors",
    "keywords",
    "source",
    "dblp_source",
    "year",
    "citation_count",
    "full_paper",
]

DEV_OUTPUT_FIELDS = list(SCALAR_FIELDS)
# Prod-only derived flags (not copied from paper_new).
# has_embedding mirrors whether a dense vector was successfully written — Milvus
# cannot filter FLOAT_VECTOR with `is null`, so this BOOL is the searchable marker.
PROD_LOOKUP_FIELDS = [*SCALAR_FIELDS, "embedding_model", "has_embedding"]

# Eligible papers for production promotion.
ELIGIBLE_EXPR = " and ".join(f'{field} != ""' for field in ELIGIBILITY_FIELDS)
# Catch false and null (e.g. rows written before the field existed).
MISSING_EMBEDDING_EXPR = "has_embedding != true"


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


_CLIENT = None


def connect_zilliz():
    """Return a shared MilvusClient (creates one on first call)."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    from pymilvus import MilvusClient

    load_dotenv_file(PROJECT_ROOT / ".env")
    uri = os.environ.get("ZILLIZ_URI")
    token = os.environ.get("ZILLIZ_TOKEN")
    if not uri or not token:
        raise SystemExit("Missing ZILLIZ_URI or ZILLIZ_TOKEN in environment or project .env.")
    _CLIENT = MilvusClient(uri=uri, token=token)
    return _CLIENT


def has_collection(name: str) -> bool:
    return bool(connect_zilliz().has_collection(name))


def list_collections() -> list[str]:
    return sorted(str(item) for item in connect_zilliz().list_collections())


def escape_str(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def quote_uid(uid: str) -> str:
    return f'"{escape_str(uid)}"'


def uid_in_expr(uids: list[str]) -> str:
    if not uids:
        raise ValueError("uid_in_expr requires at least one uid")
    return f"paper_uid in [{', '.join(quote_uid(uid) for uid in uids)}]"


def field_nonempty(row: dict[str, Any], field: str) -> bool:
    return bool(str(row.get(field) or "").strip())


def is_eligible_row(row: dict[str, Any]) -> bool:
    return all(field_nonempty(row, field) for field in ELIGIBILITY_FIELDS)


def embed_input_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        normalize_embed_field(left.get(field)) == normalize_embed_field(right.get(field))
        for field in EMBEDDING_FIELDS
    )


def build_embed_text(row: dict[str, Any]) -> str:
    parts = [str(row.get(field) or "").strip() for field in EMBEDDING_FIELDS]
    return "\n".join(part for part in parts if part).strip()


def normalize_embed_field(value: Any) -> str:
    return str(value or "").strip().lower()


def values_equal(left: Any, right: Any) -> bool:
    if left is None and right is None:
        return True
    if isinstance(left, list) or isinstance(right, list):
        left_list = list(left or [])
        right_list = list(right or [])
        return left_list == right_list
    return left == right


def embedding_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    try:
        return len(value) > 0
    except TypeError:
        return True


def ask_confirm(prompt: str, *, yes_all: bool) -> str:
    """Return 'y', 'n', or 'all'."""
    if yes_all:
        return "all"
    while True:
        answer = input(f"{prompt} [y/n/yes for all]: ").strip().lower()
        if answer in {"y", "yes"}:
            return "y"
        if answer in {"n", "no"}:
            return "n"
        if answer in {"a", "all", "yes for all", "yes-for-all"}:
            return "all"
        print("Please answer y, n, or yes for all.")


def count_by_expr(client, collection_name: str, expr: str) -> int:
    rows = client.query(
        collection_name=collection_name,
        filter=expr,
        output_fields=["count(*)"],
    )
    if not rows:
        return 0
    value = rows[0].get("count(*)")
    return int(value or 0)
