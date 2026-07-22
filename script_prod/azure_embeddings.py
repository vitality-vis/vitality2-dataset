"""Azure OpenAI embedding helpers for production sync."""

from __future__ import annotations

import os
import time
from typing import Sequence

from common import PROJECT_ROOT, load_dotenv_file


class AzureEmbedder:
    """Calls Azure with AZURE_OPENAI_EMBED_DEPLOYMENT (deployment name, not model id)."""

    def __init__(
        self,
        *,
        deployment: str | None = None,
        api_version: str | None = None,
        max_retries: int = 5,
        retry_backoff: float = 2.0,
    ) -> None:
        load_dotenv_file(PROJECT_ROOT / ".env")
        endpoint = (os.environ.get("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY") or ""
        self.deployment = (deployment or os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT") or "").strip()
        self.api_version = (
            api_version
            or os.environ.get("AZURE_OPENAI_EMBED_API_VERSION")
            or "2024-02-01"
        )
        if not endpoint or not api_key:
            raise SystemExit(
                "Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY in environment or .env."
            )
        if not self.deployment:
            raise SystemExit(
                "Missing AZURE_OPENAI_EMBED_DEPLOYMENT in environment or .env "
                "(Azure deployment name; not the same as config embedding_model)."
            )

        from openai import AzureOpenAI

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=self.api_version,
        )
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def embed_texts(self, texts: Sequence[str]) -> list[list[float] | None]:
        """Embed texts; failed items become None (caller may still upsert)."""
        if not texts:
            return []

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.deployment,
                    input=list(texts),
                )
                by_index = {item.index: item.embedding for item in response.data}
                return [by_index.get(i) for i in range(len(texts))]
            except Exception as exc:  # noqa: BLE001 - surface and retry API errors
                last_error = exc
                sleep_for = self.retry_backoff * (2**attempt)
                print(
                    f"Embedding batch failed (attempt {attempt + 1}/{self.max_retries}): {exc}. "
                    f"Retrying in {sleep_for:.1f}s...",
                    flush=True,
                )
                time.sleep(sleep_for)

        print(f"Embedding batch failed permanently: {last_error}", flush=True)
        return [None] * len(texts)
