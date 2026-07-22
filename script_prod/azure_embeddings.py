"""Azure OpenAI embedding helpers for production sync."""

from __future__ import annotations

import os
import re
import time
from typing import Sequence

from common import EMBED_TIMEOUT_SECONDS, PROJECT_ROOT, load_dotenv_file

_RETRY_AFTER_RE = re.compile(r"retry after\s+(\d+(?:\.\d+)?)\s+seconds?", re.IGNORECASE)


class AzureEmbedder:
    """Calls Azure with AZURE_OPENAI_EMBED_DEPLOYMENT (deployment name, not model id)."""

    def __init__(
        self,
        *,
        deployment: str | None = None,
        api_version: str | None = None,
        timeout: float | None = None,
        max_retries: int = 5,
        retry_backoff: float = 2.0,
        timeout_retry_wait: float = 60.0,
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

        self.timeout = float(EMBED_TIMEOUT_SECONDS if timeout is None else timeout)
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=self.api_version,
            timeout=self.timeout,
            max_retries=0,  # we handle retries ourselves in embed_texts
        )
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.timeout_retry_wait = timeout_retry_wait

    def _retry_wait_seconds(self, exc: BaseException, attempt: int) -> float:
        """Prefer Azure's 'retry after N seconds'; timeouts wait 60s; else exponential."""
        from openai import APITimeoutError, RateLimitError

        match = _RETRY_AFTER_RE.search(str(exc))
        if match:
            # +1s buffer past the server hint
            return float(match.group(1)) + 1.0

        if isinstance(exc, APITimeoutError):
            return float(self.timeout_retry_wait)

        # httpx / openai sometimes wrap timeouts without APITimeoutError
        msg = str(exc).lower()
        if "timeout" in msg or "timed out" in msg:
            return float(self.timeout_retry_wait)

        if isinstance(exc, RateLimitError):
            return max(60.0, self.retry_backoff * (2**attempt))

        return self.retry_backoff * (2**attempt)

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
                    timeout=self.timeout,
                )
                by_index = {item.index: item.embedding for item in response.data}
                return [by_index.get(i) for i in range(len(texts))]
            except Exception as exc:  # noqa: BLE001 - surface and retry API errors
                last_error = exc
                sleep_for = self._retry_wait_seconds(exc, attempt)
                print(
                    f"Embedding batch failed (attempt {attempt + 1}/{self.max_retries}): {exc}. "
                    f"Retrying in {sleep_for:.1f}s...",
                    flush=True,
                )
                time.sleep(sleep_for)

        print(f"Embedding batch failed permanently: {last_error}", flush=True)
        return [None] * len(texts)
