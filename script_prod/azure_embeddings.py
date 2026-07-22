"""Azure OpenAI embedding helpers for production sync."""

from __future__ import annotations

import os
import re
import time
from typing import Sequence

from common import EMBED_TIMEOUT_SECONDS, PROJECT_ROOT, load_dotenv_file

_RETRY_AFTER_RE = re.compile(r"retry after\s+(\d+(?:\.\d+)?)\s+seconds?", re.IGNORECASE)


def _parse_deployments(raw: str) -> list[str]:
    """Split comma/whitespace-separated deployment names; keep order, drop empties."""
    parts = re.split(r"[\s,]+", (raw or "").strip())
    return [p for p in parts if p]


class AzureEmbedder:
    """Calls Azure with one or more AZURE_OPENAI_EMBED_DEPLOYMENT names (round-robin)."""

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
        raw_deployment = deployment if deployment is not None else (os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT") or "")
        self.deployments = _parse_deployments(raw_deployment)
        self.api_version = (
            api_version
            or os.environ.get("AZURE_OPENAI_EMBED_API_VERSION")
            or "2024-02-01"
        )
        if not endpoint or not api_key:
            raise SystemExit(
                "Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY in environment or project .env."
            )
        if not self.deployments:
            raise SystemExit(
                "Missing AZURE_OPENAI_EMBED_DEPLOYMENT in environment or project .env "
                "(one deployment name, or comma-separated list for round-robin). "
                "Not the same as config embedding_model."
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
        self._rr_index = 0
        print(
            f"Azure embeddings: {len(self.deployments)} deployment(s) "
            f"[{', '.join(self.deployments)}]",
            flush=True,
        )

    @property
    def deployment(self) -> str:
        """Primary / current deployment name (compat for single-deployment callers)."""
        return self.deployments[self._rr_index % len(self.deployments)]

    def _next_deployment(self) -> str:
        dep = self.deployments[self._rr_index % len(self.deployments)]
        self._rr_index += 1
        return dep

    def _retry_wait_seconds(self, exc: BaseException, attempt: int) -> float:
        """Prefer Azure's 'retry after N seconds'; timeouts wait 60s; else exponential."""
        from openai import APITimeoutError, RateLimitError

        match = _RETRY_AFTER_RE.search(str(exc))
        if match:
            return float(match.group(1)) + 1.0

        if isinstance(exc, APITimeoutError):
            return float(self.timeout_retry_wait)

        msg = str(exc).lower()
        if "timeout" in msg or "timed out" in msg:
            return float(self.timeout_retry_wait)

        if isinstance(exc, RateLimitError):
            return max(60.0, self.retry_backoff * (2**attempt))

        return self.retry_backoff * (2**attempt)

    def _is_rate_limit(self, exc: BaseException) -> bool:
        from openai import RateLimitError

        if isinstance(exc, RateLimitError):
            return True
        msg = str(exc).lower()
        return "ratelimitreached" in msg or "rate limit" in msg

    def embed_texts(self, texts: Sequence[str]) -> list[list[float] | None]:
        """Embed texts; failed items become None (caller may still upsert)."""
        if not texts:
            return []

        last_error: Exception | None = None
        # Each attempt may try every deployment once before sleeping on rate limits.
        for attempt in range(self.max_retries):
            # Start from next RR slot so load spreads across deployments.
            start = self._rr_index % len(self.deployments)
            for offset in range(len(self.deployments)):
                dep = self.deployments[(start + offset) % len(self.deployments)]
                try:
                    response = self.client.embeddings.create(
                        model=dep,
                        input=list(texts),
                        timeout=self.timeout,
                    )
                    self._rr_index = (start + offset + 1) % len(self.deployments)
                    by_index = {item.index: item.embedding for item in response.data}
                    return [by_index.get(i) for i in range(len(texts))]
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if self._is_rate_limit(exc) and offset + 1 < len(self.deployments):
                        print(
                            f"Embedding rate-limited on {dep}; trying next deployment...",
                            flush=True,
                        )
                        continue
                    sleep_for = self._retry_wait_seconds(exc, attempt)
                    print(
                        f"Embedding batch failed on {dep} "
                        f"(attempt {attempt + 1}/{self.max_retries}): {exc}. "
                        f"Retrying in {sleep_for:.1f}s...",
                        flush=True,
                    )
                    time.sleep(sleep_for)
                    break  # next outer attempt

        print(f"Embedding batch failed permanently: {last_error}", flush=True)
        return [None] * len(texts)
