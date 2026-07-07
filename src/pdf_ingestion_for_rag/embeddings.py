"""Azure OpenAI embedding backend for `Chunk`s (optional).

Install the extra:  pip install pdf-ingestion-for-rag[azure]

Designed for `text-embedding-3-small` / `text-embedding-3-large`:
* batches inputs by both a max item count and a per-request token budget
  (the API caps a single request at 8191 tokens total across inputs),
* supports dimension shortening via the `dimensions` param,
* retries transient errors (rate limits, timeouts) with exponential backoff,
* preserves chunk order in the returned vectors.

Credentials are read from the standard Azure OpenAI environment variables by
default and can be overridden explicitly:
    AZURE_OPENAI_ENDPOINT       e.g. https://my-resource.openai.azure.com
    AZURE_OPENAI_API_KEY        (omit to use azure_ad_token / managed identity)
    OPENAI_API_VERSION          e.g. 2024-02-01
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from .models import Chunk
from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)

# Azure/OpenAI hard limit: a single embeddings request may total 8191 tokens
# across all inputs. Keep a margin for safety.
_MAX_TOKENS_PER_REQUEST = 8000
# The API also caps the number of inputs per request.
_MAX_ITEMS_PER_REQUEST = 2048


@dataclass
class AzureEmbedderConfig:
    """Connection + model settings for the Azure embedder."""

    deployment: str  # the Azure *deployment name* (not the base model name)
    endpoint: str | None = None
    api_key: str | None = None
    api_version: str = "2024-02-01"
    dimensions: int | None = None  # e.g. 256/512/1024 for text-embedding-3
    max_items_per_request: int = _MAX_ITEMS_PER_REQUEST
    max_tokens_per_request: int = _MAX_TOKENS_PER_REQUEST
    tokenizer_encoding: str = "cl100k_base"

    def __post_init__(self) -> None:
        self.endpoint = self.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = self.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("OPENAI_API_VERSION", self.api_version)
        if not self.endpoint:
            raise ValueError(
                "Azure endpoint not set (pass endpoint= or set AZURE_OPENAI_ENDPOINT)."
            )


@dataclass
class EmbeddedChunk:
    """A chunk paired with its embedding vector."""

    chunk: Chunk
    embedding: list[float]


@dataclass
class AzureOpenAIEmbedder:
    """Embeds chunks via Azure OpenAI with batching and retries."""

    config: AzureEmbedderConfig
    _client: object = field(default=None, init=False, repr=False)
    _tok: Tokenizer = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The Azure embedder requires the 'azure' extra: "
                "pip install pdf-ingestion-for-rag[azure]"
            ) from exc

        kwargs = {
            "azure_endpoint": self.config.endpoint,
            "api_version": self.config.api_version,
        }
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        # If no api_key is given, AzureOpenAI falls back to azure_ad_token /
        # AZURE_OPENAI_AD_TOKEN for managed-identity / Entra ID auth.
        self._client = AzureOpenAI(**kwargs)
        self._tok = Tokenizer(self.config.tokenizer_encoding)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed raw strings, preserving order."""
        vectors: list[list[float]] = []
        for batch in self._batches(texts):
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed `Chunk`s using their embedding-input text."""
        texts = [c.to_embedding_input() for c in chunks]
        vectors = self.embed_texts(texts)
        return [EmbeddedChunk(chunk=c, embedding=v) for c, v in zip(chunks, vectors)]

    # -- internals ----------------------------------------------------------

    def _batches(self, texts: list[str]) -> list[list[str]]:
        """Greedily pack texts into requests within item and token budgets."""
        cfg = self.config
        batches: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0
        for text in texts:
            t = self._tok.count(text)
            # A single text over the per-request budget still goes alone; each
            # chunk is already <= max_tokens (< 8191) from ingestion.
            if current and (
                len(current) >= cfg.max_items_per_request
                or current_tokens + t > cfg.max_tokens_per_request
            ):
                batches.append(current)
                current, current_tokens = [], 0
            current.append(text)
            current_tokens += t
        if current:
            batches.append(current)
        return batches

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        try:
            from openai import APIConnectionError, APITimeoutError, RateLimitError

            transient = (RateLimitError, APITimeoutError, APIConnectionError)
        except ImportError:  # pragma: no cover
            transient = (Exception,)

        @retry(
            retry=retry_if_exception_type(transient),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            stop=stop_after_attempt(6),
            reraise=True,
        )
        def _call() -> list[list[float]]:
            kwargs = {"model": self.config.deployment, "input": batch}
            if self.config.dimensions is not None:
                kwargs["dimensions"] = self.config.dimensions
            resp = self._client.embeddings.create(**kwargs)
            # API returns items in input order, but sort defensively on index.
            items = sorted(resp.data, key=lambda d: d.index)
            return [item.embedding for item in items]

        vectors = _call()
        if len(vectors) != len(batch):  # pragma: no cover - API contract guard
            raise RuntimeError(
                f"Embedding count mismatch: got {len(vectors)} for {len(batch)} inputs."
            )
        return vectors
