"""Ollama cloud embedding helpers for semantic retrieval."""

from __future__ import annotations

import os
from typing import Sequence

from ollama import Client

from src.ollama_client import API_KEY_ENV, OLLAMA_CLOUD_HOST, build_cloud_client

DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_BATCH_SIZE = 32
LOCAL_OLLAMA_HOST = "http://127.0.0.1:11434"


class EmbeddingUnavailableError(RuntimeError):
    """Raised when no Ollama endpoint can run the embedding model."""

# Prefixes recommended for nomic-embed-text retrieval tasks.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def format_for_embedding(text: str, *, as_query: bool) -> str:
    """Prefix text for nomic-embed-text document or query embedding."""
    prefix = QUERY_PREFIX if as_query else DOCUMENT_PREFIX
    return f"{prefix}{text}"


def embed_texts(
    texts: Sequence[str],
    *,
    client: Client | None = None,
    model: str = DEFAULT_EMBED_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    as_query: bool = False,
) -> list[list[float]]:
    """Embed one or more texts via Ollama cloud, returning vectors in order."""
    if not texts:
        return []

    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    if client is None:
        client = build_cloud_client()

    prefixed = [format_for_embedding(text, as_query=as_query) for text in texts]
    embeddings: list[list[float]] = []
    for start in range(0, len(prefixed), batch_size):
        batch = prefixed[start : start + batch_size]
        response = client.embed(model=model, input=batch)
        embeddings.extend(list(response.embeddings))
    return embeddings


def embed_query(
    query: str,
    *,
    client: Client | None = None,
    model: str = DEFAULT_EMBED_MODEL,
) -> list[float]:
    """Embed a single search query."""
    vectors = embed_texts([query], client=client, model=model, as_query=True)
    return vectors[0]


def resolve_embed_client() -> Client:
    """Return the first Ollama client that can embed with ``DEFAULT_EMBED_MODEL``.

    Tries the Ollama cloud API when ``OLLAMA_API_KEY`` is set, then a local
    daemon at :data:`LOCAL_OLLAMA_HOST`.
    """
    errors: list[str] = []

    if os.environ.get(API_KEY_ENV):
        cloud_client = build_cloud_client()
        try:
            embed_query("connectivity check", client=cloud_client)
            return cloud_client
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            errors.append(f"cloud ({OLLAMA_CLOUD_HOST}): {status or exc}")

    local_client = Client(host=LOCAL_OLLAMA_HOST)
    try:
        embed_query("connectivity check", client=local_client)
        return local_client
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        errors.append(f"local ({LOCAL_OLLAMA_HOST}): {status or exc}")

    joined = "; ".join(errors)
    raise EmbeddingUnavailableError(
        f"No working Ollama embedding endpoint for {DEFAULT_EMBED_MODEL} ({joined}). "
        "Cloud keys may not authorize /api/embed; for local fallback run "
        f"`ollama pull {DEFAULT_EMBED_MODEL}`."
    )
