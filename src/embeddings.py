"""Ollama cloud embedding helpers for semantic retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from src.ollama_client import build_cloud_client

if TYPE_CHECKING:
    from ollama import Client

DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_BATCH_SIZE = 32

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
