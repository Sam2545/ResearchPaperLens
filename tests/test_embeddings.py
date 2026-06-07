"""Tests for :mod:`src.embeddings`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.embeddings import (
    DEFAULT_EMBED_MODEL,
    DOCUMENT_PREFIX,
    QUERY_PREFIX,
    embed_query,
    embed_texts,
    format_for_embedding,
)
from src.ollama_client import API_KEY_ENV, build_cloud_client
import src.ollama_client as ollama_client_module


class FakeEmbedClient:
    def __init__(self):
        self.calls: list[dict] = []

    def embed(self, **kwargs):
        self.calls.append(kwargs)
        inputs = kwargs["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        embeddings = [[float(i), float(i + 1), float(i + 2)] for i in range(len(inputs))]
        return SimpleNamespace(embeddings=embeddings)


def test_format_for_embedding_uses_document_prefix():
    assert format_for_embedding("hello", as_query=False) == f"{DOCUMENT_PREFIX}hello"


def test_format_for_embedding_uses_query_prefix():
    assert format_for_embedding("hello", as_query=True) == f"{QUERY_PREFIX}hello"


def test_embed_texts_returns_empty_for_empty_input():
    client = FakeEmbedClient()
    assert embed_texts([], client=client) == []
    assert client.calls == []


def test_embed_texts_batches_and_prefixes_documents():
    client = FakeEmbedClient()
    texts = ["alpha", "beta", "gamma"]
    vectors = embed_texts(texts, client=client, batch_size=2)

    assert len(vectors) == 3
    assert len(client.calls) == 2
    assert client.calls[0]["model"] == DEFAULT_EMBED_MODEL
    assert client.calls[0]["input"] == [
        f"{DOCUMENT_PREFIX}alpha",
        f"{DOCUMENT_PREFIX}beta",
    ]
    assert client.calls[1]["input"] == [f"{DOCUMENT_PREFIX}gamma"]


def test_embed_query_prefixes_and_returns_single_vector():
    client = FakeEmbedClient()
    vector = embed_query("BLEU score", client=client)

    assert vector == [0.0, 1.0, 2.0]
    assert client.calls[0]["input"] == [f"{QUERY_PREFIX}BLEU score"]


def test_embed_texts_rejects_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        embed_texts(["x"], client=FakeEmbedClient(), batch_size=0)


def test_build_cloud_client_requires_api_key(monkeypatch):
    monkeypatch.setattr(ollama_client_module, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=API_KEY_ENV):
        build_cloud_client()
