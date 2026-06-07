"""Tests for :mod:`src.vector_index`."""

from __future__ import annotations

import pytest

from src.chunk_store import ChunkStore, IndexedChunk
from src.vector_index import InMemoryVectorIndex, SearchResult


def _chunk(chunk_id: str, embedding: list[float], text: str = "") -> IndexedChunk:
    return IndexedChunk(
        id=chunk_id,
        paper_source="data/paper.pdf",
        section_heading="Section",
        text=text or chunk_id,
        word_count=1,
        start_word=0,
        end_word=1,
        embedding=embedding,
    )


def test_search_ranks_by_cosine_similarity():
    chunks = [
        _chunk("a", [1.0, 0.0, 0.0]),
        _chunk("b", [0.0, 1.0, 0.0]),
        _chunk("c", [0.9, 0.1, 0.0]),
    ]
    index = InMemoryVectorIndex(chunks)
    results = index.search([1.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    assert results[0].chunk.id == "a"
    assert results[1].chunk.id == "c"
    assert results[0].score > results[1].score


def test_from_store_builds_index():
    store = ChunkStore(
        paper_source="data/paper.pdf",
        paper_title="Paper",
        embed_model="nomic-embed-text",
        embed_dimensions=2,
        chunks=[_chunk("only", [1.0, 0.0])],
    )
    index = InMemoryVectorIndex.from_store(store)
    results = index.search([1.0, 0.0], top_k=1)
    assert results[0].chunk.id == "only"


def test_requires_embeddings():
    chunk = IndexedChunk(
        id="missing",
        paper_source="data/paper.pdf",
        section_heading="Section",
        text="text",
        word_count=1,
        start_word=0,
        end_word=1,
    )
    with pytest.raises(ValueError, match="embeddings"):
        InMemoryVectorIndex([chunk])


def test_rejects_zero_query_vector():
    index = InMemoryVectorIndex([_chunk("a", [1.0, 0.0])])
    with pytest.raises(ValueError, match="non-zero"):
        index.search([0.0, 0.0], top_k=1)
