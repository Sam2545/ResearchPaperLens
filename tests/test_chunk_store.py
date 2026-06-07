"""Tests for :mod:`src.chunk_store`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.chunk_store import (
    ChunkStore,
    IndexedChunk,
    default_chunk_store_path,
    load_chunk_store,
    save_chunk_store,
    store_from_dict,
    store_from_json,
    store_to_json,
)
from src.paper import ResearchPaper


@pytest.fixture
def sample_store() -> ChunkStore:
    return ChunkStore(
        paper_source="data/AttentionIsAllYouNeed.pdf",
        paper_title="Attention Is All You Need",
        embed_model="nomic-embed-text",
        embed_dimensions=3,
        chunks=[
            IndexedChunk(
                id="attention-is-all-you-need:6-results:0",
                paper_source="data/AttentionIsAllYouNeed.pdf",
                section_heading="6 Results",
                text="Our model achieves BLEU scores of 28.4.",
                word_count=8,
                start_word=100,
                end_word=108,
                embedding=[0.1, 0.2, 0.3],
            )
        ],
    )


def test_store_json_round_trip(sample_store: ChunkStore):
    assert store_from_json(store_to_json(sample_store)) == sample_store


def test_store_from_dict_ignores_unknown_keys(sample_store: ChunkStore):
    data = json.loads(store_to_json(sample_store))
    data["unexpected"] = "ignored"
    data["chunks"][0]["extra"] = "ignored"
    assert store_from_dict(data) == sample_store


def test_save_and_load_round_trip(tmp_path, sample_store: ChunkStore):
    path = tmp_path / "paper.chunks.json"
    save_chunk_store(sample_store, path)
    assert load_chunk_store(path) == sample_store


def test_default_chunk_store_path_uses_source_stem():
    paper = ResearchPaper(source_path="data/AttentionIsAllYouNeed.pdf")
    assert default_chunk_store_path(paper) == Path("outputs/AttentionIsAllYouNeed.chunks.json")
