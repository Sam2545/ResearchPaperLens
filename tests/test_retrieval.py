"""Tests for :mod:`src.retrieval`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.chunk_store import ChunkStore
from src.chunking import ABSTRACT_SECTION_HEADING, PREAMBLE_SECTION_HEADING
from src.paper import ResearchPaper
from src.retrieval import (
    chunks_from_plan,
    index_paper,
    make_chunk_id,
    plan_retrieval_chunks,
    search,
)


class FakeEmbedClient:
    def __init__(self):
        self.calls: list[dict] = []

    def embed(self, **kwargs):
        self.calls.append(kwargs)
        inputs = kwargs["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        embeddings = []
        for text in inputs:
            if "BLEU" in text:
                embeddings.append([1.0, 0.0, 0.0])
            elif "attention mechanism" in text.lower():
                embeddings.append([0.0, 1.0, 0.0])
            else:
                embeddings.append([0.0, 0.0, 1.0])
        return SimpleNamespace(embeddings=embeddings)


@pytest.fixture
def sample_paper() -> ResearchPaper:
    return ResearchPaper(
        title="Sample Paper",
        abstract="We propose a new attention mechanism.",
        full_text=(
            "Title block before sections.\n\n"
            "1 Introduction\n"
            "This paper introduces transformers.\n\n"
            "6 Results\n"
            "Our model achieves BLEU scores of 28.4 on WMT."
        ),
        source_path="data/SamplePaper.pdf",
    )


def test_plan_retrieval_chunks_includes_abstract_not_preamble(sample_paper: ResearchPaper):
    plans = plan_retrieval_chunks(sample_paper)
    headings = [plan.section.heading for plan in plans]
    assert ABSTRACT_SECTION_HEADING in headings
    assert PREAMBLE_SECTION_HEADING not in headings
    assert "1 Introduction" in headings
    assert "6 Results" in headings


def test_make_chunk_id_is_stable(sample_paper: ResearchPaper):
    assert make_chunk_id(sample_paper, "6 Results", 0) == "SamplePaper:6-results:0"


def test_index_paper_embeds_hybrid_chunks(sample_paper: ResearchPaper):
    client = FakeEmbedClient()
    store = index_paper(sample_paper, client=client)

    assert store.paper_title == "Sample Paper"
    assert store.embed_model == "nomic-embed-text"
    assert store.embed_dimensions == 3
    assert len(store.chunks) >= 3
    assert all(chunk.embedding for chunk in store.chunks)
    assert client.calls


def test_search_returns_top_matching_chunk(sample_paper: ResearchPaper):
    client = FakeEmbedClient()
    store = index_paper(sample_paper, client=client)
    results = search(store, "What BLEU score was reported?", client=client, top_k=1)

    assert len(results) == 1
    assert "BLEU" in results[0].chunk.text
    assert results[0].score > 0.9


def test_search_empty_store_returns_empty():
    store = ChunkStore(
        paper_source="data/empty.pdf",
        paper_title="Empty",
        embed_model="nomic-embed-text",
        embed_dimensions=0,
        chunks=[],
    )
    assert search(store, "anything", client=FakeEmbedClient()) == []


def test_chunks_from_plan_skips_empty_text():
    paper = ResearchPaper(source_path="data/x.pdf")
    plans = plan_retrieval_chunks(paper)
    assert chunks_from_plan(paper, plans) == []
