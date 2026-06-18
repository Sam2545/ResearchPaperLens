"""Tests for :mod:`src.rag`."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import src.rag as rag_module
from src.chunk_store import ChunkStore, IndexedChunk
from src.rag import (
    RagAnswer,
    ask,
    build_context,
    format_rag_answer,
    rerank_results,
)
from src.vector_index import SearchResult


class FakeChatClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(self.payload))
        )


class FakeEmbedClient:
    def embed(self, **kwargs):
        inputs = kwargs["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        embeddings = [[1.0, 0.0, 0.0] for _ in inputs]
        return SimpleNamespace(embeddings=embeddings)


def _chunk(chunk_id: str, heading: str, text: str) -> IndexedChunk:
    return IndexedChunk(
        id=chunk_id,
        paper_source="data/paper.pdf",
        section_heading=heading,
        text=text,
        word_count=len(text.split()),
        start_word=0,
        end_word=len(text.split()),
        embedding=[1.0, 0.0, 0.0],
    )


def _store(*chunks: IndexedChunk) -> ChunkStore:
    return ChunkStore(
        paper_source="data/paper.pdf",
        paper_title="Sample Paper",
        embed_model="nomic-embed-text",
        embed_dimensions=3,
        chunks=list(chunks),
    )


def _hit(chunk: IndexedChunk, score: float = 0.9) -> SearchResult:
    return SearchResult(chunk=chunk, score=score)


def test_build_context_includes_ranked_sources_in_order():
    results = [
        _hit(_chunk("a", "6 Results", "BLEU 28.4")),
        _hit(_chunk("b", "1 Introduction", "We propose transformers.")),
    ]
    context, included = build_context(results, max_chars=10_000)
    assert "Source 1" in context
    assert "6 Results" in context
    assert "BLEU 28.4" in context
    assert len(included) == 2


def test_build_context_respects_max_chars():
    long_text = "word " * 5000
    results = [_hit(_chunk("a", "Section", long_text))]
    context, included = build_context(results, max_chars=500)
    assert len(context) <= 500
    assert len(included) == 1


def test_build_context_includes_multiple_large_chunks():
    results = [
        _hit(_chunk("a", "2 Background", "background " * 400), score=0.66),
        _hit(_chunk("b", "4 Why Self-Attention", "analysis " * 700), score=0.65),
        _hit(
            _chunk(
                "c",
                "3 Model Architecture",
                "multi-head attention maps queries keys values " * 50,
            ),
            score=0.64,
        ),
    ]
    context, included = build_context(results, max_chars=12_000)
    ids = [hit.chunk.id for hit in included]
    assert "a" in ids
    assert "c" in ids
    assert "multi-head attention" in context


def test_rerank_promotes_keyword_rich_chunks():
    results = [
        _hit(_chunk("a", "2 Background", "brief mention of attention"), score=0.66),
        _hit(
            _chunk(
                "b",
                "3 Model Architecture",
                "multi-head attention uses parallel attention heads",
            ),
            score=0.65,
        ),
    ]
    reranked = rerank_results("How does multi-head attention work?", results)
    assert reranked[0].chunk.id == "b"
    assert reranked[0].score > results[0].score


def test_ask_returns_empty_message_when_no_chunks():
    store = _store()
    result = ask(
        store,
        "What is BLEU?",
        embed_client=FakeEmbedClient(),
        chat_client=FakeChatClient({"answer": "unused", "citations": []}),
    )
    assert "No relevant passages" in result.answer
    assert result.citations == []


def test_ask_uses_retrieval_and_chat():
    chunk = _chunk("paper:6-results:0", "6 Results", "BLEU score was 28.4.")
    store = _store(chunk)
    chat_client = FakeChatClient(
        {
            "answer": "The model achieved a BLEU score of 28.4.",
            "citations": ["6 Results"],
        }
    )

    result = ask(
        store,
        "What BLEU score was reported?",
        embed_client=FakeEmbedClient(),
        chat_client=chat_client,
        top_k=1,
    )

    assert isinstance(result, RagAnswer)
    assert "28.4" in result.answer
    assert result.citations == ["6 Results"]
    assert result.sources == ["paper:6-results:0"]
    assert chat_client.calls
    user_msg = chat_client.calls[0]["messages"][-1]["content"]
    assert "BLEU score was 28.4" in user_msg
    assert chat_client.calls[0]["format"] == rag_module._RESPONSE_SCHEMA


def test_format_rag_answer_includes_citations_and_sources():
    result = RagAnswer(
        question="Q?",
        answer="An answer.",
        citations=["6 Results"],
        sources=["paper:6-results:0"],
    )
    text = format_rag_answer(result)
    assert "Q?" in text
    assert "An answer." in text
    assert "6 Results" in text
    assert "paper:6-results:0" in text


def test_ask_rejects_empty_question():
    store = _store(_chunk("a", "Section", "text"))
    with pytest.raises(ValueError, match="question"):
        ask(store, "  ", embed_client=FakeEmbedClient())
