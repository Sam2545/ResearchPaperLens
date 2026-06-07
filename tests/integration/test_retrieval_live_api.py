"""Live integration test for semantic retrieval with real embeddings.

Indexes the bundled Attention paper with hybrid section chunks, embeds them
through Ollama, and checks that a BLEU-focused query retrieves content from
the Results section.

Uses :func:`~src.embeddings.resolve_embed_client` (cloud first, then local).
If no endpoint works, the test is skipped.

Run explicitly with::

    python -m pytest tests/integration/test_retrieval_live_api.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.analyzer import analyze
from src.chunk_store import load_chunk_store, save_chunk_store
from src.embeddings import DEFAULT_EMBED_MODEL, EmbeddingUnavailableError, resolve_embed_client
from src.pdf_reader import read_pdf
from src.retrieval import index_paper, search

load_dotenv()

PDF_PATH = Path(__file__).resolve().parents[2] / "data" / "AttentionIsAllYouNeed.pdf"

BLEU_QUERY = (
    "What BLEU scores did the Transformer achieve on English-German "
    "machine translation?"
)


def test_live_index_and_search_finds_bleu_in_results_section(tmp_path):
    try:
        client = resolve_embed_client()
    except EmbeddingUnavailableError as exc:
        pytest.skip(str(exc))

    full_text, page_count = read_pdf(PDF_PATH)
    paper = analyze(
        full_text=full_text,
        page_count=page_count,
        source_path=str(PDF_PATH),
    )

    store = index_paper(paper, client=client)
    assert store.chunks, "expected indexed chunks for the Attention paper"
    assert store.embed_dimensions > 0, "expected non-empty embedding vectors"
    assert store.embed_model == DEFAULT_EMBED_MODEL

    chunk_path = tmp_path / "AttentionIsAllYouNeed.chunks.json"
    save_chunk_store(store, chunk_path)
    reloaded = load_chunk_store(chunk_path)
    assert len(reloaded.chunks) == len(store.chunks)

    results = search(reloaded, BLEU_QUERY, client=client, top_k=5)
    assert results, "expected at least one search result"

    top = results[0]
    assert "results" in top.chunk.section_heading.lower(), (
        f"expected Results section in top hit, got {top.chunk.section_heading!r}"
    )
    assert "bleu" in top.chunk.text.lower(), (
        "expected BLEU metrics in the top retrieved chunk text"
    )
    assert top.score > 0, "expected a positive cosine similarity score"

    headings = [hit.chunk.section_heading for hit in results[:3]]
    assert any("results" in heading.lower() for heading in headings), (
        f"expected a Results-section chunk in top 3, got {headings}"
    )
