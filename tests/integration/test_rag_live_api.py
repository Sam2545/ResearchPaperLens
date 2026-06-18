"""Live integration test for RAG Q&A over indexed paper chunks."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.analyzer import analyze
from src.chunk_store import save_chunk_store
from src.embeddings import EmbeddingUnavailableError, resolve_embed_client
from src.ollama_client import API_KEY_ENV, build_cloud_client
from src.pdf_reader import read_pdf
from src.rag import ask
from src.retrieval import index_paper

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.environ.get(API_KEY_ENV),
    reason=f"{API_KEY_ENV} not set; skipping live RAG test",
)

PDF_PATH = Path(__file__).resolve().parents[2] / "data" / "AttentionIsAllYouNeed.pdf"

BLEU_QUESTION = (
    "What BLEU scores did the Transformer achieve on English-to-German "
    "machine translation?"
)


def test_live_rag_answers_bleu_question_from_retrieved_context(tmp_path):
    try:
        embed_client = resolve_embed_client()
    except EmbeddingUnavailableError as exc:
        pytest.skip(str(exc))

    full_text, page_count = read_pdf(PDF_PATH)
    paper = analyze(
        full_text=full_text,
        page_count=page_count,
        source_path=str(PDF_PATH),
    )
    store = index_paper(paper, client=embed_client)
    chunk_path = tmp_path / "AttentionIsAllYouNeed.chunks.json"
    save_chunk_store(store, chunk_path)

    chat_client = build_cloud_client()
    result = ask(
        store,
        BLEU_QUESTION,
        top_k=5,
        embed_client=embed_client,
        chat_client=chat_client,
    )

    assert result.answer.strip(), "expected a non-empty RAG answer"
    joined = " ".join([result.answer, *result.citations]).lower()
    assert "bleu" in joined
    assert result.sources, "expected retrieved chunk ids in sources"
