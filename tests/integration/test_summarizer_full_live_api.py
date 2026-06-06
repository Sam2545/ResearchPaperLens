Live integration test for full-paper chunked summarization."""Live integration test for full-paper chunked summarization.

Calls the real Ollama cloud API to chunk a paper, summarize each chunk, and
merge the partial summaries. Skipped unless ``OLLAMA_API_KEY`` is available.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from src.analyzer import analyze
from src.chunking import chunk_paper
from src.paper import PaperSummary
from src.pdf_reader import read_pdf
from src.summarizer import API_KEY_ENV, summarize_full

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.environ.get(API_KEY_ENV),
    reason=f"{API_KEY_ENV} not set; skipping live Ollama cloud API test",
)

PDF_PATH = (
    __import__("pathlib").Path(__file__).resolve().parents[2]
    / "data"
    / "AttentionIsAllYouNeed.pdf"
)

# Keep chunk count modest so the live test stays reasonably fast.
LIVE_WORDS_PER_CHUNK = 1200
LIVE_OVERLAP_WORDS = 100


def test_live_summarize_full_produces_coherent_summary():
    full_text, page_count = read_pdf(PDF_PATH)
    paper = analyze(
        full_text=full_text,
        page_count=page_count,
        source_path=str(PDF_PATH),
    )
    chunks = chunk_paper(
        paper,
        LIVE_WORDS_PER_CHUNK,
        overlap_words=LIVE_OVERLAP_WORDS,
    )
    assert len(chunks) >= 2, "expected multiple chunks for a full-paper test"

    result = summarize_full(
        paper,
        words_per_chunk=LIVE_WORDS_PER_CHUNK,
        overlap_words=LIVE_OVERLAP_WORDS,
    )
    summary = result.summary

    assert isinstance(summary, PaperSummary)
    assert summary.tldr.strip(), "expected a non-empty tldr"
    assert summary.problem.strip(), "expected a non-empty problem"
    assert summary.approach.strip(), "expected a non-empty approach"
    assert summary.key_results, "expected populated key_results"
    assert summary.contributions, "expected populated contributions"
    assert summary.key_insights, "expected populated key_insights"

    joined = " ".join(
        [
            summary.tldr,
            summary.problem,
            summary.approach,
            *summary.key_results,
            *summary.contributions,
            *summary.key_insights,
        ]
    ).lower()
    assert "transformer" in joined or "attention" in joined
