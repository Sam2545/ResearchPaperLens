"""Live integration test for full-paper hybrid section summarization.

Calls the real Ollama cloud API to split a paper into sections, summarize
each section (subdividing long ones with word chunks), and merge into a final
summary. Skipped unless ``OLLAMA_API_KEY`` is available.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from src.analyzer import analyze
from src.chunking import plan_hybrid_chunks
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

LIVE_MAX_SECTION_WORDS = 1200
LIVE_WORDS_PER_CHUNK = 1200
LIVE_OVERLAP_WORDS = 100


def test_live_summarize_full_hybrid_produces_coherent_summary():
    full_text, page_count = read_pdf(PDF_PATH)
    paper = analyze(
        full_text=full_text,
        page_count=page_count,
        source_path=str(PDF_PATH),
    )
    plans = plan_hybrid_chunks(
        paper,
        max_section_words=LIVE_MAX_SECTION_WORDS,
        words_per_chunk=LIVE_WORDS_PER_CHUNK,
        overlap_words=LIVE_OVERLAP_WORDS,
    )
    assert len(plans) >= 3, "expected multiple hybrid sections for a full-paper test"
    assert any(
        plan.section.heading == "6 Results" for plan in plans
    ), "expected a Results section in the hybrid plan"

    result = summarize_full(
        paper,
        strategy="hybrid",
        max_section_words=LIVE_MAX_SECTION_WORDS,
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
