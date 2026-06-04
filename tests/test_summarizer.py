"""Tests for the placeholder summarizer (:mod:`src.summarizer`).

These verify the summarizer is wired up correctly -- not that it produces real
summaries (it is a stub). They check the function signatures, return shapes, and
that the summary fields are attached to the ResearchPaper without disturbing the
rest of the paper.
"""

from __future__ import annotations

from dataclasses import replace

from src.paper import ResearchPaper
from src.summarizer import summarize, summarize_text


def test_summarize_text_returns_str_and_list():
    summary, key_insights = summarize_text("some paper body text")
    assert isinstance(summary, str)
    assert isinstance(key_insights, list)


def test_summarize_returns_research_paper_with_summary_fields():
    paper = ResearchPaper(title="A Paper", full_text="body")
    result = summarize(paper)
    assert isinstance(result, ResearchPaper)
    assert isinstance(result.summary, str)
    assert isinstance(result.key_insights, list)


def test_summarize_preserves_other_fields():
    paper = ResearchPaper(
        title="A Paper",
        authors=["Jane Doe"],
        abstract="An abstract.",
        full_text="body",
    )
    result = summarize(paper)
    # Everything except the two summary fields should be unchanged.
    assert replace(result, summary="", key_insights=[]) == paper


def test_summarize_does_not_mutate_input():
    paper = ResearchPaper(title="A Paper", full_text="body")
    summarize(paper)
    assert paper.summary == ""
    assert paper.key_insights == []
