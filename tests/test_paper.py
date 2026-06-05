"""Tests for the :class:`ResearchPaper` data model (:mod:`src.paper`)."""

from __future__ import annotations

from dataclasses import fields

from src.paper import PaperSummary, ResearchPaper


def test_summary_field_exists_and_defaults_to_paper_summary():
    field_names = {f.name for f in fields(ResearchPaper)}
    assert "summary" in field_names
    paper = ResearchPaper()
    assert isinstance(paper.summary, PaperSummary)


def test_paper_summary_has_empty_defaults():
    summary = PaperSummary()
    assert summary.tldr == ""
    assert summary.problem == ""
    assert summary.approach == ""
    assert summary.key_results == []
    assert summary.contributions == []
    assert summary.limitations == []
    assert summary.key_insights == []


def test_summary_default_is_not_shared_between_instances():
    first = ResearchPaper()
    first.summary.key_insights.append("insight")
    second = ResearchPaper()
    assert second.summary.key_insights == []


def test_summary_accepts_a_paper_summary():
    summary = PaperSummary(
        tldr="A short overview.",
        key_results=["BLEU 28.4"],
        key_insights=["insight one", "insight two"],
    )
    paper = ResearchPaper(summary=summary)
    assert paper.summary.tldr == "A short overview."
    assert paper.summary.key_results == ["BLEU 28.4"]
    assert paper.summary.key_insights == ["insight one", "insight two"]
