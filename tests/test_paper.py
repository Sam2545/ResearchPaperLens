"""Tests for the :class:`ResearchPaper` data model (:mod:`src.paper`)."""

from __future__ import annotations

from dataclasses import fields

from src.paper import ResearchPaper


def test_summary_and_key_insights_fields_exist():
    field_names = {f.name for f in fields(ResearchPaper)}
    assert "summary" in field_names
    assert "key_insights" in field_names


def test_summary_and_key_insights_have_empty_defaults():
    paper = ResearchPaper()
    assert paper.summary == ""
    assert paper.key_insights == []


def test_key_insights_default_is_not_shared_between_instances():
    first = ResearchPaper()
    first.key_insights.append("insight")
    second = ResearchPaper()
    assert second.key_insights == []


def test_summary_and_key_insights_accept_values():
    paper = ResearchPaper(
        summary="A short summary.",
        key_insights=["insight one", "insight two"],
    )
    assert paper.summary == "A short summary."
    assert paper.key_insights == ["insight one", "insight two"]
