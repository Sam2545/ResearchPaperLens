"""Tests for :mod:`src.storage` (ResearchPaper <-> JSON persistence)."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from src.paper import ResearchPaper
from src.storage import (
    load_paper,
    paper_from_dict,
    paper_from_json,
    paper_to_json,
    save_paper,
)


@pytest.fixture
def sample_paper() -> ResearchPaper:
    return ResearchPaper(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        abstract="The dominant sequence transduction models...",
        keywords=["attention", "transformer"],
        section_headings=["1 Introduction", "2 Background"],
        page_count=15,
        word_count=2033,
        full_text="full text body",
        references=["[1] some reference"],
        doi="10.0000/example",
        publication_year=2017,
        venue="NIPS",
        source_path="data/AttentionIsAllYouNeed.pdf",
    )


def test_dict_round_trip(sample_paper: ResearchPaper):
    assert paper_from_dict(asdict(sample_paper)) == sample_paper


def test_json_round_trip(sample_paper: ResearchPaper):
    assert paper_from_json(paper_to_json(sample_paper)) == sample_paper


def test_paper_to_json_is_pretty_printed(sample_paper: ResearchPaper):
    text = paper_to_json(sample_paper)
    assert text.startswith("{\n")
    assert '  "title": "Attention Is All You Need"' in text


def test_paper_to_json_preserves_unicode(sample_paper: ResearchPaper):
    paper = ResearchPaper(authors=["Łukasz Kaiser"])
    text = paper_to_json(paper)
    # ensure_ascii=False keeps the original characters readable.
    assert "Łukasz Kaiser" in text


def test_save_and_load_round_trip(tmp_path, sample_paper: ResearchPaper):
    path = tmp_path / "paper.json"
    save_paper(sample_paper, path)
    assert path.is_file()
    assert load_paper(path) == sample_paper


def test_save_creates_parent_directories(tmp_path, sample_paper: ResearchPaper):
    path = tmp_path / "nested" / "dir" / "paper.json"
    save_paper(sample_paper, path)
    assert path.is_file()


def test_saved_file_is_valid_json(tmp_path, sample_paper: ResearchPaper):
    path = tmp_path / "paper.json"
    save_paper(sample_paper, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["title"] == sample_paper.title


def test_from_dict_ignores_unknown_keys():
    data = {"title": "T", "unexpected_field": "ignored", "page_count": 3}
    paper = paper_from_dict(data)
    assert paper.title == "T"
    assert paper.page_count == 3
    assert not hasattr(paper, "unexpected_field")


def test_from_dict_uses_defaults_for_missing_keys():
    paper = paper_from_dict({"title": "Only Title"})
    assert paper.title == "Only Title"
    assert paper.authors == []
    assert paper.page_count == 0
    assert paper.abstract == ""
