"""Tests for :mod:`src.analyzer`.

Pure helper functions are tested with small synthetic text (fast and
deterministic). The :class:`PaperAnalyzer` integration is tested once against
the real "Attention Is All You Need" PDF via a module-scoped fixture so the PDF
is opened and extracted a single time for the whole module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.analyzer import (
    PaperAnalyzer,
    analyze,
    count_words,
    guess_abstract,
    guess_authors,
    guess_keywords,
    guess_section_headings,
    guess_title,
)
from src.paper import ResearchPaper
from src.pdf_reader import read_pdf

PDF_PATH = Path(__file__).resolve().parent.parent / "data" / "AttentionIsAllYouNeed.pdf"

SAMPLE_TEXT = "\n".join(
    [
        "Some boilerplate notice line that is one glued token here",
        "A Great Paper About Things",
        "JaneDoe∗ JohnSmith∗",
        "Example University",
        "jane@example.com john@example.com",
        "Abstract",
        "This paper introduces a method. It works well on benchmarks.",
        "Keywords: machine learning, transformers, attention",
        "1 Introduction",
        "Intro body text goes here.",
        "2 Related Work",
        "More body text.",
        "2.1 Background",
        "Even more text.",
    ]
)

EXPECTED_AUTHORS = [
    "Ashish Vaswani",
    "Noam Shazeer",
    "Niki Parmar",
    "Jakob Uszkoreit",
    "Llion Jones",
    "Aidan N. Gomez",
    "Łukasz Kaiser",
    "Illia Polosukhin",
]


# --- Pure helper tests (synthetic input) ------------------------------------


def test_count_words():
    assert count_words("one two three") == 3
    assert count_words("   spaced   out  words ") == 3
    assert count_words("") == 0


def test_guess_title():
    assert guess_title(SAMPLE_TEXT) == "A Great Paper About Things"


def test_guess_title_skips_glued_single_token_lines():
    text = "Singlegluedtokenlinewithnospaces\nReal Title Here\nAbstract\nbody"
    assert guess_title(text) == "Real Title Here"


def test_guess_title_returns_empty_when_absent():
    assert guess_title("abstract\nbody text only") == ""


def test_guess_title_joins_wrapped_lines():
    text = "A Multi Line Title\nThat Wraps Here\nJohn Smith1\nAbstract\nbody"
    assert guess_title(text) == "A Multi Line Title That Wraps Here"


def test_guess_title_stops_at_author_line():
    # The author line (has a marker) must not be absorbed into the title.
    text = "Real Title Line\nAuthor One∗ Author Two∗\nAbstract\nbody"
    assert guess_title(text) == "Real Title Line"


def test_guess_section_headings():
    assert guess_section_headings(SAMPLE_TEXT) == [
        "1 Introduction",
        "2 Related Work",
        "2.1 Background",
    ]


def test_guess_section_headings_ignores_lowercase_and_long_lines():
    text = "1 introduction lowercase\n2 A" + "x" * 80 + "\n3 Valid Heading"
    assert guess_section_headings(text) == ["3 Valid Heading"]


def test_guess_authors():
    assert guess_authors(SAMPLE_TEXT) == ["Jane Doe", "John Smith"]


def test_guess_authors_comma_separated_with_affiliation_digits():
    text = (
        "A Paper Title\n"
        "Jane Doe1, John Smith1, Mary Major2\n"
        "1Example University 2Other Institute\n"
        "Abstract\n"
        "body"
    )
    assert guess_authors(text) == ["Jane Doe", "John Smith", "Mary Major"]


def test_guess_authors_skips_affiliation_and_email_lines():
    text = (
        "A Paper Title\n"
        "Jane Doe∗\n"
        "Example University\n"
        "jane@example.com\n"
        "Abstract\n"
        "body"
    )
    assert guess_authors(text) == ["Jane Doe"]


def test_guess_authors_empty_without_markers():
    # A plain capitalized line with no markers/digits is treated as a title
    # continuation, so there is no author line to parse.
    assert guess_authors("Title Line\nMore Title\nAbstract\nbody") == []


def test_guess_abstract():
    assert (
        guess_abstract(SAMPLE_TEXT)
        == "This paper introduces a method. It works well on benchmarks."
    )


def test_guess_abstract_stops_at_section_heading():
    text = "Abstract\nLine one.\n1 Introduction\nbody"
    assert guess_abstract(text) == "Line one."


def test_guess_abstract_empty_when_absent():
    assert guess_abstract("Title\nbody with no abstract marker") == ""


def test_guess_abstract_inline_colon():
    text = "Title\nAbstract: This is the abstract.\n1 Introduction\nbody"
    assert guess_abstract(text) == "This is the abstract."


def test_guess_abstract_tolerates_stray_prefix():
    # A stray one-letter prefix (e.g. a figure label) before "Abstract".
    text = "Title\nb Abstract\nThe real abstract text.\n1 Introduction"
    assert guess_abstract(text) == "The real abstract text."


def test_guess_abstract_ignores_word_starting_with_abstract():
    # "Abstractive" must not be mistaken for an abstract heading.
    assert guess_abstract("Title\nAbstractive methods are great") == ""


def test_guess_keywords():
    assert guess_keywords(SAMPLE_TEXT) == [
        "machine learning",
        "transformers",
        "attention",
    ]


def test_guess_keywords_empty_when_absent():
    assert guess_keywords("Title\nAbstract\nbody") == []


# --- Integration tests against the real PDF ---------------------------------


@pytest.fixture(scope="module")
def analyzer() -> PaperAnalyzer:
    # Extract the text once at the boundary, then hand it to the analyzer.
    full_text, page_count = read_pdf(PDF_PATH)
    return PaperAnalyzer(
        full_text=full_text,
        page_count=page_count,
        source_path=str(PDF_PATH),
    )


def test_pdf_exists():
    assert PDF_PATH.is_file(), f"Missing test PDF: {PDF_PATH}"


def test_analyzer_page_count(analyzer: PaperAnalyzer):
    assert analyzer.page_count == 15


def test_analyzer_word_count(analyzer: PaperAnalyzer):
    assert analyzer.word_count == count_words(analyzer.full_text)
    assert analyzer.word_count > 1000


def test_analyzer_title(analyzer: PaperAnalyzer):
    assert analyzer.title == "Attention Is All You Need"


def test_analyzer_section_headings(analyzer: PaperAnalyzer):
    headings = analyzer.section_headings
    assert headings[0] == "1 Introduction"
    assert "7 Conclusion" in headings
    assert len(headings) == 22


def test_analyzer_authors(analyzer: PaperAnalyzer):
    assert analyzer.authors == EXPECTED_AUTHORS


def test_analyzer_abstract(analyzer: PaperAnalyzer):
    abstract = analyzer.abstract
    assert abstract.startswith("The dominant sequence transduction models")
    assert "Transformer" in abstract
    assert abstract.endswith("large and limited training data.")


def test_analyzer_keywords_is_empty_for_this_paper(analyzer: PaperAnalyzer):
    # This paper has no explicit keywords / index terms line.
    assert analyzer.keywords == []


def test_cached_property_returns_same_object(analyzer: PaperAnalyzer):
    # cached_property should return the identical object on repeated access.
    assert analyzer.section_headings is analyzer.section_headings
    assert analyzer.authors is analyzer.authors


def test_to_paper_builds_research_paper(analyzer: PaperAnalyzer):
    paper = analyzer.to_paper()
    assert isinstance(paper, ResearchPaper)
    assert paper.title == "Attention Is All You Need"
    assert paper.page_count == 15
    assert paper.word_count == analyzer.word_count
    assert paper.source_path == str(PDF_PATH)
    assert paper.full_text == analyzer.full_text


def test_to_paper_populates_derived_fields(analyzer: PaperAnalyzer):
    paper = analyzer.to_paper()
    assert paper.authors == EXPECTED_AUTHORS
    assert paper.abstract == analyzer.abstract
    assert paper.keywords == analyzer.keywords


def test_to_paper_leaves_remaining_fields_at_defaults(analyzer: PaperAnalyzer):
    # These fields are intentionally not derived yet.
    paper = analyzer.to_paper()
    assert paper.references == []
    assert paper.doi == ""
    assert paper.publication_year == 0
    assert paper.venue == ""


def test_analyze_convenience_wrapper():
    full_text, page_count = read_pdf(PDF_PATH)
    paper = analyze(
        full_text=full_text,
        page_count=page_count,
        source_path=str(PDF_PATH),
    )
    assert isinstance(paper, ResearchPaper)
    assert paper.title == "Attention Is All You Need"
    assert paper.page_count == 15
    assert paper.source_path == str(PDF_PATH)
