"""Tests for :mod:`src.pdf_reader` using a real research paper PDF."""

from __future__ import annotations

from pathlib import Path

import pdfplumber
import pytest

from src.pdf_reader import extract_text, get_page_count, read_pdf

PDF_PATH = Path(__file__).resolve().parent.parent / "data" / "AttentionIsAllYouNeed.pdf"
EXPECTED_PAGE_COUNT = 15


@pytest.fixture
def opened_pdf():
    with pdfplumber.open(PDF_PATH) as pdf:
        yield pdf


def test_pdf_fixture_exists():
    assert PDF_PATH.is_file(), f"Missing test PDF: {PDF_PATH}"


def test_get_page_count(opened_pdf):
    assert get_page_count(opened_pdf) == EXPECTED_PAGE_COUNT


def test_extract_text_returns_nonempty(opened_pdf):
    text = extract_text(opened_pdf)
    assert isinstance(text, str)
    assert text.strip(), "Extracted text should not be empty"


def test_extract_text_contains_title(opened_pdf):
    text = extract_text(opened_pdf)
    assert "Attention Is All You Need" in text
    assert "Transformer" in text


def test_read_pdf_returns_text_and_page_count():
    text, page_count = read_pdf(PDF_PATH)
    assert page_count == EXPECTED_PAGE_COUNT
    assert isinstance(text, str)
    assert text.strip()


def test_read_pdf_matches_helpers():
    text, page_count = read_pdf(PDF_PATH)
    with pdfplumber.open(PDF_PATH) as pdf:
        assert text == extract_text(pdf)
    with pdfplumber.open(PDF_PATH) as pdf:
        assert page_count == get_page_count(pdf)
