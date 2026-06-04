"""Utilities for extracting text from PDF files."""

from __future__ import annotations

from pathlib import Path

import pdfplumber
from pdfplumber.pdf import PDF

# pdfplumber infers word spaces from the horizontal gap between glyphs. Its
# default (3) is too wide for dense, justified paper text and glues words
# together; 2 recovers spaces reliably without splitting words apart.
DEFAULT_X_TOLERANCE = 2


def extract_text(pdf: PDF, x_tolerance: float = DEFAULT_X_TOLERANCE) -> str:
    """Return the full text of an opened ``pdf``."""
    pages = [page.extract_text(x_tolerance=x_tolerance) or "" for page in pdf.pages]
    return "\n".join(pages)


def get_page_count(pdf: PDF) -> int:
    """Return the number of pages in an opened ``pdf``."""
    return len(pdf.pages)


def read_pdf(pdf_path: str | Path) -> tuple[str, int]:
    """Return the full text and page count of the PDF at ``pdf_path``."""
    with pdfplumber.open(pdf_path) as pdf:
        return extract_text(pdf), get_page_count(pdf)
