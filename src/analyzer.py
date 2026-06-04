"""Analysis routines that turn already-extracted text into a ResearchPaper.

This module is intentionally decoupled from PDF reading: it does not know or
care where the text came from. Callers extract the text (e.g. via
:mod:`src.pdf_reader`) and pass it in. Every field is derived by a small, pure
helper function, and each derived field is cached on :class:`PaperAnalyzer` via
:func:`functools.cached_property` so repeated access never recomputes.

Only the fields we can derive reliably are populated for now (title, section
headings, word count). Richer fields (authors, abstract, keywords, references,
doi, publication_year, venue) are intentionally left at their dataclass
defaults and can be added later.
"""

from __future__ import annotations

import re
from functools import cached_property

from src.paper import ResearchPaper

# Matches a numbered section heading line, e.g. "3.2.1 Scaled Dot-Product".
_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(\S.*)$")


def count_words(text: str) -> int:
    """Return the number of whitespace-separated words in ``text``."""
    return len(text.split())


def guess_title(text: str) -> str:
    """Best-effort guess of the paper title.

    Returns the first multi-word, mostly-capitalized line that appears before
    the abstract. Lines extracted as a single glued token (a common PDF
    artifact for body paragraphs) are skipped.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("abstract"):
            break
        if "@" in stripped or len(stripped) > 100:
            continue
        tokens = stripped.split()
        if len(tokens) < 2:
            continue
        alpha_tokens = [t for t in tokens if any(c.isalpha() for c in t)]
        if not alpha_tokens:
            continue
        capitalized = sum(1 for t in alpha_tokens if t[0].isupper())
        if capitalized / len(alpha_tokens) >= 0.6:
            return stripped
    return ""


def guess_section_headings(text: str) -> list[str]:
    """Return numbered section headings (e.g. ``"3.1 Encoder and Decoder"``)."""
    headings: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) > 60:
            continue
        match = _SECTION_RE.match(stripped)
        if not match:
            continue
        rest = match.group(2)
        if rest[:1].isupper() and stripped not in seen:
            seen.add(stripped)
            headings.append(stripped)
    return headings


class PaperAnalyzer:
    """Derive structured :class:`ResearchPaper` fields from extracted text.

    The analyzer accepts text that has already been extracted (it never reads a
    PDF itself). Each field is a cached property, so it is computed at most once
    per analyzer instance.
    """

    def __init__(
        self,
        full_text: str,
        page_count: int = 0,
        source_path: str = "",
    ) -> None:
        self.full_text = full_text
        self.page_count = page_count
        self.source_path = source_path

    @cached_property
    def word_count(self) -> int:
        return count_words(self.full_text)

    @cached_property
    def title(self) -> str:
        return guess_title(self.full_text)

    @cached_property
    def section_headings(self) -> list[str]:
        return guess_section_headings(self.full_text)

    def to_paper(self) -> ResearchPaper:
        """Assemble the derived fields into a :class:`ResearchPaper`.

        Authors, abstract, keywords, references, doi, publication_year and venue
        are intentionally left at their dataclass defaults for now.
        """
        return ResearchPaper(
            title=self.title,
            section_headings=self.section_headings,
            page_count=self.page_count,
            word_count=self.word_count,
            full_text=self.full_text,
            source_path=self.source_path,
        )


def analyze(
    full_text: str,
    page_count: int = 0,
    source_path: str = "",
) -> ResearchPaper:
    """Analyze already-extracted text and return a :class:`ResearchPaper`."""
    return PaperAnalyzer(
        full_text=full_text,
        page_count=page_count,
        source_path=source_path,
    ).to_paper()
