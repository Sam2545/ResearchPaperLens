"""Analysis routines that turn already-extracted text into a ResearchPaper.

This module is intentionally decoupled from PDF reading: it does not know or
care where the text came from. Callers extract the text (e.g. via
:mod:`src.pdf_reader`) and pass it in. Every field is derived by a small, pure
helper function, and each derived field is cached on :class:`PaperAnalyzer` via
:func:`functools.cached_property` so repeated access never recomputes.

The heuristics target the common single-column research-paper layout (title at
the top, an author block, an ``Abstract`` heading, then numbered sections).
They are best-effort and may need tuning for other layouts. The remaining
fields (references, doi, publication_year, venue) are still left at their
dataclass defaults.
"""

from __future__ import annotations

import re
from functools import cached_property

from src.paper import ResearchPaper

# Matches a numbered section heading line, e.g. "3.2.1 Scaled Dot-Product".
_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(\S.*)$")

# Superscript / footnote markers attached to author names and footnotes.
_AUTHOR_MARKERS = "∗*†‡§¶"
_AUTHOR_MARKER_RE = re.compile(f"[{re.escape(_AUTHOR_MARKERS)}]")

# Line prefixes that introduce a keyword list.
_KEYWORD_PREFIXES = ("keywords", "key words", "index terms")


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


def _split_camel_case(token: str) -> str:
    """Insert spaces at lower/period -> upper boundaries (``"AbC"`` -> ``"Ab C"``)."""
    return re.sub(r"(?<=[a-z.])(?=[A-Z])", " ", token)


def _abstract_index(lines: list[str]) -> int:
    """Return the index of the ``Abstract`` marker line, or ``len(lines)``."""
    for i, line in enumerate(lines):
        if line.strip().lower() == "abstract":
            return i
    return len(lines)


def guess_authors(text: str) -> list[str]:
    """Best-effort guess of author names from the header block.

    Author lines in the header carry a superscript marker (e.g. ``∗``) after
    each name, so we split those lines on the markers to separate the authors.
    Email/affiliation lines are skipped. As a fallback for extractors that glue
    ``CamelCase`` names together, each name is also split at case boundaries.
    """
    lines = [line.strip() for line in text.splitlines()]
    header = lines[: _abstract_index(lines)]

    authors: list[str] = []
    seen: set[str] = set()
    for line in header:
        if "@" in line or not _AUTHOR_MARKER_RE.search(line):
            continue
        for piece in _AUTHOR_MARKER_RE.split(line):
            name = piece.strip(" 0123456789.,")
            if len(name) < 2 or not name[0].isupper():
                continue
            name = _split_camel_case(name)
            if name not in seen:
                seen.add(name)
                authors.append(name)
    return authors


def guess_abstract(text: str) -> str:
    """Return the abstract text if an ``Abstract`` marker is present.

    Collects lines after the ``Abstract`` heading, stopping at the first
    footnote marker, numbered section heading, or keyword line.
    """
    lines = text.splitlines()
    start = _abstract_index(lines)
    if start == len(lines):
        return ""

    collected: list[str] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if (
            stripped[:1] in _AUTHOR_MARKERS
            or _SECTION_RE.match(stripped)
            or any(lowered.startswith(prefix) for prefix in _KEYWORD_PREFIXES)
        ):
            break
        collected.append(stripped)
    return " ".join(collected)


def guess_keywords(text: str) -> list[str]:
    """Return keywords if a ``Keywords``/``Index Terms`` line is present."""
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        for prefix in _KEYWORD_PREFIXES:
            if lowered.startswith(prefix):
                rest = stripped[len(prefix) :].lstrip(" :-\u2014\u2013")
                parts = re.split(r"[;,]", rest)
                return [p.strip() for p in parts if p.strip()]
    return []


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

    @cached_property
    def authors(self) -> list[str]:
        return guess_authors(self.full_text)

    @cached_property
    def abstract(self) -> str:
        return guess_abstract(self.full_text)

    @cached_property
    def keywords(self) -> list[str]:
        return guess_keywords(self.full_text)

    def to_paper(self) -> ResearchPaper:
        """Assemble the derived fields into a :class:`ResearchPaper`.

        References, doi, publication_year and venue are still left at their
        dataclass defaults for now.
        """
        return ResearchPaper(
            title=self.title,
            authors=self.authors,
            abstract=self.abstract,
            keywords=self.keywords,
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
