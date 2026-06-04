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

# Splits an author line into individual names: on commas and on any marker.
_AUTHOR_SPLIT_RE = re.compile(f"[,;{re.escape(_AUTHOR_MARKERS)}]")

# Matches an "Abstract" heading, tolerating a stray one-letter prefix (e.g. the
# "b Abstract" produced when a figure label bleeds in) and an inline colon with
# trailing text ("Abstract: ..."). group(1) captures any inline abstract text.
_ABSTRACT_RE = re.compile(r"^(?:[A-Za-z]\s+)?abstract\b\s*:?\s*(.*)$", re.IGNORECASE)

# Substrings that mark a header line as an affiliation rather than authors.
_AFFILIATION_KEYWORDS = (
    "universit",
    "institute",
    "laborator",
    "college",
    "department",
    "academy",
    "hospital",
    "affiliation",
    "research",
    "school of",
)

# Line prefixes that introduce a keyword list.
_KEYWORD_PREFIXES = ("keywords", "key words", "index terms")


def count_words(text: str) -> int:
    """Return the number of whitespace-separated words in ``text``."""
    return len(text.split())


def _find_abstract(lines: list[str]) -> tuple[int, str]:
    """Locate the abstract heading.

    Returns ``(index, inline_text)`` where ``index`` is the line of the heading
    (or ``len(lines)`` if absent) and ``inline_text`` is any abstract text that
    appeared on the heading line itself (e.g. ``"Abstract: ..."``).
    """
    for i, line in enumerate(lines):
        match = _ABSTRACT_RE.match(line.strip())
        if match:
            return i, match.group(1).strip()
    return len(lines), ""


def _looks_like_title(line: str) -> bool:
    """True if ``line`` looks like a title (multi-word, mostly capitalized)."""
    if not line or "@" in line or len(line) > 200:
        return False
    tokens = line.split()
    if len(tokens) < 2:
        return False
    alpha_tokens = [t for t in tokens if any(c.isalpha() for c in t)]
    if not alpha_tokens:
        return False
    capitalized = sum(1 for t in alpha_tokens if t[0].isupper())
    return capitalized / len(alpha_tokens) >= 0.6


def _is_title_continuation(line: str) -> bool:
    """True if ``line`` looks like a wrapped continuation of the title.

    Titles often wrap across two or three lines. A continuation has no author
    markers, emails, or digits (which signal author/affiliation lines), and its
    words are title-cased.
    """
    if not line or "@" in line or len(line) > 200:
        return False
    if any(c.isdigit() for c in line) or any(m in line for m in _AUTHOR_MARKERS):
        return False
    alpha_tokens = [t for t in line.split() if any(c.isalpha() for c in t)]
    if not alpha_tokens:
        return False
    return all(t[0].isupper() for t in alpha_tokens)


def _title_span(lines: list[str]) -> tuple[int, int]:
    """Return ``(start, end)`` line indices spanning the title (end exclusive)."""
    abstract_idx, _ = _find_abstract(lines)
    start = None
    for i in range(min(abstract_idx, len(lines))):
        if _looks_like_title(lines[i].strip()):
            start = i
            break
    if start is None:
        return 0, 0
    end = start + 1
    # Capture up to two wrapped continuation lines.
    while (
        end < abstract_idx
        and end - start < 3
        and _is_title_continuation(lines[end].strip())
    ):
        end += 1
    return start, end


def guess_title(text: str) -> str:
    """Best-effort guess of the paper title.

    Finds the first multi-word, mostly-capitalized line before the abstract and
    joins any wrapped continuation lines into the full title.
    """
    lines = text.splitlines()
    start, end = _title_span(lines)
    if start == end:
        return ""
    return " ".join(lines[i].strip() for i in range(start, end))


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


def _clean_author_name(piece: str) -> str:
    """Normalize one split author token into a name, or ``""`` if not a name.

    Strips surrounding affiliation digits/markers/punctuation, splits glued
    ``CamelCase`` names, and accepts only 2-4 token, title-cased names.
    """
    name = _split_camel_case(piece.strip(" \t0123456789.")).strip()
    tokens = name.split()
    if not (2 <= len(tokens) <= 4):
        return ""
    if not all(t[0].isupper() for t in tokens):
        return ""
    if not any(c.isalpha() for c in name):
        return ""
    return name


def guess_authors(text: str) -> list[str]:
    """Best-effort guess of author names from the header block.

    Looks only at lines between the title and the abstract, skipping emails and
    affiliation lines. Author lines are split on commas and superscript markers,
    and each piece is normalized into a plausible name.
    """
    lines = [line.strip() for line in text.splitlines()]
    abstract_idx, _ = _find_abstract(lines)
    _, title_end = _title_span(lines)

    authors: list[str] = []
    seen: set[str] = set()
    for line in lines[title_end:abstract_idx]:
        if not line or "@" in line or line[0].isdigit():
            continue
        lowered = line.lower()
        if any(keyword in lowered for keyword in _AFFILIATION_KEYWORDS):
            continue
        for piece in _AUTHOR_SPLIT_RE.split(line):
            name = _clean_author_name(piece)
            if name and name not in seen:
                seen.add(name)
                authors.append(name)
    return authors


def guess_abstract(text: str) -> str:
    """Return the abstract text if an ``Abstract`` heading is present.

    Collects lines after the heading (plus any inline text on the heading line),
    stopping at the first footnote marker, numbered section heading, or keyword
    line.
    """
    lines = text.splitlines()
    start, inline = _find_abstract(lines)
    if start == len(lines):
        return ""

    collected: list[str] = []
    if inline:
        collected.append(inline)
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
