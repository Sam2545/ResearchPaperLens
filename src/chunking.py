"""Word-based text chunking for long research papers.

Splits extracted paper text into fixed-size word chunks so downstream steps
(e.g. the summarizer) can process each piece within context limits. Chunks
slice the original string by word boundaries, preserving internal whitespace
(newlines, indentation) rather than rejoining with single spaces. Optional
overlap between consecutive chunks helps avoid cutting sentences or ideas at
chunk boundaries.

Also supports section-aware splitting: top-level numbered sections are
identified in ``full_text``, optionally preceded by an Abstract pseudo-section.
Long sections can be subdivided with :func:`chunk_section_hybrid`.

Word boundaries match :func:`~src.analyzer.count_words`: consecutive
whitespace-separated non-empty tokens (``str.split()`` semantics).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from src.analyzer import count_words
from src.paper import ResearchPaper

# Default chunk size for callers that do not override ``words_per_chunk``.
DEFAULT_WORDS_PER_CHUNK = 1500

# Recommended overlap when callers opt in (roughly 10% of the default chunk size).
DEFAULT_OVERLAP_WORDS = 150

# Sections longer than this are subdivided with fixed-word chunking.
DEFAULT_MAX_SECTION_WORDS = 1200

# Pseudo-section headings used by :func:`split_into_sections`.
ABSTRACT_SECTION_HEADING = "Abstract"
PREAMBLE_SECTION_HEADING = "Preamble"

# Headings that mark bibliographies; skipped when ``skip_references`` is true.
_REFERENCE_HEADING_RE = re.compile(
    r"^(references|bibliography|works cited)\.?\s*$",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"\S+")

# Top-level numbered headings only (``1 Introduction``, not ``3.1 Encoder``).
_TOP_LEVEL_SECTION_RE = re.compile(r"^(\d+)\s+(\S.*)$")

# Any numbered heading line (used when ``top_level_only`` is false).
_NUMBERED_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(\S.*)$")

# Mirrors :mod:`src.analyzer` heuristics for rejecting false-positive headings.
_MAX_HEADING_LINE_LEN = 60


@dataclass(frozen=True)
class TextChunk:
    """One contiguous slice of a source text, measured in words.

    ``start_word`` and ``end_word`` are half-open indices into the sequence of
    words found in the original text (``end_word`` is exclusive).
    """

    index: int
    text: str
    word_count: int
    start_word: int
    end_word: int


@dataclass(frozen=True)
class PaperSection:
    """A logical section of a paper, sliced from ``full_text`` or metadata.

    ``start_char`` / ``end_char`` are half-open character offsets into the
    paper's ``full_text`` (except the Abstract pseudo-section, which uses the
    analyzer's ``abstract`` field and has ``start_char``/``end_char`` of ``-1``).
    ``start_word`` / ``end_word`` are half-open word indices into ``full_text``.
    """

    index: int
    heading: str
    text: str
    word_count: int
    start_char: int
    end_char: int
    start_word: int
    end_word: int


@dataclass(frozen=True)
class SectionChunkPlan:
    """A section and the word chunks used to summarize it in hybrid mode."""

    section: PaperSection
    chunks: list[TextChunk]


def _word_spans(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` character spans for each word in ``text``."""
    return [(match.start(), match.end()) for match in _WORD_RE.finditer(text)]


def _word_index_at_char(full_text: str, char_pos: int) -> int:
    """Return the word index in ``full_text`` at ``char_pos``."""
    if char_pos <= 0:
        return 0
    return len(_word_spans(full_text[:char_pos]))


def _looks_like_section_heading(line: str, *, top_level_only: bool) -> str | None:
    """Return the heading text if ``line`` is a section heading, else ``None``."""
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADING_LINE_LEN:
        return None
    pattern = _TOP_LEVEL_SECTION_RE if top_level_only else _NUMBERED_SECTION_RE
    match = pattern.match(stripped)
    if not match:
        return None
    rest = match.group(2)
    if not rest[:1].isupper():
        return None
    return stripped


def _find_numbered_section_starts(
    full_text: str,
    *,
    top_level_only: bool,
) -> list[tuple[str, int]]:
    """Return ``(heading, char_offset)`` for each numbered section heading."""
    starts: list[tuple[str, int]] = []
    seen: set[str] = set()
    offset = 0
    for line in full_text.splitlines(keepends=True):
        heading = _looks_like_section_heading(line, top_level_only=top_level_only)
        if heading and heading not in seen:
            line_start = offset + line.find(line.strip())
            starts.append((heading, line_start))
            seen.add(heading)
        offset += len(line)
    return starts


def _is_reference_heading(heading: str) -> bool:
    return bool(_REFERENCE_HEADING_RE.match(heading.strip()))


def split_into_sections(
    paper: ResearchPaper,
    *,
    top_level_only: bool = True,
    include_abstract: bool = True,
    include_preamble: bool = True,
    skip_references: bool = True,
) -> list[PaperSection]:
    """Split ``paper.full_text`` into logical sections for hybrid summarization.

    Inserts an Abstract pseudo-section from ``paper.abstract`` when present.
    Optionally includes a Preamble (text before the first numbered section).
    Numbered sections are located by scanning ``full_text`` for heading lines
    using the same heuristics as :mod:`src.analyzer`.

    Returns an empty list when the paper has no usable text.
    """
    full_text = paper.full_text
    if not full_text.strip() and not (include_abstract and paper.abstract.strip()):
        return []

    sections: list[PaperSection] = []
    numbered = _find_numbered_section_starts(full_text, top_level_only=top_level_only)

    if include_abstract and paper.abstract.strip():
        sections.append(
            PaperSection(
                index=len(sections),
                heading=ABSTRACT_SECTION_HEADING,
                text=paper.abstract.strip(),
                word_count=count_words(paper.abstract),
                start_char=-1,
                end_char=-1,
                start_word=-1,
                end_word=-1,
            )
        )

    if include_preamble and numbered:
        preamble_end = numbered[0][1]
        preamble_text = full_text[:preamble_end]
        if preamble_text.strip():
            sections.append(
                PaperSection(
                    index=len(sections),
                    heading=PREAMBLE_SECTION_HEADING,
                    text=preamble_text,
                    word_count=count_words(preamble_text),
                    start_char=0,
                    end_char=preamble_end,
                    start_word=0,
                    end_word=_word_index_at_char(full_text, preamble_end),
                )
            )

    for i, (heading, start_char) in enumerate(numbered):
        if skip_references and _is_reference_heading(heading):
            continue
        end_char = numbered[i + 1][1] if i + 1 < len(numbered) else len(full_text)
        section_text = full_text[start_char:end_char]
        start_word = _word_index_at_char(full_text, start_char)
        end_word = _word_index_at_char(full_text, end_char)
        sections.append(
            PaperSection(
                index=len(sections),
                heading=heading,
                text=section_text,
                word_count=count_words(section_text),
                start_char=start_char,
                end_char=end_char,
                start_word=start_word,
                end_word=end_word,
            )
        )

    if not sections and full_text.strip():
        sections.append(
            PaperSection(
                index=0,
                heading="(document)",
                text=full_text,
                word_count=count_words(full_text),
                start_char=0,
                end_char=len(full_text),
                start_word=0,
                end_word=count_words(full_text),
            )
        )

    return sections


def chunk_text(
    text: str,
    words_per_chunk: int,
    *,
    overlap_words: int = 0,
    word_offset: int = 0,
) -> list[TextChunk]:
    """Split ``text`` into chunks of at most ``words_per_chunk`` words.

    Consecutive chunks share ``overlap_words`` words when ``overlap_words`` is
    greater than zero. Each chunk after the first starts
    ``words_per_chunk - overlap_words`` words after the previous chunk's start.

    ``word_offset`` is added to each chunk's ``start_word``/``end_word`` so
    callers can express positions relative to a larger document (e.g. a section
    slice within ``full_text``).

    Returns an empty list when ``text`` contains no words (empty, whitespace
    only, etc.). Raises :class:`ValueError` when ``words_per_chunk`` is less
    than 1, when ``overlap_words`` is negative, or when ``overlap_words`` is
    greater than or equal to ``words_per_chunk``.
    """
    if words_per_chunk < 1:
        raise ValueError(f"words_per_chunk must be at least 1, got {words_per_chunk}")
    if overlap_words < 0:
        raise ValueError(f"overlap_words must be at least 0, got {overlap_words}")
    if overlap_words >= words_per_chunk:
        raise ValueError(
            f"overlap_words must be less than words_per_chunk ({words_per_chunk}), "
            f"got {overlap_words}"
        )

    spans = _word_spans(text)
    if not spans:
        return []

    chunks: list[TextChunk] = []
    step = words_per_chunk - overlap_words
    batch_start = 0
    while batch_start < len(spans):
        batch_end = min(batch_start + words_per_chunk, len(spans))
        batch = spans[batch_start:batch_end]
        chunks.append(
            TextChunk(
                index=len(chunks),
                text=text[batch[0][0] : batch[-1][1]],
                word_count=len(batch),
                start_word=batch_start + word_offset,
                end_word=batch_end + word_offset,
            )
        )
        if batch_end >= len(spans):
            break
        batch_start += step
    return chunks


def chunk_section_hybrid(
    section: PaperSection,
    *,
    max_section_words: int = DEFAULT_MAX_SECTION_WORDS,
    words_per_chunk: int = DEFAULT_MAX_SECTION_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[TextChunk]:
    """Chunk one section: single chunk if short, else fixed-word sub-chunks."""
    if section.word_count <= max_section_words:
        if not section.text.strip():
            return []
        return [
            TextChunk(
                index=0,
                text=section.text,
                word_count=section.word_count,
                start_word=max(section.start_word, 0),
                end_word=(
                    section.end_word
                    if section.end_word >= 0
                    else section.word_count
                ),
            )
        ]

    offset = max(section.start_word, 0)
    return chunk_text(
        section.text,
        words_per_chunk,
        overlap_words=overlap_words,
        word_offset=offset,
    )


def plan_hybrid_chunks(
    paper: ResearchPaper,
    *,
    top_level_only: bool = True,
    include_abstract: bool = True,
    include_preamble: bool = True,
    skip_references: bool = True,
    max_section_words: int = DEFAULT_MAX_SECTION_WORDS,
    words_per_chunk: int = DEFAULT_MAX_SECTION_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[SectionChunkPlan]:
    """Build a hybrid chunking plan: sections, with word chunks for long ones."""
    sections = split_into_sections(
        paper,
        top_level_only=top_level_only,
        include_abstract=include_abstract,
        include_preamble=include_preamble,
        skip_references=skip_references,
    )
    return [
        SectionChunkPlan(
            section=section,
            chunks=chunk_section_hybrid(
                section,
                max_section_words=max_section_words,
                words_per_chunk=words_per_chunk,
                overlap_words=overlap_words,
            ),
        )
        for section in sections
    ]


def chunk_paper(
    paper: ResearchPaper,
    words_per_chunk: int,
    *,
    overlap_words: int = 0,
) -> list[TextChunk]:
    """Chunk ``paper.full_text``; returns ``[]`` when the paper has no text."""
    return chunk_text(paper.full_text, words_per_chunk, overlap_words=overlap_words)
