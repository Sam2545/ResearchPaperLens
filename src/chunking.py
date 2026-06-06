"""Word-based text chunking for long research papers.

Splits extracted paper text into fixed-size word chunks so downstream steps
(e.g. the summarizer) can process each piece within context limits. Chunks
slice the original string by word boundaries, preserving internal whitespace
(newlines, indentation) rather than rejoining with single spaces. Optional
overlap between consecutive chunks helps avoid cutting sentences or ideas at
chunk boundaries.

Word boundaries match :func:`~src.analyzer.count_words`: consecutive
whitespace-separated non-empty tokens (``str.split()`` semantics).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.paper import ResearchPaper

# Default chunk size for callers that do not override ``words_per_chunk``.
DEFAULT_WORDS_PER_CHUNK = 1500

# Recommended overlap when callers opt in (roughly 10% of the default chunk size).
DEFAULT_OVERLAP_WORDS = 150

_WORD_RE = re.compile(r"\S+")


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


def _word_spans(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` character spans for each word in ``text``."""
    return [(match.start(), match.end()) for match in _WORD_RE.finditer(text)]


def chunk_text(
    text: str,
    words_per_chunk: int,
    *,
    overlap_words: int = 0,
) -> list[TextChunk]:
    """Split ``text`` into chunks of at most ``words_per_chunk`` words.

    Consecutive chunks share ``overlap_words`` words when ``overlap_words`` is
    greater than zero. Each chunk after the first starts
    ``words_per_chunk - overlap_words`` words after the previous chunk's start.

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
                start_word=batch_start,
                end_word=batch_end,
            )
        )
        if batch_end >= len(spans):
            break
        batch_start += step
    return chunks


def chunk_paper(
    paper: ResearchPaper,
    words_per_chunk: int,
    *,
    overlap_words: int = 0,
) -> list[TextChunk]:
    """Chunk ``paper.full_text``; returns ``[]`` when the paper has no text."""
    return chunk_text(paper.full_text, words_per_chunk, overlap_words=overlap_words)
