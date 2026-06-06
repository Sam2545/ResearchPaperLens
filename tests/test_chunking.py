"""Tests for word-based text chunking (:mod:`src.chunking`)."""

from __future__ import annotations

import pytest

from src.analyzer import count_words
from src.chunking import (
    DEFAULT_OVERLAP_WORDS,
    DEFAULT_WORDS_PER_CHUNK,
    TextChunk,
    chunk_paper,
    chunk_text,
)
from src.paper import ResearchPaper


def _words(n: int) -> str:
    """Return ``n`` distinct whitespace-separated words."""
    return " ".join(f"w{i}" for i in range(n))


def test_default_words_per_chunk_is_positive():
    assert DEFAULT_WORDS_PER_CHUNK >= 1


def test_default_overlap_words_is_valid():
    assert 0 < DEFAULT_OVERLAP_WORDS < DEFAULT_WORDS_PER_CHUNK


def test_empty_string_returns_no_chunks():
    assert chunk_text("", words_per_chunk=10) == []


def test_whitespace_only_returns_no_chunks():
    assert chunk_text("   \n\t  \n", words_per_chunk=10) == []


def test_single_word_produces_one_chunk():
    chunks = chunk_text("hello", words_per_chunk=10)
    assert len(chunks) == 1
    assert chunks[0].text == "hello"
    assert chunks[0].word_count == 1
    assert chunks[0].index == 0
    assert chunks[0].start_word == 0
    assert chunks[0].end_word == 1


def test_text_shorter_than_chunk_size_is_single_chunk():
    text = "one two three four five"
    chunks = chunk_text(text, words_per_chunk=10)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].word_count == 5


def test_text_exactly_chunk_size_is_single_chunk():
    text = _words(5)
    chunks = chunk_text(text, words_per_chunk=5)
    assert len(chunks) == 1
    assert chunks[0].word_count == 5
    assert chunks[0].text == text


def test_text_one_word_over_produces_two_chunks():
    text = _words(6)
    chunks = chunk_text(text, words_per_chunk=5)
    assert len(chunks) == 2
    assert chunks[0].word_count == 5
    assert chunks[1].word_count == 1
    assert chunks[0].index == 0
    assert chunks[1].index == 1


def test_even_split_produces_equal_sized_chunks():
    text = _words(10)
    chunks = chunk_text(text, words_per_chunk=5)
    assert len(chunks) == 2
    assert chunks[0].word_count == 5
    assert chunks[1].word_count == 5
    assert chunks[0].start_word == 0
    assert chunks[0].end_word == 5
    assert chunks[1].start_word == 5
    assert chunks[1].end_word == 10


def test_uneven_split_puts_remainder_in_final_chunk():
    text = _words(11)
    chunks = chunk_text(text, words_per_chunk=5)
    assert len(chunks) == 3
    assert [c.word_count for c in chunks] == [5, 5, 1]


def test_words_per_chunk_one_splits_every_word():
    text = "alpha beta gamma"
    chunks = chunk_text(text, words_per_chunk=1)
    assert len(chunks) == 3
    assert [c.text for c in chunks] == ["alpha", "beta", "gamma"]
    assert [c.word_count for c in chunks] == [1, 1, 1]


def test_chunk_indices_are_sequential():
    chunks = chunk_text(_words(7), words_per_chunk=3)
    assert [c.index for c in chunks] == [0, 1, 2]


def test_word_spans_are_contiguous_and_non_overlapping():
    text = _words(10)
    chunks = chunk_text(text, words_per_chunk=3)
    assert chunks[0].start_word == 0 and chunks[0].end_word == 3
    assert chunks[1].start_word == 3 and chunks[1].end_word == 6
    assert chunks[2].start_word == 6 and chunks[2].end_word == 9
    assert chunks[3].start_word == 9 and chunks[3].end_word == 10


def test_total_word_count_across_chunks_matches_source_without_overlap():
    text = _words(23)
    chunks = chunk_text(text, words_per_chunk=7, overlap_words=0)
    assert sum(c.word_count for c in chunks) == count_words(text)


def test_preserves_internal_whitespace_and_newlines():
    text = "first\n\nsecond   third"
    chunks = chunk_text(text, words_per_chunk=10)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_preserves_leading_whitespace_before_first_word():
    text = "   leading\nword"
    chunks = chunk_text(text, words_per_chunk=10)
    assert chunks[0].text == "leading\nword"


def test_strips_only_trailing_whitespace_beyond_last_word():
    text = "hello world   \n\t"
    chunks = chunk_text(text, words_per_chunk=10)
    assert chunks[0].text == "hello world"


def test_multiline_text_splits_on_word_count_not_lines():
    text = "line1 word\nline2 word\nline3 word"
    chunks = chunk_text(text, words_per_chunk=2)
    assert len(chunks) == 3
    assert chunks[0].text == "line1 word"
    assert chunks[1].text == "line2 word"
    assert chunks[2].text == "line3 word"


def test_punctuation_attached_to_words_is_not_split():
    text = "Hello, world! How are you?"
    chunks = chunk_text(text, words_per_chunk=10)
    assert chunks[0].word_count == 5
    assert "Hello," in chunks[0].text
    assert "world!" in chunks[0].text


def test_unicode_words_are_chunked_correctly():
    text = "café naïve 日本語 test"
    chunks = chunk_text(text, words_per_chunk=2)
    assert len(chunks) == 2
    assert chunks[0].text == "café naïve"
    assert chunks[1].text == "日本語 test"


def test_words_per_chunk_zero_raises():
    with pytest.raises(ValueError, match="words_per_chunk must be at least 1"):
        chunk_text("hello", words_per_chunk=0)


def test_words_per_chunk_negative_raises():
    with pytest.raises(ValueError, match="words_per_chunk must be at least 1"):
        chunk_text("hello", words_per_chunk=-3)


def test_text_chunk_is_frozen():
    chunk = TextChunk(
        index=0, text="a", word_count=1, start_word=0, end_word=1
    )
    with pytest.raises(AttributeError):
        chunk.text = "b"  # type: ignore[misc]


def test_chunk_paper_uses_full_text():
    paper = ResearchPaper(full_text="one two three four")
    chunks = chunk_paper(paper, words_per_chunk=2)
    assert len(chunks) == 2
    assert chunks[0].text == "one two"
    assert chunks[1].text == "three four"


def test_chunk_paper_empty_full_text_returns_no_chunks():
    paper = ResearchPaper(title="Empty", full_text="")
    assert chunk_paper(paper, words_per_chunk=100) == []


def test_chunk_paper_whitespace_full_text_returns_no_chunks():
    paper = ResearchPaper(full_text="  \n  ")
    assert chunk_paper(paper, words_per_chunk=100) == []


def test_chunk_paper_does_not_use_other_fields():
    paper = ResearchPaper(
        title="Title",
        abstract="abstract words here",
        full_text="body one body two",
    )
    chunks = chunk_paper(paper, words_per_chunk=10)
    assert len(chunks) == 1
    assert chunks[0].text == "body one body two"


def test_large_text_produces_expected_chunk_count():
    word_count = 10_001
    chunks = chunk_text(_words(word_count), words_per_chunk=1500, overlap_words=0)
    assert len(chunks) == 7  # ceil(10001 / 1500)
    assert chunks[-1].word_count == 10001 - 6 * 1500
    assert sum(c.word_count for c in chunks) == word_count


def test_overlap_zero_matches_non_overlapping_behavior():
    text = _words(11)
    assert chunk_text(text, words_per_chunk=5, overlap_words=0) == chunk_text(
        text, words_per_chunk=5
    )


def test_overlapping_chunks_share_words():
    text = _words(10)
    chunks = chunk_text(text, words_per_chunk=5, overlap_words=2)
    assert len(chunks) == 3
    assert chunks[0].start_word == 0 and chunks[0].end_word == 5
    assert chunks[1].start_word == 3 and chunks[1].end_word == 8
    assert chunks[2].start_word == 6 and chunks[2].end_word == 10
    assert chunks[0].end_word - chunks[1].start_word == 2
    assert chunks[1].end_word - chunks[2].start_word == 2


def test_overlap_includes_shared_text_in_chunk_strings():
    text = "a b c d e f g h i j"
    chunks = chunk_text(text, words_per_chunk=5, overlap_words=2)
    assert chunks[0].text == "a b c d e"
    assert chunks[1].text == "d e f g h"
    assert chunks[2].text == "g h i j"


def test_overlap_produces_more_chunks_than_without():
    text = _words(20)
    without = chunk_text(text, words_per_chunk=5, overlap_words=0)
    with_overlap = chunk_text(text, words_per_chunk=5, overlap_words=2)
    assert len(with_overlap) > len(without)


def test_overlap_covers_every_word_at_least_once():
    text = _words(17)
    chunks = chunk_text(text, words_per_chunk=5, overlap_words=2)
    covered = set()
    for chunk in chunks:
        covered.update(range(chunk.start_word, chunk.end_word))
    assert covered == set(range(count_words(text)))


def test_short_text_with_overlap_still_produces_one_chunk():
    text = _words(4)
    chunks = chunk_text(text, words_per_chunk=10, overlap_words=2)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_overlap_words_negative_raises():
    with pytest.raises(ValueError, match="overlap_words must be at least 0"):
        chunk_text("hello world", words_per_chunk=5, overlap_words=-1)


def test_overlap_words_equal_to_chunk_size_raises():
    with pytest.raises(ValueError, match="overlap_words must be less than words_per_chunk"):
        chunk_text("hello world", words_per_chunk=5, overlap_words=5)


def test_overlap_words_greater_than_chunk_size_raises():
    with pytest.raises(ValueError, match="overlap_words must be less than words_per_chunk"):
        chunk_text("hello world", words_per_chunk=5, overlap_words=10)


def test_chunk_paper_forwards_overlap():
    paper = ResearchPaper(full_text=_words(10))
    chunks = chunk_paper(paper, words_per_chunk=5, overlap_words=2)
    assert len(chunks) == 3
    assert chunks[1].start_word == 3
