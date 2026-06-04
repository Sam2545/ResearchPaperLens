"""Placeholder summarizer: fills the ``summary``/``key_insights`` fields.

This module is a deliberate stub for a future summarization phase. It does NOT
call any LLM, model, or external service yet -- it only provides the wiring
(function signatures and return shapes) so the rest of the pipeline can depend
on a stable interface while the real implementation is built later.

Mirroring :mod:`src.analyzer`, this module is decoupled from PDF reading and
analysis: callers hand it an already-assembled :class:`~src.paper.ResearchPaper`
and get back a new one with the summary fields populated.
"""

from __future__ import annotations

from dataclasses import replace

from src.paper import ResearchPaper

# Placeholder values returned until a real summarizer is implemented. Keeping
# them as module constants makes the stub obvious and easy to assert against in
# tests.
_PLACEHOLDER_SUMMARY = ""
_PLACEHOLDER_KEY_INSIGHTS: list[str] = []


def summarize_text(full_text: str) -> tuple[str, list[str]]:
    """Return ``(summary, key_insights)`` for ``full_text``.

    Placeholder implementation: ignores the input and returns empty values.

    TODO: replace with real summarization (e.g. an LLM call) in a later phase.
    Until then this keeps a stable signature so callers can be written/tested.
    """
    # No analysis is performed yet -- the input is intentionally unused.
    return _PLACEHOLDER_SUMMARY, list(_PLACEHOLDER_KEY_INSIGHTS)


def summarize(paper: ResearchPaper) -> ResearchPaper:
    """Return a copy of ``paper`` with ``summary``/``key_insights`` filled in.

    Placeholder implementation: derives the (empty) summary fields from the
    paper's ``full_text`` via :func:`summarize_text` and returns a new
    :class:`ResearchPaper`, leaving the original untouched.
    """
    summary, key_insights = summarize_text(paper.full_text)
    return replace(paper, summary=summary, key_insights=key_insights)
