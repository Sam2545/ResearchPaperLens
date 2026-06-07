"""LLM summarizer backed by Ollama cloud models.

Generates a structured :class:`~src.paper.PaperSummary` (tldr, problem,
approach, key results/metrics, contributions, limitations, key insights) for a
research paper using an Ollama cloud model (https://docs.ollama.com/cloud). The
model is selectable so callers (e.g. the CLI) can switch between cloud models.

Design notes:
- All network/auth lives in one place, :func:`build_cloud_client`, which the
  summarizer calls lazily. Keeping that boundary isolated is good structure and
  also gives tests a single seam to patch so they never hit the network.
- Structured outputs: the request asks Ollama to return JSON matching a fixed
  schema, so parsing the response is reliable rather than scraping prose.

Mirroring :mod:`src.analyzer`, this module is decoupled from PDF reading and
analysis: callers hand it an already-assembled :class:`~src.paper.ResearchPaper`
and get back a new one with the summary fields populated.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, replace
from typing import Any

from ollama import Client

from src.chunking import (
    DEFAULT_MAX_SECTION_WORDS,
    DEFAULT_OVERLAP_WORDS,
    DEFAULT_WORDS_PER_CHUNK,
    PaperSection,
    SectionChunkPlan,
    TextChunk,
    chunk_paper,
    plan_hybrid_chunks,
)
from src.paper import PaperSummary, ResearchPaper
from src.ollama_client import API_KEY_ENV, OLLAMA_CLOUD_HOST, build_cloud_client

# Default cloud model. Other options include "gpt-oss:20b", "qwen3-coder:480b",
# and "deepseek-v3.1:671b"; see https://ollama.com/search?c=cloud.
DEFAULT_MODEL = "gpt-oss:120b"

# Cap on how much paper text we send to the model. Research papers can be very
# long; sending the whole thing wastes tokens and can exceed context limits.
_MAX_INPUT_CHARS = 12_000

# Sampling temperature. Low for stable, repeatable summaries.
_TEMPERATURE = 0.2

_STR_LIST = {"type": "array", "items": {"type": "string"}}

# JSON schema describing the structured "summary card" we want back from the
# model. Mirrors the fields of :class:`~src.paper.PaperSummary`.
_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tldr": {"type": "string"},
        "problem": {"type": "string"},
        "approach": {"type": "string"},
        "key_results": _STR_LIST,
        "contributions": _STR_LIST,
        "limitations": _STR_LIST,
        "key_insights": _STR_LIST,
    },
    "required": [
        "tldr",
        "problem",
        "approach",
        "key_results",
        "contributions",
        "limitations",
        "key_insights",
    ],
}

_SYSTEM_PROMPT = (
    "You are a research assistant. Summarize academic papers accurately and "
    "concisely so a reader can quickly understand the paper. Respond ONLY with "
    "JSON matching the requested schema:\n"
    "- 'tldr': 1-2 plain-English sentences on what the paper is about.\n"
    "- 'problem': the problem or motivation it addresses.\n"
    "- 'approach': the method or technique used.\n"
    "- 'key_results': array of main findings; include concrete quantitative "
    "metrics (e.g. scores, accuracy, speedups) when stated.\n"
    "- 'contributions': array of the paper's novel contributions.\n"
    "- 'limitations': array of caveats or weaknesses.\n"
    "- 'key_insights': array of the most important standalone takeaways.\n"
    "Use empty strings/arrays for fields the text does not support. Do not "
    "invent details not supported by the text."
)

_SCHEMA_FIELD_DESCRIPTIONS = (
    "- 'tldr': 1-2 plain-English sentences on what the paper is about.\n"
    "- 'problem': the problem or motivation it addresses.\n"
    "- 'approach': the method or technique used.\n"
    "- 'key_results': array of main findings; include concrete quantitative "
    "metrics (e.g. scores, accuracy, speedups) when stated.\n"
    "- 'contributions': array of the paper's novel contributions.\n"
    "- 'limitations': array of caveats or weaknesses.\n"
    "- 'key_insights': array of the most important standalone takeaways."
)

_CHUNK_SYSTEM_PROMPT = (
    "You are a research assistant summarizing part of an academic paper. "
    "Respond ONLY with JSON matching the requested schema:\n"
    f"{_SCHEMA_FIELD_DESCRIPTIONS}\n"
    "Use empty strings/arrays for fields the chunk does not support. Do not "
    "invent details not supported by the chunk text."
)

_SECTION_SYSTEM_PROMPT = (
    "You are a research assistant summarizing one section of an academic "
    "paper. Respond ONLY with JSON matching the requested schema:\n"
    f"{_SCHEMA_FIELD_DESCRIPTIONS}\n"
    "Use empty strings/arrays for fields the section does not support. Do not "
    "invent details not supported by the section text."
)

_SECTION_MERGE_SYSTEM_PROMPT = (
    "You are a research assistant combining partial chunk summaries from the "
    "same paper section into one section summary. Respond ONLY with JSON "
    "matching the requested schema:\n"
    f"{_SCHEMA_FIELD_DESCRIPTIONS}\n"
    "Use empty strings/arrays when the combined evidence does not support a "
    "field. Do not invent details not supported by the partial summaries."
)

_FLAT_MERGE_SYSTEM_PROMPT = (
    "You are a research assistant combining partial summaries of the same "
    "paper into one coherent summary. Respond ONLY with JSON matching the "
    "requested schema:\n"
    f"{_SCHEMA_FIELD_DESCRIPTIONS}\n"
    "Use empty strings/arrays when the combined evidence does not support a "
    "field. Do not invent details not supported by the partial summaries or "
    "metadata."
)

_FINAL_MERGE_SYSTEM_PROMPT = (
    "You are a research assistant combining section-level summaries of the "
    "same paper into one coherent summary. Respond ONLY with JSON matching "
    "the requested schema:\n"
    f"{_SCHEMA_FIELD_DESCRIPTIONS}\n"
    "Use empty strings/arrays when the combined evidence does not support a "
    "field. Do not invent details not supported by the section summaries or "
    "metadata."
)

# Fields that are list-typed in PaperSummary (validated as such on parse).
_LIST_FIELDS = ("key_results", "contributions", "limitations", "key_insights")
_STR_FIELDS = ("tldr", "problem", "approach")

# Matches a JSON object inside a ```json ... ``` (or plain ``` ... ```) fence.
# Reasoning models sometimes wrap their answer in a code fence and/or prepend
# their chain-of-thought, so we cannot assume the content is pure JSON.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


class SummarizationError(RuntimeError):
    """Raised when the model response cannot be turned into summary fields."""


def _build_user_prompt(full_text: str) -> str:
    """Build the user message, truncating overly long input."""
    text = full_text.strip()
    if len(text) > _MAX_INPUT_CHARS:
        text = text[:_MAX_INPUT_CHARS]
    return (
        "Summarize the following research paper into the structured JSON "
        "summary described by the schema.\n\n"
        f"PAPER TEXT:\n{text}"
    )


def _build_chunk_user_prompt(
    paper: ResearchPaper,
    chunk: TextChunk,
    *,
    section_heading: str | None = None,
    chunk_count: int | None = None,
) -> str:
    """Build the user message for summarizing a single chunk."""
    title = paper.title or "(untitled)"
    lines = [
        "You are summarizing one chunk of a longer research paper.",
        "",
        f"Paper title:\n{title}",
    ]
    if section_heading:
        lines.extend(["", f"Section:\n{section_heading}"])
    chunk_label = str(chunk.index + 1)
    if chunk_count is not None:
        chunk_label = f"{chunk.index + 1} of {chunk_count}"
    lines.extend(
        [
            "",
            f"Chunk number:\n{chunk_label}",
            "",
            f"Chunk text:\n{chunk.text}",
            "",
            "Return structured JSON using the required schema.",
            "Only include information supported by this chunk.",
            "If a field is not present in this chunk, use an empty string or "
            "empty list.",
        ]
    )
    return "\n".join(lines)


def _build_section_user_prompt(paper: ResearchPaper, section: PaperSection) -> str:
    """Build the user message for summarizing a whole section."""
    title = paper.title or "(untitled)"
    return (
        "You are summarizing one section of a longer research paper.\n\n"
        f"Paper title:\n{title}\n\n"
        f"Section:\n{section.heading}\n\n"
        f"Section text:\n{section.text}\n\n"
        "Return structured JSON using the required schema.\n"
        "Only include information supported by this section.\n"
        "If a field is not present in this section, use an empty string or "
        "empty list."
    )


def _build_section_merge_user_prompt(
    paper: ResearchPaper,
    section: PaperSection,
    partial_summaries: list[PaperSummary],
) -> str:
    """Build the user message for merging chunk summaries within one section."""
    partials_json = json.dumps(
        [asdict(summary) for summary in partial_summaries],
        indent=2,
        ensure_ascii=False,
    )
    title = paper.title or "(untitled)"
    return (
        "You are combining partial chunk summaries from the same section of "
        "a paper.\n\n"
        f"Paper title:\n{title}\n\n"
        f"Section:\n{section.heading}\n\n"
        f"Partial chunk summaries:\n{partials_json}\n\n"
        "Combine these into one section summary.\n"
        "Remove duplicates.\n"
        "Prefer specific results over vague statements.\n"
        "Do not invent limitations.\n"
        "Return one structured JSON object."
    )


def _format_paper_metadata(paper: ResearchPaper) -> str:
    """Render paper metadata for merge prompts."""
    authors = ", ".join(paper.authors) if paper.authors else "(none)"
    keywords = ", ".join(paper.keywords) if paper.keywords else "(none)"
    sections = ", ".join(paper.section_headings) if paper.section_headings else "(none)"
    abstract = paper.abstract.strip() or "(none)"
    return (
        f"Title: {paper.title or '(untitled)'}\n"
        f"Authors: {authors}\n"
        f"Abstract: {abstract}\n"
        f"Keywords: {keywords}\n"
        f"Section headings: {sections}"
    )


def _build_merge_user_prompt(
    paper: ResearchPaper,
    partial_summaries: list[PaperSummary],
    *,
    partials_label: str = "Partial summaries",
) -> str:
    """Build the user message for merging summaries into the final paper summary."""
    partials_json = json.dumps(
        [asdict(summary) for summary in partial_summaries],
        indent=2,
        ensure_ascii=False,
    )
    return (
        "You are combining partial summaries from chunks of the same paper.\n\n"
        f"Metadata:\n{_format_paper_metadata(paper)}\n\n"
        f"{partials_label}:\n{partials_json}\n\n"
        "Use the abstract and metadata as anchors.\n"
        "Remove duplicates.\n"
        "Prefer specific results over vague statements.\n"
        "Do not invent limitations.\n"
        "Return one final structured JSON object."
    )


def _build_final_merge_user_prompt(
    paper: ResearchPaper, section_summaries: list[PaperSummary]
) -> str:
    """Build the user message for the final paper-level merge."""
    partials_json = json.dumps(
        [asdict(summary) for summary in section_summaries],
        indent=2,
        ensure_ascii=False,
    )
    return (
        "You are combining section summaries from the same paper.\n\n"
        f"Metadata:\n{_format_paper_metadata(paper)}\n\n"
        f"Section summaries:\n{partials_json}\n\n"
        "Use the abstract and metadata as anchors.\n"
        "Remove duplicates.\n"
        "Prefer specific results over vague statements.\n"
        "Do not invent limitations.\n"
        "Return one final structured JSON object."
    )


def _extract_json_object(content: str) -> str:
    """Pull the JSON object out of a possibly-noisy model response.

    Handles three cases: pure JSON, JSON wrapped in a ``` code fence, and JSON
    preceded by reasoning/prose. Falls back to the first ``{`` ... last ``}``
    span. Returns the original content if nothing object-like is found (so the
    caller's ``json.loads`` raises a clear error).
    """
    match = _JSON_FENCE_RE.search(content)
    if match:
        return match.group(1)
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end > start:
        return content[start : end + 1]
    return content


def _parse_response(content: str) -> PaperSummary:
    """Parse the model's JSON content into a :class:`PaperSummary`."""
    candidate = _extract_json_object(content) if isinstance(content, str) else content
    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, TypeError) as exc:
        raise SummarizationError(
            f"Model did not return valid JSON: {content!r}"
        ) from exc
    if not isinstance(data, dict):
        raise SummarizationError(f"Expected a JSON object, got: {content!r}")

    parsed: dict[str, Any] = {}
    for name in _STR_FIELDS:
        value = data.get(name, "")
        if not isinstance(value, str):
            raise SummarizationError(f"{name!r} is not a string: {value!r}")
        parsed[name] = value
    for name in _LIST_FIELDS:
        value = data.get(name, [])
        if not isinstance(value, list):
            raise SummarizationError(f"{name!r} is not a list: {value!r}")
        parsed[name] = [str(item) for item in value]

    return PaperSummary(**parsed)


def _summary_has_content(summary: PaperSummary) -> bool:
    """True when ``summary`` has at least one non-empty field."""
    if summary.tldr.strip() or summary.problem.strip() or summary.approach.strip():
        return True
    return bool(
        summary.key_results
        or summary.contributions
        or summary.limitations
        or summary.key_insights
    )


class OllamaSummarizer:
    """Summarize papers with an Ollama cloud model.

    The cloud client is built lazily on first use via :func:`build_cloud_client`
    and cached for the lifetime of the instance.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self._client: "Client | None" = None

    @property
    def client(self) -> "Client":
        if self._client is None:
            self._client = build_cloud_client()
        return self._client

    def _complete_summary(self, messages: list[dict[str, str]]) -> PaperSummary:
        """Send ``messages`` to the model and parse a :class:`PaperSummary`."""
        response = self.client.chat(
            model=self.model,
            messages=messages,
            format=_RESPONSE_SCHEMA,
            options={"temperature": _TEMPERATURE},
        )
        message = response.message
        content = message.content or getattr(message, "thinking", "") or ""
        return _parse_response(content)

    def summarize_text(self, full_text: str) -> PaperSummary:
        """Return a :class:`PaperSummary` derived from ``full_text``."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(full_text)},
        ]
        return self._complete_summary(messages)

    def summarize_chunk(
        self,
        paper: ResearchPaper,
        chunk: TextChunk,
        *,
        section_heading: str | None = None,
        chunk_count: int | None = None,
    ) -> PaperSummary:
        """Return a partial :class:`PaperSummary` for one chunk."""
        messages = [
            {"role": "system", "content": _CHUNK_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_chunk_user_prompt(
                    paper,
                    chunk,
                    section_heading=section_heading,
                    chunk_count=chunk_count,
                ),
            },
        ]
        return self._complete_summary(messages)

    def summarize_section(
        self, paper: ResearchPaper, section: PaperSection
    ) -> PaperSummary:
        """Return a partial :class:`PaperSummary` for one whole section."""
        messages = [
            {"role": "system", "content": _SECTION_SYSTEM_PROMPT},
            {"role": "user", "content": _build_section_user_prompt(paper, section)},
        ]
        return self._complete_summary(messages)

    def merge_section_summaries(
        self,
        paper: ResearchPaper,
        section: PaperSection,
        partial_summaries: list[PaperSummary],
    ) -> PaperSummary:
        """Merge chunk-level summaries into one section summary."""
        messages = [
            {"role": "system", "content": _SECTION_MERGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_section_merge_user_prompt(
                    paper, section, partial_summaries
                ),
            },
        ]
        return self._complete_summary(messages)

    def merge_summaries(
        self, paper: ResearchPaper, partial_summaries: list[PaperSummary]
    ) -> PaperSummary:
        """Merge flat chunk-level summaries into one final :class:`PaperSummary`."""
        messages = [
            {"role": "system", "content": _FLAT_MERGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_merge_user_prompt(paper, partial_summaries),
            },
        ]
        return self._complete_summary(messages)

    def merge_section_summaries_into_final(
        self, paper: ResearchPaper, section_summaries: list[PaperSummary]
    ) -> PaperSummary:
        """Merge section-level summaries into the final paper summary."""
        messages = [
            {"role": "system", "content": _FINAL_MERGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_final_merge_user_prompt(paper, section_summaries),
            },
        ]
        return self._complete_summary(messages)

    def _summarize_section_plan(
        self, paper: ResearchPaper, plan: SectionChunkPlan
    ) -> PaperSummary:
        """Summarize one section, subdividing with word chunks when needed."""
        section = plan.section
        chunks = plan.chunks
        if not chunks:
            return PaperSummary()
        if len(chunks) == 1:
            return self.summarize_section(paper, section)
        partials = [
            self.summarize_chunk(
                paper,
                chunk,
                section_heading=section.heading,
                chunk_count=len(chunks),
            )
            for chunk in chunks
        ]
        return self.merge_section_summaries(paper, section, partials)

    def summarize(self, paper: ResearchPaper) -> ResearchPaper:
        """Return a copy of ``paper`` with its ``summary`` filled in."""
        summary = self.summarize_text(paper.full_text)
        return replace(paper, summary=summary)

    def summarize_full_flat(
        self,
        paper: ResearchPaper,
        *,
        words_per_chunk: int = DEFAULT_WORDS_PER_CHUNK,
        overlap_words: int = DEFAULT_OVERLAP_WORDS,
    ) -> ResearchPaper:
        """Summarize via flat word chunking, then merge partial summaries."""
        chunks = chunk_paper(
            paper,
            words_per_chunk,
            overlap_words=overlap_words,
        )
        if not chunks:
            return replace(paper, summary=PaperSummary())

        partial_summaries = [
            self.summarize_chunk(paper, chunk) for chunk in chunks
        ]
        summary = self.merge_summaries(paper, partial_summaries)
        return replace(paper, summary=summary)

    def summarize_full_hybrid(
        self,
        paper: ResearchPaper,
        *,
        max_section_words: int = DEFAULT_MAX_SECTION_WORDS,
        words_per_chunk: int = DEFAULT_MAX_SECTION_WORDS,
        overlap_words: int = DEFAULT_OVERLAP_WORDS,
    ) -> ResearchPaper:
        """Summarize via section-aware hybrid chunking and hierarchical merge."""
        plans = plan_hybrid_chunks(
            paper,
            max_section_words=max_section_words,
            words_per_chunk=words_per_chunk,
            overlap_words=overlap_words,
        )
        if not plans:
            return replace(paper, summary=PaperSummary())

        section_summaries = [
            self._summarize_section_plan(paper, plan) for plan in plans
        ]
        section_summaries = [s for s in section_summaries if _summary_has_content(s)]
        if not section_summaries:
            return replace(paper, summary=PaperSummary())

        summary = self.merge_section_summaries_into_final(paper, section_summaries)
        return replace(paper, summary=summary)

    def summarize_full(
        self,
        paper: ResearchPaper,
        *,
        strategy: str = "hybrid",
        words_per_chunk: int = DEFAULT_WORDS_PER_CHUNK,
        overlap_words: int = DEFAULT_OVERLAP_WORDS,
        max_section_words: int = DEFAULT_MAX_SECTION_WORDS,
    ) -> ResearchPaper:
        """Summarize the full paper using ``hybrid`` or ``flat`` chunking."""
        if strategy == "flat":
            return self.summarize_full_flat(
                paper,
                words_per_chunk=words_per_chunk,
                overlap_words=overlap_words,
            )
        if strategy == "hybrid":
            return self.summarize_full_hybrid(
                paper,
                max_section_words=max_section_words,
                words_per_chunk=words_per_chunk,
                overlap_words=overlap_words,
            )
        raise ValueError(
            f"strategy must be 'hybrid' or 'flat', got {strategy!r}"
        )


def summarize_text(full_text: str, *, model: str = DEFAULT_MODEL) -> PaperSummary:
    """Convenience wrapper: summarize raw text with a one-off summarizer."""
    return OllamaSummarizer(model=model).summarize_text(full_text)


def summarize(paper: ResearchPaper, *, model: str = DEFAULT_MODEL) -> ResearchPaper:
    """Convenience wrapper: summarize a paper with a one-off summarizer."""
    return OllamaSummarizer(model=model).summarize(paper)


def summarize_full(
    paper: ResearchPaper,
    *,
    model: str = DEFAULT_MODEL,
    strategy: str = "hybrid",
    words_per_chunk: int = DEFAULT_WORDS_PER_CHUNK,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
    max_section_words: int = DEFAULT_MAX_SECTION_WORDS,
) -> ResearchPaper:
    """Convenience wrapper: chunk, summarize, and merge with a one-off summarizer."""
    return OllamaSummarizer(model=model).summarize_full(
        paper,
        strategy=strategy,
        words_per_chunk=words_per_chunk,
        overlap_words=overlap_words,
        max_section_words=max_section_words,
    )
