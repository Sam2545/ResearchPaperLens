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
import os
import re
from dataclasses import replace
from typing import Any

from dotenv import load_dotenv
from ollama import Client

from src.paper import PaperSummary, ResearchPaper

# Default cloud model. Other options include "gpt-oss:20b", "qwen3-coder:480b",
# and "deepseek-v3.1:671b"; see https://ollama.com/search?c=cloud.
DEFAULT_MODEL = "gpt-oss:120b"

# Ollama cloud endpoint used for direct API access.
OLLAMA_CLOUD_HOST = "https://ollama.com"

# Environment variable holding the ollama.com API key.
API_KEY_ENV = "OLLAMA_API_KEY"

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

# Fields that are list-typed in PaperSummary (validated as such on parse).
_LIST_FIELDS = ("key_results", "contributions", "limitations", "key_insights")
_STR_FIELDS = ("tldr", "problem", "approach")

# Matches a JSON object inside a ```json ... ``` (or plain ``` ... ```) fence.
# Reasoning models sometimes wrap their answer in a code fence and/or prepend
# their chain-of-thought, so we cannot assume the content is pure JSON.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


class SummarizationError(RuntimeError):
    """Raised when the model response cannot be turned into summary fields."""


def build_cloud_client(
    *,
    host: str = OLLAMA_CLOUD_HOST,
    api_key: str | None = None,
) -> "Client":
    """Build an Ollama client pointed at the cloud API.

    The API key is read from the ``api_key`` argument or, if omitted, the
    ``OLLAMA_API_KEY`` environment variable (loaded from a local ``.env`` file
    if present). Raises :class:`RuntimeError` with a clear message if no key is
    available. This is the only place that reads the environment or constructs a
    real network client.
    """
    if api_key is None:
        # Load variables from a local .env file if one exists. Existing
        # environment variables take precedence and are never overwritten.
        load_dotenv()
    key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{API_KEY_ENV} is not set. Create an API key at "
            "https://ollama.com/settings/keys and export it, e.g. "
            f"`export {API_KEY_ENV}=your_api_key`."
        )
    return Client(host=host, headers={"Authorization": f"Bearer {key}"})


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

    def summarize_text(self, full_text: str) -> PaperSummary:
        """Return a :class:`PaperSummary` derived from ``full_text``."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(full_text)},
        ]
        response = self.client.chat(
            model=self.model,
            messages=messages,
            format=_RESPONSE_SCHEMA,
            options={"temperature": _TEMPERATURE},
        )
        message = response.message
        # "Thinking" models (e.g. gpt-oss) may return the answer in the
        # ``thinking`` channel and leave ``content`` empty, so fall back to it.
        content = message.content or getattr(message, "thinking", "") or ""
        return _parse_response(content)

    def summarize(self, paper: ResearchPaper) -> ResearchPaper:
        """Return a copy of ``paper`` with its ``summary`` filled in."""
        summary = self.summarize_text(paper.full_text)
        return replace(paper, summary=summary)


def summarize_text(full_text: str, *, model: str = DEFAULT_MODEL) -> PaperSummary:
    """Convenience wrapper: summarize raw text with a one-off summarizer."""
    return OllamaSummarizer(model=model).summarize_text(full_text)


def summarize(paper: ResearchPaper, *, model: str = DEFAULT_MODEL) -> ResearchPaper:
    """Convenience wrapper: summarize a paper with a one-off summarizer."""
    return OllamaSummarizer(model=model).summarize(paper)
