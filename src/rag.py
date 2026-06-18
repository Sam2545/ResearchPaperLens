"""Retrieval-augmented Q&A over indexed paper chunks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ollama import Client

from src.chunk_store import ChunkStore, load_chunk_store
from src.ollama_client import build_cloud_client
from src.retrieval import search
from src.summarizer import DEFAULT_MODEL
from src.vector_index import SearchResult

_DEFAULT_TOP_K = 8
_MAX_CONTEXT_CHARS = 12_000
_MIN_CHARS_PER_CHUNK = 2_000
_MIN_USEFUL_SNIPPET = 200
_KEYWORD_RERANK_WEIGHT = 0.08
_TEMPERATURE = 0.2

_STOPWORDS = frozenset(
    {
        "how",
        "does",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "are",
        "was",
        "were",
        "about",
        "into",
        "work",
        "works",
    }
)
_TERM_RE = re.compile(r"[a-z0-9]+")

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)

_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "citations"],
}

_SYSTEM_PROMPT = (
    "You are a research assistant answering questions about an academic paper "
    "using ONLY the provided source excerpts. Respond ONLY with JSON matching "
    "the schema:\n"
    "- 'answer': a clear plain-English answer grounded in the sources.\n"
    "- 'citations': array of section headings you relied on (must match headings "
    "from the provided sources).\n"
    "If the excerpts do not support an answer, explain that in 'answer' and "
    "return an empty 'citations' array. Do not invent facts not in the sources."
)


@dataclass(frozen=True)
class RagAnswer:
    """Grounded answer to a question about an indexed paper."""

    question: str
    answer: str
    citations: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


class RagError(RuntimeError):
    """Raised when the model response cannot be turned into a RAG answer."""


def _extract_json_object(content: str) -> str:
    match = _JSON_FENCE_RE.search(content)
    if match:
        return match.group(1)
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        return content[start : end + 1]
    return content


def _parse_response(content: str) -> tuple[str, list[str]]:
    candidate = _extract_json_object(content) if isinstance(content, str) else content
    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RagError(f"Model did not return valid JSON: {content!r}") from exc
    if not isinstance(data, dict):
        raise RagError(f"Expected a JSON object, got: {content!r}")

    answer = data.get("answer", "")
    citations = data.get("citations", [])
    if not isinstance(answer, str):
        raise RagError(f"'answer' is not a string: {answer!r}")
    if not isinstance(citations, list):
        raise RagError(f"'citations' is not a list: {citations!r}")
    return answer, [str(item) for item in citations]


def _format_source_block(rank: int, hit: SearchResult) -> str:
    heading = hit.chunk.section_heading
    return (
        f"[Source {rank} | section={heading} | id={hit.chunk.id} | "
        f"score={hit.score:.4f}]\n"
        f"{hit.chunk.text.strip()}"
    )


def _truncate_block(block: str, limit: int) -> str:
    """Trim ``block`` to at most ``limit`` characters."""
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    if len(block) <= limit:
        return block
    if limit <= 3:
        return block[:limit]
    return block[: limit - 3].rstrip() + "..."


def _query_terms(query: str) -> list[str]:
    """Return salient lowercase terms from a user question."""
    terms = [
        term
        for term in _TERM_RE.findall(query.lower())
        if len(term) >= 3 and term not in _STOPWORDS
    ]
    return terms


def rerank_results(query: str, results: list[SearchResult]) -> list[SearchResult]:
    """Boost chunks whose text/headings overlap the question keywords."""
    terms = _query_terms(query)
    if not terms:
        return results

    def boosted_score(hit: SearchResult) -> float:
        haystack = f"{hit.chunk.section_heading}\n{hit.chunk.text}".lower()
        matches = sum(1 for term in terms if term in haystack)
        return hit.score + _KEYWORD_RERANK_WEIGHT * (matches / len(terms))

    ranked = sorted(results, key=boosted_score, reverse=True)
    return [
        SearchResult(chunk=hit.chunk, score=boosted_score(hit)) for hit in ranked
    ]


def build_context(
    results: list[SearchResult],
    *,
    max_chars: int = _MAX_CONTEXT_CHARS,
    max_chars_per_chunk: int | None = None,
) -> tuple[str, list[SearchResult]]:
    """Format retrieved chunks as LLM context, respecting a character budget."""
    if max_chars < 1:
        raise ValueError(f"max_chars must be at least 1, got {max_chars}")
    if not results:
        return "", []

    if max_chars_per_chunk is None:
        max_chars_per_chunk = max(
            _MIN_CHARS_PER_CHUNK,
            max_chars // min(len(results), 6),
        )

    parts: list[str] = []
    included: list[SearchResult] = []
    total = 0
    for rank, hit in enumerate(results, start=1):
        block = _truncate_block(
            _format_source_block(rank, hit),
            max_chars_per_chunk,
        )
        separator = 2 if parts else 0
        remaining = max_chars - total - separator
        if parts and remaining < _MIN_USEFUL_SNIPPET:
            break
        if len(block) > remaining:
            block = _truncate_block(block, remaining)
            if len(block) < _MIN_USEFUL_SNIPPET:
                break

        parts.append(block)
        included.append(hit)
        total += len(block) + separator
    return "\n\n".join(parts), included


def _build_user_prompt(
    *,
    paper_title: str,
    question: str,
    context: str,
) -> str:
    title = paper_title or "(untitled)"
    return (
        f"Paper title:\n{title}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"SOURCE EXCERPTS:\n{context}"
    )


class OllamaRag:
    """Answer questions about a paper using retrieved chunks and Ollama cloud."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self._client: Client | None = None

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = build_cloud_client()
        return self._client

    def complete_answer(self, messages: list[dict[str, str]]) -> tuple[str, list[str]]:
        """Send ``messages`` to the model and parse answer JSON."""
        response = self.client.chat(
            model=self.model,
            messages=messages,
            format=_RESPONSE_SCHEMA,
            options={"temperature": _TEMPERATURE},
        )
        message = response.message
        content = message.content or getattr(message, "thinking", "") or ""
        return _parse_response(content)

    def ask(
        self,
        store: ChunkStore,
        question: str,
        *,
        top_k: int = _DEFAULT_TOP_K,
        embed_client: Client | None = None,
        max_context_chars: int = _MAX_CONTEXT_CHARS,
    ) -> RagAnswer:
        """Retrieve relevant chunks and answer ``question`` from the index."""
        if not question.strip():
            raise ValueError("question must not be empty")
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")

        results = search(store, question, client=embed_client, top_k=top_k)
        if not results:
            return RagAnswer(
                question=question,
                answer="No relevant passages were found in the indexed paper.",
                citations=[],
                sources=[],
            )

        results = rerank_results(question, results)
        context, included = build_context(results, max_chars=max_context_chars)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_prompt(
                    paper_title=store.paper_title,
                    question=question,
                    context=context,
                ),
            },
        ]
        answer, citations = self.complete_answer(messages)
        return RagAnswer(
            question=question,
            answer=answer,
            citations=citations,
            sources=[hit.chunk.id for hit in included],
        )


def ask(
    store: ChunkStore,
    question: str,
    *,
    model: str = DEFAULT_MODEL,
    top_k: int = _DEFAULT_TOP_K,
    embed_client: Client | None = None,
    chat_client: Client | None = None,
    max_context_chars: int = _MAX_CONTEXT_CHARS,
) -> RagAnswer:
    """Convenience wrapper: RAG Q&A with a one-off :class:`OllamaRag`."""
    rag = OllamaRag(model=model)
    if chat_client is not None:
        rag._client = chat_client
    return rag.ask(
        store,
        question,
        top_k=top_k,
        embed_client=embed_client,
        max_context_chars=max_context_chars,
    )


def ask_store(
    chunk_store_path: str,
    question: str,
    *,
    model: str = DEFAULT_MODEL,
    top_k: int = _DEFAULT_TOP_K,
    embed_client: Client | None = None,
    chat_client: Client | None = None,
    max_context_chars: int = _MAX_CONTEXT_CHARS,
) -> RagAnswer:
    """Load a chunk index from disk and run :func:`ask`."""
    store = load_chunk_store(chunk_store_path)
    return ask(
        store,
        question,
        model=model,
        top_k=top_k,
        embed_client=embed_client,
        chat_client=chat_client,
        max_context_chars=max_context_chars,
    )


def format_rag_answer(result: RagAnswer) -> str:
    """Render a :class:`RagAnswer` for terminal output."""
    lines = [f'Question: "{result.question}"', "", result.answer.strip()]
    if result.citations:
        lines.extend(["", "Citations:"])
        lines.extend(f"  - {heading}" for heading in result.citations)
    if result.sources:
        lines.extend(["", "Sources:"])
        lines.extend(f"  - {source_id}" for source_id in result.sources)
    return "\n".join(lines)
