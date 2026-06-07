"""Tests for the Ollama-cloud-backed summarizer (:mod:`src.summarizer`).

These run entirely offline. Rather than threading a fake through the production
API, they patch the single network boundary -- :func:`build_cloud_client` -- so
the summarizer builds a ``FakeClient`` instead of a real one. No network calls
or API key are required. The tests verify request construction (model, schema,
message contents), response parsing, truncation, non-mutation, error handling,
and the API-key boundary in :func:`build_cloud_client`.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

import src.ollama_client as ollama_client_module
import src.summarizer as summarizer_module
from src.paper import PaperSummary, ResearchPaper
from src.summarizer import (
    _MAX_INPUT_CHARS,
    API_KEY_ENV,
    DEFAULT_MODEL,
    OllamaSummarizer,
    SummarizationError,
    build_cloud_client,
    summarize,
    summarize_full,
    summarize_text,
)

# A complete, valid structured-summary payload the model might return.
_FULL_SUMMARY = {
    "tldr": "A concise overview.",
    "problem": "The problem addressed.",
    "approach": "The method used.",
    "key_results": ["BLEU 28.4", "+3% accuracy"],
    "contributions": ["A novel architecture"],
    "limitations": ["Only tested on English"],
    "key_insights": ["insight one", "insight two"],
}


class FakeClient:
    """Stands in for ``ollama.Client``: records calls, returns canned content.

    ``chat`` returns an object shaped like a real ollama ``ChatResponse``
    (``response.message.content``) so the production code path is unchanged.
    """

    def __init__(self, content: str):
        self._content = content
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content=self._content))


class FakeClientQueue:
    """Returns a sequence of canned responses across successive ``chat`` calls."""

    def __init__(self, payloads: list):
        self._contents = [
            json.dumps(payload) if isinstance(payload, dict) else payload
            for payload in payloads
        ]
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if not self._contents:
            raise AssertionError("FakeClientQueue ran out of canned responses")
        content = self._contents.pop(0)
        return SimpleNamespace(message=SimpleNamespace(content=content))


@pytest.fixture
def fake_client(monkeypatch):
    """Patch build_cloud_client so the summarizer uses a FakeClient.

    Returns an installer: call it with a dict payload (serialized to JSON) or a
    raw string (used verbatim, for malformed-response tests). It returns the
    FakeClient so tests can inspect ``.calls``.
    """

    def _install(payload) -> FakeClient:
        content = json.dumps(payload) if isinstance(payload, dict) else payload
        client = FakeClient(content)
        monkeypatch.setattr(
            summarizer_module, "build_cloud_client", lambda **kwargs: client
        )
        return client

    return _install


@pytest.fixture
def fake_client_queue(monkeypatch):
    """Patch build_cloud_client with a :class:`FakeClientQueue`."""

    def _install(payloads) -> FakeClientQueue:
        client = FakeClientQueue(payloads)
        monkeypatch.setattr(
            summarizer_module, "build_cloud_client", lambda **kwargs: client
        )
        return client

    return _install


def _chunky_paper(word_count: int = 250) -> ResearchPaper:
    text = " ".join(f"word{i}" for i in range(word_count))
    return ResearchPaper(
        title="Chunky Paper",
        authors=["Ada Lovelace"],
        abstract="This paper studies chunked summarization.",
        keywords=["chunking", "summaries"],
        section_headings=["1 Introduction", "2 Results"],
        full_text=text,
    )


_CHUNK_PARTIAL = {
    "tldr": "Partial chunk summary.",
    "problem": "A local problem.",
    "approach": "A local approach.",
    "key_results": ["local result"],
    "contributions": ["local contribution"],
    "limitations": [],
    "key_insights": ["local insight"],
}


@pytest.fixture
def no_dotenv(monkeypatch):
    """Stop build_cloud_client from reading a real local .env during tests."""
    monkeypatch.setattr(ollama_client_module, "load_dotenv", lambda *a, **k: False)


def test_summarize_text_parses_full_structured_summary(fake_client):
    fake_client(_FULL_SUMMARY)
    summary = summarize_text("paper body")
    assert isinstance(summary, PaperSummary)
    assert summary.tldr == "A concise overview."
    assert summary.problem == "The problem addressed."
    assert summary.approach == "The method used."
    assert summary.key_results == ["BLEU 28.4", "+3% accuracy"]
    assert summary.contributions == ["A novel architecture"]
    assert summary.limitations == ["Only tested on English"]
    assert summary.key_insights == ["insight one", "insight two"]


def test_summarize_text_sends_model_schema_and_text(fake_client):
    client = fake_client(_FULL_SUMMARY)
    summarize_text("the paper body text", model="gpt-oss:20b")
    call = client.calls[0]
    assert call["model"] == "gpt-oss:20b"
    props = call["format"]["properties"]
    assert "tldr" in props
    assert "key_results" in props
    assert "key_insights" in props
    user_msg = call["messages"][-1]["content"]
    assert "the paper body text" in user_msg


def test_default_model_is_used_when_unspecified(fake_client):
    client = fake_client(_FULL_SUMMARY)
    summarize_text("body")
    assert client.calls[0]["model"] == DEFAULT_MODEL


def test_long_text_is_truncated_before_sending(fake_client):
    client = fake_client(_FULL_SUMMARY)
    summarize_text("x" * (_MAX_INPUT_CHARS + 5000))
    user_msg = client.calls[0]["messages"][-1]["content"]
    # The prompt has a wrapper, but the paper body portion must be capped.
    assert user_msg.count("x") == _MAX_INPUT_CHARS


def test_summarize_returns_research_paper_with_summary(fake_client):
    fake_client(_FULL_SUMMARY)
    paper = ResearchPaper(title="A Paper", full_text="body")
    result = summarize(paper)
    assert isinstance(result, ResearchPaper)
    assert isinstance(result.summary, PaperSummary)
    assert result.summary.tldr == "A concise overview."
    assert result.summary.key_results == ["BLEU 28.4", "+3% accuracy"]


def test_summarize_does_not_mutate_input(fake_client):
    fake_client(_FULL_SUMMARY)
    paper = ResearchPaper(title="A Paper", full_text="body")
    summarize(paper)
    assert paper.summary == PaperSummary()


def test_summarize_preserves_other_fields(fake_client):
    fake_client(_FULL_SUMMARY)
    paper = ResearchPaper(title="A Paper", authors=["Jane Doe"], full_text="body")
    result = summarize(paper)
    assert replace(result, summary=PaperSummary()) == paper


def test_invalid_json_raises_summarization_error(fake_client):
    fake_client("not json at all")
    with pytest.raises(SummarizationError):
        summarize_text("body")


def test_parses_json_wrapped_in_code_fence(fake_client):
    # Reasoning models often prepend chain-of-thought and fence the JSON.
    reasoning = "Let's think. We must return JSON.\n\n"
    fenced = "```json\n" + json.dumps(_FULL_SUMMARY) + "\n```"
    fake_client(reasoning + fenced)
    summary = summarize_text("body")
    assert summary.tldr == "A concise overview."
    assert summary.key_results == ["BLEU 28.4", "+3% accuracy"]


def test_parses_json_with_leading_prose(fake_client):
    fake_client("Here is the summary: " + json.dumps(_FULL_SUMMARY))
    summary = summarize_text("body")
    assert summary.tldr == "A concise overview."


def test_missing_string_field_defaults_to_empty_string(fake_client):
    fake_client({"key_insights": ["k"]})
    summary = summarize_text("body")
    assert summary.tldr == ""
    assert summary.key_insights == ["k"]


def test_missing_list_field_defaults_to_empty_list(fake_client):
    fake_client({"tldr": "s"})
    summary = summarize_text("body")
    assert summary.tldr == "s"
    assert summary.key_insights == []
    assert summary.key_results == []


def test_non_list_field_raises(fake_client):
    fake_client({"tldr": "s", "key_insights": "nope"})
    with pytest.raises(SummarizationError):
        summarize_text("body")


def test_non_string_field_raises(fake_client):
    fake_client({"tldr": ["should be a string"]})
    with pytest.raises(SummarizationError):
        summarize_text("body")


def test_client_is_built_once_and_cached(monkeypatch):
    built = {"count": 0}
    client = FakeClient(json.dumps(_FULL_SUMMARY))

    def factory(**kwargs):
        built["count"] += 1
        return client

    monkeypatch.setattr(summarizer_module, "build_cloud_client", factory)
    summarizer = OllamaSummarizer()
    summarizer.summarize_text("first")
    summarizer.summarize_text("second")
    assert built["count"] == 1


def test_build_cloud_client_raises_without_api_key(monkeypatch, no_dotenv):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=API_KEY_ENV):
        build_cloud_client()


def test_build_cloud_client_uses_api_key(monkeypatch, no_dotenv):
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    client = build_cloud_client()
    # The ollama Client stores auth headers; verify ours is present.
    assert any(
        "secret-key" in str(value)
        for value in getattr(client, "_client", client).headers.values()
    )


def test_summarize_full_returns_merged_summary(fake_client_queue):
    fake_client_queue([_CHUNK_PARTIAL, _CHUNK_PARTIAL, _CHUNK_PARTIAL, _FULL_SUMMARY])
    paper = _chunky_paper()
    result = summarize_full(
        paper,
        strategy="flat",
        words_per_chunk=100,
        overlap_words=10,
    )
    assert result.summary.tldr == "A concise overview."
    assert result.summary.key_results == ["BLEU 28.4", "+3% accuracy"]


def test_summarize_full_calls_once_per_chunk_plus_merge(fake_client_queue):
    client = fake_client_queue(
        [_CHUNK_PARTIAL, _CHUNK_PARTIAL, _CHUNK_PARTIAL, _FULL_SUMMARY]
    )
    summarize_full(_chunky_paper(), strategy="flat", words_per_chunk=100, overlap_words=10)
    assert len(client.calls) == 4


def test_summarize_full_chunk_prompts_contain_title_number_and_text(
    fake_client_queue,
):
    client = fake_client_queue([_CHUNK_PARTIAL, _FULL_SUMMARY])
    paper = ResearchPaper(title="My Title", full_text=" ".join(f"w{i}" for i in range(40)))
    summarize_full(paper, strategy="flat", words_per_chunk=50, overlap_words=5)
    chunk_call = client.calls[0]
    user_msg = chunk_call["messages"][-1]["content"]
    assert "You are summarizing one chunk of a longer research paper." in user_msg
    assert "Paper title:\nMy Title" in user_msg
    assert "Chunk number:\n1" in user_msg
    assert "Chunk text:\n" in user_msg
    assert "w0" in user_msg


def test_summarize_full_merge_prompt_contains_metadata_and_partials(
    fake_client_queue,
):
    client = fake_client_queue(
        [_CHUNK_PARTIAL, _CHUNK_PARTIAL, _CHUNK_PARTIAL, _FULL_SUMMARY]
    )
    paper = _chunky_paper(word_count=120)
    summarize_full(paper, strategy="flat", words_per_chunk=60, overlap_words=5)
    merge_call = client.calls[-1]
    user_msg = merge_call["messages"][-1]["content"]
    assert "You are combining partial summaries from chunks of the same paper." in user_msg
    assert "Metadata:" in user_msg
    assert "Chunky Paper" in user_msg
    assert "This paper studies chunked summarization." in user_msg
    assert "Partial summaries:" in user_msg
    assert "Partial chunk summary." in user_msg
    assert "Use the abstract and metadata as anchors." in user_msg
    assert "Do not invent limitations." in user_msg


def test_summarize_full_does_not_mutate_input(fake_client_queue):
    fake_client_queue([_CHUNK_PARTIAL, _CHUNK_PARTIAL, _FULL_SUMMARY])
    paper = ResearchPaper(title="Stable", full_text=" ".join(f"w{i}" for i in range(40)))
    summarize_full(paper, strategy="flat", words_per_chunk=25, overlap_words=5)
    assert paper.summary == PaperSummary()


def test_summarize_full_preserves_other_fields(fake_client_queue):
    fake_client_queue([_CHUNK_PARTIAL, _CHUNK_PARTIAL, _FULL_SUMMARY])
    paper = ResearchPaper(
        title="Stable",
        authors=["Jane Doe"],
        full_text=" ".join(f"w{i}" for i in range(40)),
    )
    result = summarize_full(paper, strategy="flat", words_per_chunk=25, overlap_words=5)
    assert replace(result, summary=PaperSummary()) == paper


def test_summarize_full_empty_text_skips_llm(monkeypatch):
    called = {"count": 0}

    def fail_if_called(**kwargs):
        called["count"] += 1
        raise AssertionError("LLM should not be called for empty text")

    monkeypatch.setattr(summarizer_module, "build_cloud_client", fail_if_called)
    result = summarize_full(ResearchPaper(full_text=""))
    assert result.summary == PaperSummary()
    assert called["count"] == 0


def test_summarize_full_forwards_model(fake_client_queue):
    client = fake_client_queue([_CHUNK_PARTIAL, _CHUNK_PARTIAL, _FULL_SUMMARY])
    summarize_full(
        ResearchPaper(full_text=" ".join(f"w{i}" for i in range(40))),
        strategy="flat",
        model="gpt-oss:20b",
        words_per_chunk=25,
        overlap_words=5,
    )
    assert all(call["model"] == "gpt-oss:20b" for call in client.calls)


def _sectioned_paper(*, long_section_words: int = 30) -> ResearchPaper:
    intro = "1 Introduction\n" + " ".join(f"intro{i}" for i in range(30))
    results = "6 Results\n" + " ".join(f"result{i}" for i in range(long_section_words))
    full_text = f"{intro}\n\n{results}"
    return ResearchPaper(
        title="Sectioned Paper",
        abstract="Studies chunked summarization with BLEU 99.0.",
        full_text=full_text,
        section_headings=["1 Introduction", "6 Results"],
    )


def test_summarize_full_hybrid_returns_merged_summary(fake_client_queue):
    # Abstract + 2 short sections + final merge = 4 calls
    fake_client_queue(
        [_CHUNK_PARTIAL, _CHUNK_PARTIAL, _CHUNK_PARTIAL, _FULL_SUMMARY]
    )
    result = summarize_full(_sectioned_paper(), strategy="hybrid")
    assert result.summary.tldr == "A concise overview."


def test_summarize_full_hybrid_uses_section_and_final_merge_prompts(
    fake_client_queue,
):
    client = fake_client_queue(
        [_CHUNK_PARTIAL, _CHUNK_PARTIAL, _CHUNK_PARTIAL, _FULL_SUMMARY]
    )
    summarize_full(_sectioned_paper(), strategy="hybrid")
    section_call = client.calls[1]
    final_call = client.calls[-1]
    assert "summarizing one section" in section_call["messages"][-1]["content"]
    assert "Section:\n1 Introduction" in section_call["messages"][-1]["content"]
    assert "combining section summaries" in final_call["messages"][-1]["content"]


def test_summarize_full_hybrid_subdivides_long_section(fake_client_queue):
    # Abstract + intro section + 2 result chunks + section merge + final = 6
    client = fake_client_queue(
        [
            _CHUNK_PARTIAL,
            _CHUNK_PARTIAL,
            _CHUNK_PARTIAL,
            _CHUNK_PARTIAL,
            _CHUNK_PARTIAL,
            _FULL_SUMMARY,
        ]
    )
    summarize_full(
        _sectioned_paper(long_section_words=80),
        strategy="hybrid",
        max_section_words=50,
        words_per_chunk=50,
        overlap_words=5,
    )
    chunk_call = client.calls[2]
    section_merge_call = client.calls[4]
    assert "Section:\n6 Results" in chunk_call["messages"][-1]["content"]
    assert "Chunk number:\n1 of 2" in chunk_call["messages"][-1]["content"]
    assert "combining partial chunk summaries" in section_merge_call["messages"][-1]["content"]
    assert "Section:\n6 Results" in section_merge_call["messages"][-1]["content"]
    assert len(client.calls) == 6


def test_summarize_full_invalid_strategy_raises():
    with pytest.raises(ValueError, match="strategy must be"):
        summarize_full(ResearchPaper(full_text="hello world"), strategy="nope")
