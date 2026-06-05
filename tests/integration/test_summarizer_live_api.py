"""Live integration test that calls the real Ollama cloud API.

Unlike the unit tests (which patch the network boundary), this test makes a real
request to ollama.com and asserts that a populated :class:`PaperSummary` comes
back. It is skipped automatically unless an API key is available, so it never
breaks runs on machines without credentials.

A local ``.env`` is loaded so the key check matches normal runtime behavior. Run
it explicitly with::

    python -m pytest tests/integration/test_summarizer_live_api.py
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from src.paper import PaperSummary
from src.summarizer import API_KEY_ENV, summarize_text

# Pick up OLLAMA_API_KEY from a local .env if present (same as build_cloud_client).
load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.environ.get(API_KEY_ENV),
    reason=f"{API_KEY_ENV} not set; skipping live Ollama cloud API test",
)

# A small but coherent abstract so the model has real content to summarize.
SAMPLE_TEXT = (
    "Deep neural networks are powerful but often require large labeled datasets. "
    "In this work we introduce ProtoNet, a metric-learning approach for few-shot "
    "image classification. ProtoNet computes a prototype representation for each "
    "class by averaging the embeddings of its few labeled examples, then "
    "classifies query images by nearest prototype in the embedding space. On the "
    "miniImageNet benchmark, ProtoNet reaches 68.2% accuracy on 5-way 5-shot "
    "classification, outperforming prior methods while remaining simple to train. "
    "A limitation is that performance degrades when classes are visually similar."
)


def test_live_summarize_populates_paper_summary():
    # Uses the default cloud model, which reliably returns structured JSON.
    summary = summarize_text(SAMPLE_TEXT)

    assert isinstance(summary, PaperSummary)
    # The model should always produce at least a TL;DR for coherent input.
    assert summary.tldr.strip(), "expected a non-empty tldr from the live model"
    # And at least one of the structured lists should be populated.
    assert any(
        [
            summary.key_results,
            summary.contributions,
            summary.key_insights,
        ]
    ), "expected at least one populated list field from the live model"
