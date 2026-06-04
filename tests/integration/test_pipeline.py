"""End-to-end integration test for the full PDF -> JSON pipeline.

Exercises the real wiring (`pdf_reader` -> `analyzer` -> `storage`) via
:func:`main.process_pdf` against the bundled "Attention Is All You Need" PDF,
writing to a temporary directory so the repo's ``outputs/`` stays clean.
"""

from __future__ import annotations

from pathlib import Path

from main import process_pdf
from src.paper import ResearchPaper
from src.storage import load_paper

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = PROJECT_ROOT / "data" / "AttentionIsAllYouNeed.pdf"

EXPECTED_AUTHORS = [
    "Ashish Vaswani",
    "Noam Shazeer",
    "Niki Parmar",
    "Jakob Uszkoreit",
    "Llion Jones",
    "Aidan N. Gomez",
    "Łukasz Kaiser",
    "Illia Polosukhin",
]

EXPECTED_SECTION_HEADINGS = [
    "1 Introduction",
    "2 Background",
    "3 Model Architecture",
    "3.1 Encoder and Decoder Stacks",
    "3.2 Attention",
    "3.2.1 Scaled Dot-Product Attention",
    "3.2.2 Multi-Head Attention",
    "3.2.3 Applications of Attention in our Model",
    "3.3 Position-wise Feed-Forward Networks",
    "3.4 Embeddings and Softmax",
    "3.5 Positional Encoding",
    "4 Why Self-Attention",
    "5 Training",
    "5.1 Training Data and Batching",
    "5.2 Hardware and Schedule",
    "5.3 Optimizer",
    "5.4 Regularization",
    "6 Results",
    "6.1 Machine Translation",
    "6.2 Model Variations",
    "6.3 English Constituency Parsing",
    "7 Conclusion",
]


def test_full_pipeline_writes_loadable_json(tmp_path):
    output_path = process_pdf(PDF_PATH, output_dir=tmp_path)

    # The pipeline writes JSON named after the PDF stem.
    assert output_path == tmp_path / "AttentionIsAllYouNeed.json"
    assert output_path.is_file()

    # The file round-trips back into a fully-formed ResearchPaper.
    paper = load_paper(output_path)
    assert isinstance(paper, ResearchPaper)

    # full_text and abstract are large/extraction-dependent, so guard them
    # separately and reuse the actual values in the expected object below.
    assert paper.full_text.strip()

    expected = ResearchPaper(
        title="Attention Is All You Need",
        authors=EXPECTED_AUTHORS,
        abstract=paper.abstract,
        keywords=[],
        section_headings=EXPECTED_SECTION_HEADINGS,
        page_count=15,
        word_count=6166,
        full_text=paper.full_text,
        source_path=str(PDF_PATH),
    )
    assert paper == expected
