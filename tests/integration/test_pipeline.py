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

EXPECTED_SECTION_HEADINGS = [
    "1 Introduction",
    "2 Background",
    "3 ModelArchitecture",
    "3.1 EncoderandDecoderStacks",
    "3.2 Attention",
    "3.2.1 ScaledDot-ProductAttention",
    "3.2.2 Multi-HeadAttention",
    "3.2.3 ApplicationsofAttentioninourModel",
    "3.3 Position-wiseFeed-ForwardNetworks",
    "3.4 EmbeddingsandSoftmax",
    "3.5 PositionalEncoding",
    "4 WhySelf-Attention",
    "5 Training",
    "5.1 TrainingDataandBatching",
    "5.2 HardwareandSchedule",
    "5.3 Optimizer",
    "5.4 Regularization",
    "6 Results",
    "6.1 MachineTranslation",
    "6.2 ModelVariations",
    "6.3 EnglishConstituencyParsing",
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

    # full_text is extraction-dependent and large, so guard it separately and
    # reuse the actual value in the expected object below.
    assert paper.full_text.strip()

    expected = ResearchPaper(
        title="Attention Is All You Need",
        section_headings=EXPECTED_SECTION_HEADINGS,
        page_count=15,
        word_count=2033,
        full_text=paper.full_text,
        source_path=str(PDF_PATH),
    )
    assert paper == expected
