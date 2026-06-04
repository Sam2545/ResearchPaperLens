"""Regression checks for title/authors/page-count across several real papers.

These guard the layout-handling heuristics (multi-line titles, comma/marker
author parsing, affiliation skipping) against a range of real paper formats.
Abstracts are intentionally not asserted here: two-column papers still produce
garbled abstracts (column reconstruction is out of scope for now).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.analyzer import analyze
from src.pdf_reader import read_pdf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CASES = [
    pytest.param(
        "ResearchPaper2.pdf",
        "Beyond Prompt-Based Planning: MCP-Native Graph Planning-based "
        "Biomedical Agent System",
        [
            "Zhangtianyi Chen",
            "Florensia Widjaja",
            "Wufei Dai",
            "Xiangjun Zhang",
            "Yuhao Shen",
            "Juexiao Zhou",
        ],
        34,
        id="ResearchPaper2",
    ),
    pytest.param(
        "ResearchPaper3.pdf",
        "Online Skill Learning for Web Agents via State-Grounded Dynamic Retrieval",
        [
            "Jiaxi Li",
            "Ke Deng",
            "Yun Wang",
            "Jingyuan Huang",
            "Yucheng Shi",
            "Qiaoyu Tan",
            "Jin Lu",
            "Ninghao Liu",
        ],
        17,
        id="ResearchPaper3",
    ),
    pytest.param(
        "ResearchPaper4.pdf",
        "Stumbling Into AI Emotional Dependence: How Routine AI Interactions "
        "Reshape Human Connection",
        ["Yaoxi Shi", "Cathy Mengying Fang", "Pattie Maes", "Amit Goldenberg"],
        11,
        id="ResearchPaper4",
    ),
]


@pytest.mark.parametrize("filename,expected_title,expected_authors,expected_pages", CASES)
def test_sample_paper_fields(filename, expected_title, expected_authors, expected_pages):
    pdf_path = DATA_DIR / filename
    if not pdf_path.is_file():
        pytest.skip(f"Sample PDF not present: {filename}")

    full_text, page_count = read_pdf(pdf_path)
    paper = analyze(full_text=full_text, page_count=page_count, source_path=str(pdf_path))

    assert paper.page_count == expected_pages
    assert paper.title == expected_title
    assert paper.authors == expected_authors
