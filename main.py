"""Entry point: run the full PDF -> ResearchPaper -> JSON pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

from src.analyzer import analyze
from src.pdf_reader import read_pdf
from src.storage import save_paper

DEFAULT_PDF = Path("data/AttentionIsAllYouNeed.pdf")
DEFAULT_OUTPUT_DIR = Path("outputs")


def process_pdf(
    pdf_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Read, analyze, and persist a PDF as JSON.

    Wires the pieces together at the boundary: extract text once, analyze it,
    and save the resulting :class:`ResearchPaper`. Returns the output path.
    """
    pdf_path = Path(pdf_path)
    full_text, page_count = read_pdf(pdf_path)
    paper = analyze(
        full_text=full_text,
        page_count=page_count,
        source_path=str(pdf_path),
    )
    output_path = Path(output_dir) / f"{pdf_path.stem}.json"
    save_paper(paper, output_path)
    return output_path


def main() -> None:
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF
    output_path = process_pdf(pdf_path)
    print(f"Saved {pdf_path} -> {output_path}")


if __name__ == "__main__":
    main()
