"""Entry point: run the full PDF -> ResearchPaper -> JSON pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analyzer import analyze
from src.pdf_reader import read_pdf
from src.storage import save_paper
from src.summarizer import summarize

DEFAULT_PDF = Path("data/AttentionIsAllYouNeed.pdf")
DEFAULT_OUTPUT_DIR = Path("outputs")


def process_pdf(
    pdf_path: str | Path,
    output_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    summarize_paper: bool = False,
) -> Path:
    """Read, analyze, and persist a PDF as JSON.

    Wires the pieces together at the boundary: extract text once, analyze it,
    and save the resulting :class:`ResearchPaper`. Returns the output path.

    If ``output_path`` is given it is used verbatim; otherwise the output is
    written to ``output_dir/<pdf-stem>.json``.

    When ``summarize_paper`` is true, the (placeholder) summarizer runs after
    analysis and before saving, populating the ``summary``/``key_insights``
    fields on the stored paper.
    """
    pdf_path = Path(pdf_path)
    full_text, page_count = read_pdf(pdf_path)
    paper = analyze(
        full_text=full_text,
        page_count=page_count,
        source_path=str(pdf_path),
    )
    if summarize_paper:
        paper = summarize(paper)
    if output_path is not None:
        resolved_output = Path(output_path)
    else:
        resolved_output = Path(output_dir) / f"{pdf_path.stem}.json"
    save_paper(paper, resolved_output)
    return resolved_output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a research paper PDF into structured JSON.",
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        default=str(DEFAULT_PDF),
        help="Path to the input PDF (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON file path (default: outputs/<pdf-stem>.json)",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Run the summarizer to fill summary/key_insights before saving.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_path = process_pdf(
        args.pdf,
        output_path=args.output,
        summarize_paper=args.summarize,
    )
    print(f"Saved {args.pdf} -> {output_path}")


if __name__ == "__main__":
    main()
