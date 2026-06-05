"""Entry point: run the full PDF -> ResearchPaper -> JSON pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analyzer import analyze
from src.paper import ResearchPaper
from src.pdf_reader import read_pdf
from src.storage import load_paper, save_paper
from src.summarizer import DEFAULT_MODEL, summarize

DEFAULT_PDF = Path("data/AttentionIsAllYouNeed.pdf")
DEFAULT_OUTPUT_DIR = Path("outputs")


def process_pdf(
    pdf_path: str | Path,
    output_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    summarize_paper: bool = False,
    model: str = DEFAULT_MODEL,
) -> Path:
    """Read, analyze, and persist a PDF as JSON.

    Wires the pieces together at the boundary: extract text once, analyze it,
    and save the resulting :class:`ResearchPaper`. Returns the output path.

    If ``output_path`` is given it is used verbatim; otherwise the output is
    written to ``output_dir/<pdf-stem>.json``.

    When ``summarize_paper`` is true, the summarizer runs after analysis and
    before saving, populating the structured ``summary`` on the stored paper
    using the given Ollama cloud ``model``.
    """
    pdf_path = Path(pdf_path)
    full_text, page_count = read_pdf(pdf_path)
    paper = analyze(
        full_text=full_text,
        page_count=page_count,
        source_path=str(pdf_path),
    )
    if summarize_paper:
        paper = summarize(paper, model=model)
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
        help="Run the summarizer to fill the structured summary before saving.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Ollama cloud model used for --summarize (default: %(default)s). "
            "Ignored without --summarize."
        ),
    )
    return parser.parse_args(argv)


def format_summary(paper: ResearchPaper) -> str:
    """Render a paper's structured summary as a readable terminal "card"."""
    width = 70
    rule = "=" * width
    lines = [rule]
    lines.append(paper.title or "(untitled)")
    if paper.authors:
        lines.append("Authors: " + ", ".join(paper.authors))
    lines.append("-" * width)

    summary = paper.summary
    for label, value in (
        ("TL;DR", summary.tldr),
        ("Problem", summary.problem),
        ("Approach", summary.approach),
    ):
        lines.append(f"{label}: {value}" if value else f"{label}: (none)")

    for label, items in (
        ("Key results", summary.key_results),
        ("Contributions", summary.contributions),
        ("Limitations", summary.limitations),
        ("Key insights", summary.key_insights),
    ):
        lines.append("")
        lines.append(f"{label}:")
        if items:
            lines.extend(f"  - {item}" for item in items)
        else:
            lines.append("  (none)")

    lines.append(rule)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_path = process_pdf(
        args.pdf,
        output_path=args.output,
        summarize_paper=args.summarize,
        model=args.model,
    )
    print(f"Saved {args.pdf} -> {output_path}")
    if args.summarize:
        print()
        print(format_summary(load_paper(output_path)))


if __name__ == "__main__":
    main()
