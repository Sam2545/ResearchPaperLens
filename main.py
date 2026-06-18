"""Entry point: run the full PDF -> ResearchPaper -> JSON pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ollama import Client

from src.analyzer import analyze
from src.chunk_store import (
    default_chunk_store_path,
    load_chunk_store,
    save_chunk_store,
)
from src.embeddings import resolve_embed_client
from src.paper import ResearchPaper
from src.pdf_reader import read_pdf
from src.rag import ask_store, format_rag_answer
from src.retrieval import index_paper, search
from src.storage import load_paper, save_paper
from src.summarizer import DEFAULT_MODEL, summarize, summarize_full
from src.vector_index import SearchResult

DEFAULT_PDF = Path("data/AttentionIsAllYouNeed.pdf")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_TOP_K = 5
SEARCH_SNIPPET_CHARS = 240


def process_pdf(
    pdf_path: str | Path,
    output_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    summarize_paper: bool = False,
    summarize_full_paper: bool = False,
    model: str = DEFAULT_MODEL,
    *,
    index_chunks: bool = False,
    chunk_output_path: str | Path | None = None,
    embed_client: Client | None = None,
) -> tuple[Path, Path | None]:
    """Read, analyze, and persist a PDF as JSON.

    Wires the pieces together at the boundary: extract text once, analyze it,
    and save the resulting :class:`ResearchPaper`. Returns ``(paper_json_path,
    chunk_index_path)`` where the chunk path is set only when ``index_chunks``
    is true.

    If ``output_path`` is given it is used verbatim; otherwise the output is
    written to ``output_dir/<pdf-stem>.json``.

    When ``summarize_paper`` is true, the summarizer runs after analysis and
    before saving, populating the structured ``summary`` on the stored paper
    using the given Ollama cloud ``model``.

    When ``summarize_full_paper`` is true, the paper is split into sections,
    each section is summarized (with word chunking for long sections), and
    the section summaries are merged into the final ``summary``. ``summarize_paper`` and ``summarize_full_paper`` must not both
    be true.

    When ``index_chunks`` is true, hybrid section chunks are embedded and saved
    to ``chunk_output_path`` or ``output_dir/<pdf-stem>.chunks.json``.
    """
    if summarize_paper and summarize_full_paper:
        raise ValueError("summarize_paper and summarize_full_paper are mutually exclusive")

    pdf_path = Path(pdf_path)
    full_text, page_count = read_pdf(pdf_path)
    paper = analyze(
        full_text=full_text,
        page_count=page_count,
        source_path=str(pdf_path),
    )
    if summarize_full_paper:
        paper = summarize_full(paper, model=model)
    elif summarize_paper:
        paper = summarize(paper, model=model)
    if output_path is not None:
        resolved_output = Path(output_path)
    else:
        resolved_output = Path(output_dir) / f"{pdf_path.stem}.json"
    save_paper(paper, resolved_output)

    chunk_path: Path | None = None
    if index_chunks:
        client = embed_client if embed_client is not None else resolve_embed_client()
        store = index_paper(paper, client=client)
        chunk_path = (
            Path(chunk_output_path)
            if chunk_output_path is not None
            else default_chunk_store_path(paper, output_dir)
        )
        save_chunk_store(store, chunk_path)
    return resolved_output, chunk_path


def run_search(
    chunk_store_path: str | Path,
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    embed_client: Client | None = None,
) -> list[SearchResult]:
    """Load a chunk index and return ranked semantic matches for ``query``."""
    store = load_chunk_store(chunk_store_path)
    client = embed_client if embed_client is not None else resolve_embed_client()
    return search(store, query, client=client, top_k=top_k)


def format_search_results(
    results: list[SearchResult],
    *,
    query: str,
    chunk_store_path: str | Path,
    snippet_chars: int = SEARCH_SNIPPET_CHARS,
) -> str:
    """Render search hits for terminal output."""
    store_label = Path(chunk_store_path).name
    lines = [f'Query: "{query}"', f"Index: {store_label}", ""]

    if not results:
        lines.append("No matching chunks found.")
        return "\n".join(lines)

    for rank, hit in enumerate(results, start=1):
        text = hit.chunk.text.strip().replace("\n", " ")
        if len(text) > snippet_chars:
            text = text[: snippet_chars - 3].rstrip() + "..."
        lines.extend(
            [
                f"{rank}. score={hit.score:.4f}  section={hit.chunk.section_heading}",
                f"   id={hit.chunk.id}",
                f"   {text}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a research paper PDF into structured JSON.",
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        default=None,
        help="Path to the input PDF (default for analyze: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON file path (default: outputs/<pdf-stem>.json)",
    )
    summarize_group = parser.add_mutually_exclusive_group()
    summarize_group.add_argument(
        "--summarize",
        action="store_true",
        help="Run the summarizer to fill the structured summary before saving.",
    )
    summarize_group.add_argument(
        "--summarize-full",
        action="store_true",
        help=(
            "Section-aware hybrid summarization: split by section, summarize "
            "each section (word-chunking long ones), merge, then save."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Ollama cloud model for --summarize, --summarize-full, or --ask "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help=(
            "After analysis, embed hybrid section chunks and save a semantic "
            "index to outputs/<pdf-stem>.chunks.json (or --chunk-output)."
        ),
    )
    parser.add_argument(
        "--chunk-output",
        default=None,
        help="Output path for --index chunk store (default: outputs/<pdf-stem>.chunks.json)",
    )
    parser.add_argument(
        "--search",
        metavar="QUERY",
        default=None,
        help="Semantic search query against an existing chunk index (--from required).",
    )
    parser.add_argument(
        "--ask",
        metavar="QUESTION",
        default=None,
        help="RAG question answered from retrieved chunk context (--from required).",
    )
    parser.add_argument(
        "--from",
        dest="chunk_store",
        default=None,
        help="Chunk index JSON path for --search.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of search results to show (default: %(default)s).",
    )
    parser.set_defaults(_default_pdf=str(DEFAULT_PDF))
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

    _empty_list_msg = {
        "Limitations": "Not explicitly discussed in the provided text.",
    }
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
            lines.append(f"  {_empty_list_msg.get(label, '(none)')}")

    lines.append(rule)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.search is not None and args.ask is not None:
        print("error: --search and --ask are mutually exclusive", file=sys.stderr)
        raise SystemExit(2)

    if args.ask is not None:
        if not args.chunk_store:
            print("error: --ask requires --from <chunk-index.json>", file=sys.stderr)
            raise SystemExit(2)
        if args.top_k < 1:
            print("error: --top-k must be at least 1", file=sys.stderr)
            raise SystemExit(2)
        embed_client = resolve_embed_client()
        result = ask_store(
            args.chunk_store,
            args.ask,
            model=args.model,
            top_k=args.top_k,
            embed_client=embed_client,
        )
        print(format_rag_answer(result))
        return

    if args.search is not None:
        if not args.chunk_store:
            print("error: --search requires --from <chunk-index.json>", file=sys.stderr)
            raise SystemExit(2)
        if args.top_k < 1:
            print("error: --top-k must be at least 1", file=sys.stderr)
            raise SystemExit(2)
        results = run_search(args.chunk_store, args.search, top_k=args.top_k)
        print(format_search_results(results, query=args.search, chunk_store_path=args.chunk_store))
        return

    pdf_path = args.pdf if args.pdf is not None else args._default_pdf
    output_path, chunk_path = process_pdf(
        pdf_path,
        output_path=args.output,
        summarize_paper=args.summarize,
        summarize_full_paper=args.summarize_full,
        model=args.model,
        index_chunks=args.index,
        chunk_output_path=args.chunk_output,
    )
    print(f"Saved {pdf_path} -> {output_path}")
    if chunk_path is not None:
        print(f"Indexed chunks -> {chunk_path}")
    if args.summarize or args.summarize_full:
        print()
        print(format_summary(load_paper(output_path)))


if __name__ == "__main__":
    main()
