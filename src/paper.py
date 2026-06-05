"""Data model representing a research paper."""

from dataclasses import dataclass, field


@dataclass
class PaperSummary:
    """Structured, LLM-generated summary of a paper (a "summary card").

    Complements the metadata fields on :class:`ResearchPaper` (``title``,
    ``authors``, etc.), which are derived separately by the analyzer. All fields
    default to empty so an un-summarized paper has a valid, empty summary.
    """

    tldr: str = ""
    problem: str = ""
    approach: str = ""
    key_results: list[str] = field(default_factory=list)
    contributions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    key_insights: list[str] = field(default_factory=list)


@dataclass
class ResearchPaper:
    """Structured representation of a research paper extracted from a PDF."""

    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    keywords: list[str] = field(default_factory=list)
    section_headings: list[str] = field(default_factory=list)
    page_count: int = 0
    word_count: int = 0

    # Additional fields useful for extracting/indexing research PDFs:
    full_text: str = ""
    references: list[str] = field(default_factory=list)
    doi: str = ""
    publication_year: int = 0
    venue: str = ""
    source_path: str = ""

    # Structured summary, populated by the summarizer (empty until then).
    summary: PaperSummary = field(default_factory=PaperSummary)
