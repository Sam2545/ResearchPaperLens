"""Data model representing a research paper."""

from dataclasses import dataclass, field


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

    # Reserved for a future summarization phase. Left empty for now; no
    # summarization logic populates these yet.
    summary: str = ""
    key_insights: list[str] = field(default_factory=list)
