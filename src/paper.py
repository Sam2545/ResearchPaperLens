"""Data model representing a research paper."""

from dataclasses import dataclass, field


@dataclass
class Paper:
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    text: str = ""
    source_path: str = ""
