"""Persistence helpers: serialize a ResearchPaper to/from JSON.

This module is intentionally standalone. Its only job is converting a
:class:`~src.paper.ResearchPaper` to a plain dictionary (and back) and reading
or writing that dictionary as JSON on disk. It knows nothing about PDFs or
analysis, which keeps storage concerns cleanly separated.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

from src.paper import ResearchPaper


def paper_from_dict(data: dict) -> ResearchPaper:
    """Build a :class:`ResearchPaper` from a dict.

    Unknown keys are ignored and missing keys fall back to the dataclass
    defaults, so the loader stays forward/backward compatible as the schema
    evolves.
    """
    known = {f.name for f in fields(ResearchPaper)}
    filtered = {key: value for key, value in data.items() if key in known}
    return ResearchPaper(**filtered)


def paper_to_json(paper: ResearchPaper, *, indent: int = 2) -> str:
    """Serialize a :class:`ResearchPaper` to a JSON string."""
    return json.dumps(asdict(paper), indent=indent, ensure_ascii=False)


def paper_from_json(text: str) -> ResearchPaper:
    """Deserialize a :class:`ResearchPaper` from a JSON string."""
    return paper_from_dict(json.loads(text))


def save_paper(paper: ResearchPaper, path: str | Path, *, indent: int = 2) -> None:
    """Write a :class:`ResearchPaper` to ``path`` as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(paper_to_json(paper, indent=indent), encoding="utf-8")


def load_paper(path: str | Path) -> ResearchPaper:
    """Read a :class:`ResearchPaper` from a JSON file at ``path``."""
    text = Path(path).read_text(encoding="utf-8")
    return paper_from_json(text)
