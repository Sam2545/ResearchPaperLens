"""Persistence for indexed paper chunks and their embeddings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from src.paper import ResearchPaper


@dataclass
class IndexedChunk:
    """One searchable text chunk with optional embedding vector."""

    id: str
    paper_source: str
    section_heading: str
    text: str
    word_count: int
    start_word: int
    end_word: int
    embedding: list[float] = field(default_factory=list)


@dataclass
class ChunkStore:
    """Embeddings-backed chunk index for a single paper."""

    paper_source: str
    paper_title: str
    embed_model: str
    embed_dimensions: int
    chunks: list[IndexedChunk] = field(default_factory=list)


def chunk_from_dict(data: dict) -> IndexedChunk:
    """Build an :class:`IndexedChunk` from a dict, ignoring unknown keys."""
    known = {f.name for f in fields(IndexedChunk)}
    filtered = {key: value for key, value in data.items() if key in known}
    return IndexedChunk(**filtered)


def store_from_dict(data: dict) -> ChunkStore:
    """Build a :class:`ChunkStore` from a dict."""
    known = {f.name for f in fields(ChunkStore)}
    filtered = {key: value for key, value in data.items() if key in known}
    chunks = filtered.get("chunks", [])
    if chunks and isinstance(chunks[0], dict):
        filtered["chunks"] = [chunk_from_dict(chunk) for chunk in chunks]
    return ChunkStore(**filtered)


def store_to_json(store: ChunkStore, *, indent: int = 2) -> str:
    """Serialize a :class:`ChunkStore` to a JSON string."""
    return json.dumps(asdict(store), indent=indent, ensure_ascii=False)


def store_from_json(text: str) -> ChunkStore:
    """Deserialize a :class:`ChunkStore` from a JSON string."""
    return store_from_dict(json.loads(text))


def save_chunk_store(store: ChunkStore, path: str | Path, *, indent: int = 2) -> None:
    """Write a :class:`ChunkStore` to ``path`` as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(store_to_json(store, indent=indent), encoding="utf-8")


def load_chunk_store(path: str | Path) -> ChunkStore:
    """Read a :class:`ChunkStore` from a JSON file at ``path``."""
    text = Path(path).read_text(encoding="utf-8")
    return store_from_json(text)


def default_chunk_store_path(
    paper: ResearchPaper,
    outputs_dir: str | Path = "outputs",
) -> Path:
    """Return the default per-paper chunk index path under ``outputs_dir``."""
    stem = Path(paper.source_path).stem if paper.source_path else "paper"
    return Path(outputs_dir) / f"{stem}.chunks.json"
