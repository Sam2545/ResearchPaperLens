"""Semantic retrieval: index paper chunks and search them in memory."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from src.chunk_store import ChunkStore, IndexedChunk
from src.chunking import (
    DEFAULT_MAX_SECTION_WORDS,
    DEFAULT_OVERLAP_WORDS,
    SectionChunkPlan,
    plan_hybrid_chunks,
)
from src.embeddings import DEFAULT_EMBED_MODEL, embed_query, embed_texts
from src.paper import ResearchPaper
from src.vector_index import InMemoryVectorIndex, SearchResult

if TYPE_CHECKING:
    from ollama import Client

_SLUG_RE = re.compile(r"[^\w]+")


def paper_stem(paper: ResearchPaper) -> str:
    """Stable filename stem for ``paper`` (from ``source_path`` or ``paper``)."""
    if paper.source_path:
        return Path(paper.source_path).stem
    return "paper"


def section_slug(section_heading: str) -> str:
    """URL-safe slug from a section heading for chunk IDs."""
    slug = _SLUG_RE.sub("-", section_heading.strip().lower()).strip("-")
    return slug or "section"


def make_chunk_id(paper: ResearchPaper, section_heading: str, chunk_index: int) -> str:
    """Build a stable chunk identifier for one paper section slice."""
    return f"{paper_stem(paper)}:{section_slug(section_heading)}:{chunk_index}"


def plan_retrieval_chunks(paper: ResearchPaper) -> list[SectionChunkPlan]:
    """Hybrid chunk plan for retrieval: Abstract + numbered sections only."""
    return plan_hybrid_chunks(
        paper,
        include_abstract=True,
        include_preamble=False,
        max_section_words=DEFAULT_MAX_SECTION_WORDS,
        words_per_chunk=DEFAULT_MAX_SECTION_WORDS,
        overlap_words=DEFAULT_OVERLAP_WORDS,
    )


def chunks_from_plan(paper: ResearchPaper, plans: list[SectionChunkPlan]) -> list[IndexedChunk]:
    """Materialize :class:`IndexedChunk` records from a hybrid chunk plan."""
    source = paper.source_path or ""
    indexed: list[IndexedChunk] = []
    for plan in plans:
        for chunk in plan.chunks:
            if not chunk.text.strip():
                continue
            indexed.append(
                IndexedChunk(
                    id=make_chunk_id(paper, plan.section.heading, chunk.index),
                    paper_source=source,
                    section_heading=plan.section.heading,
                    text=chunk.text,
                    word_count=chunk.word_count,
                    start_word=chunk.start_word,
                    end_word=chunk.end_word,
                )
            )
    return indexed


def index_paper(
    paper: ResearchPaper,
    *,
    client: Client | None = None,
    model: str = DEFAULT_EMBED_MODEL,
) -> ChunkStore:
    """Chunk a paper, embed each chunk, and return an in-memory store."""
    plans = plan_retrieval_chunks(paper)
    chunks = chunks_from_plan(paper, plans)
    if not chunks:
        return ChunkStore(
            paper_source=paper.source_path or "",
            paper_title=paper.title,
            embed_model=model,
            embed_dimensions=0,
            chunks=[],
        )

    embeddings = embed_texts(
        [chunk.text for chunk in chunks],
        client=client,
        model=model,
    )
    embedded = [
        IndexedChunk(
            id=chunk.id,
            paper_source=chunk.paper_source,
            section_heading=chunk.section_heading,
            text=chunk.text,
            word_count=chunk.word_count,
            start_word=chunk.start_word,
            end_word=chunk.end_word,
            embedding=embedding,
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
    dimensions = len(embeddings[0]) if embeddings else 0
    return ChunkStore(
        paper_source=paper.source_path or "",
        paper_title=paper.title,
        embed_model=model,
        embed_dimensions=dimensions,
        chunks=embedded,
    )


def search(
    store: ChunkStore,
    query: str,
    *,
    client: Client | None = None,
    model: str | None = None,
    top_k: int = 5,
) -> list[SearchResult]:
    """Embed ``query`` and return the top matching chunks from ``store``."""
    if not store.chunks:
        return []

    embed_model = model or store.embed_model
    query_vector = embed_query(query, client=client, model=embed_model)
    index = InMemoryVectorIndex.from_store(store)
    return index.search(query_vector, top_k=top_k)
