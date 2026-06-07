"""In-memory vector index for semantic chunk search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.chunk_store import ChunkStore, IndexedChunk


@dataclass(frozen=True)
class SearchResult:
    """One ranked chunk match from a vector search."""

    chunk: IndexedChunk
    score: float


class InMemoryVectorIndex:
    """Cosine-similarity search over chunk embeddings held in memory."""

    def __init__(self, chunks: list[IndexedChunk]) -> None:
        if not chunks:
            raise ValueError("InMemoryVectorIndex requires at least one chunk")
        missing = [chunk.id for chunk in chunks if not chunk.embedding]
        if missing:
            raise ValueError(
                f"All chunks must have embeddings; missing for: {missing[0]}"
            )

        self._chunks = list(chunks)
        matrix = np.asarray([chunk.embedding for chunk in chunks], dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._matrix = matrix / norms

    @classmethod
    def from_store(cls, store: ChunkStore) -> InMemoryVectorIndex:
        """Build an index from a :class:`ChunkStore`."""
        return cls(store.chunks)

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Return the top ``top_k`` chunks by cosine similarity."""
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")

        query = np.asarray(query_embedding, dtype=np.float64)
        norm = np.linalg.norm(query)
        if norm == 0:
            raise ValueError("query_embedding must be non-zero")
        query = query / norm

        scores = self._matrix @ query
        k = min(top_k, len(self._chunks))
        if k == len(self._chunks):
            ranked_indices = np.argsort(scores)[::-1]
        else:
            ranked_indices = np.argpartition(scores, -k)[-k:]
            ranked_indices = ranked_indices[np.argsort(scores[ranked_indices])[::-1]]

        return [
            SearchResult(chunk=self._chunks[index], score=float(scores[index]))
            for index in ranked_indices
        ]
