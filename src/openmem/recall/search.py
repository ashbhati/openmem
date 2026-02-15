"""Hybrid search — semantic (vector) and keyword (FTS5) search backends."""

from __future__ import annotations

from typing import Optional

from ..models import Memory
from ..storage.sqlite_store import SQLiteStore
from ..storage.vector_cache import VectorCache


def semantic_search(
    vector_cache: VectorCache,
    store: SQLiteStore,
    user_id: str,
    namespace: str,
    query_embedding: list[float],
    top_k: int = 10,
) -> list[tuple[Memory, float]]:
    """Search memories by embedding cosine similarity.

    Lazily loads the user's vector index from SQLite if not already cached.
    Returns (Memory, score) pairs sorted by descending similarity.
    """
    user_key = f"{user_id}:{namespace}"

    # Lazy-load: build cache from store if user not present
    if not vector_cache.has_user(user_key):
        embeddings = store.get_all_embeddings(user_id, namespace=namespace)
        vector_cache.build_user_index(user_key, embeddings)

    hits = vector_cache.search(user_key, query_embedding, top_k=top_k)

    results: list[tuple[Memory, float]] = []
    for memory_id, score in hits:
        memory = store.get(memory_id)
        if memory is not None and memory.is_active:
            results.append((memory, score))
    return results


def keyword_search(
    store: SQLiteStore,
    user_id: str,
    namespace: Optional[str],
    query: str,
    top_k: int = 10,
) -> list[tuple[Memory, float]]:
    """Search memories using SQLite FTS5 full-text search.

    FTS5 returns raw rank scores (negative, more negative = better match).
    The store already converts these to positive values. This function
    normalizes them to 0-1 by dividing by the max score in the result set.
    """
    raw_results = store.fts_search(
        query=query, user_id=user_id, namespace=namespace, limit=top_k
    )

    if not raw_results:
        return []

    # Normalize scores to 0-1 range
    max_score = max(score for _, score in raw_results)
    if max_score > 0:
        return [(memory, score / max_score) for memory, score in raw_results]

    return raw_results
