"""Hybrid search — semantic (vector) and keyword (FTS5) search backends."""

from __future__ import annotations

from typing import Optional

from ..models import Memory
from ..storage.cache_utils import ensure_user_cache
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
    user_key = ensure_user_cache(store, vector_cache, user_id, namespace)

    hits = vector_cache.search(user_key, query_embedding, top_k=top_k)

    if not hits:
        return []

    # Batch fetch all memories in a single query
    hit_ids = [memory_id for memory_id, _ in hits]
    memories = store.batch_get(hit_ids)
    memory_map = {m.id: m for m in memories}

    results: list[tuple[Memory, float]] = []
    for memory_id, score in hits:
        memory = memory_map.get(memory_id)
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
