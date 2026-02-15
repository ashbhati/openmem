"""Ranking — merge hybrid search results and apply filters."""

from __future__ import annotations

from typing import Optional

from ..models import Memory
from ..types import MemoryType


def merge_and_rank(
    semantic_results: list[tuple[Memory, float]],
    keyword_results: list[tuple[Memory, float]],
    semantic_weight: float,
    keyword_weight: float,
    top_k: int,
) -> list[tuple[Memory, float]]:
    """Merge semantic and keyword results into a single ranked list.

    For memories appearing in both result sets:
        combined_score = semantic_weight * sem_score + keyword_weight * kw_score

    For memories in only one set:
        combined_score = score * that_weight

    Returns up to top_k (Memory, combined_score) pairs, sorted descending.
    """
    # Build lookup dicts keyed by memory id
    sem_map: dict[str, tuple[Memory, float]] = {
        m.id: (m, score) for m, score in semantic_results
    }
    kw_map: dict[str, tuple[Memory, float]] = {
        m.id: (m, score) for m, score in keyword_results
    }

    all_ids = set(sem_map.keys()) | set(kw_map.keys())
    scored: list[tuple[Memory, float]] = []

    for mid in all_ids:
        in_sem = mid in sem_map
        in_kw = mid in kw_map

        if in_sem and in_kw:
            memory = sem_map[mid][0]
            combined = semantic_weight * sem_map[mid][1] + keyword_weight * kw_map[mid][1]
        elif in_sem:
            memory = sem_map[mid][0]
            combined = semantic_weight * sem_map[mid][1]
        else:
            memory = kw_map[mid][0]
            combined = keyword_weight * kw_map[mid][1]

        scored.append((memory, combined))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def apply_filters(
    memories_with_scores: list[tuple[Memory, float]],
    min_confidence: float = 0.0,
    min_strength: float = 0.0,
    memory_types: Optional[list[MemoryType]] = None,
    namespace: Optional[str] = None,
) -> list[tuple[Memory, float]]:
    """Filter memories based on confidence, strength, type, and namespace."""
    results: list[tuple[Memory, float]] = []
    for memory, score in memories_with_scores:
        if memory.confidence < min_confidence:
            continue
        if memory.strength < min_strength:
            continue
        if memory_types and memory.memory_type not in memory_types:
            continue
        if namespace is not None and memory.namespace != namespace:
            continue
        results.append((memory, score))
    return results
