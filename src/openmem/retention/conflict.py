"""Conflict detection and resolution — find and handle contradictory memories."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

from ..storage.cache_utils import ensure_user_cache
from ..types import ConflictPair, ConflictStrategy, LLMCallback, LLMRequest
from .._utils import utc_now as _now

if TYPE_CHECKING:
    from ..config import OpenMemConfig
    from ..models import Memory
    from ..storage.sqlite_store import SQLiteStore
    from ..storage.vector_cache import VectorCache


_CONFLICT_SYSTEM_PROMPT = """\
You are a conflict detection engine for a memory system. Given two memories about \
a user, determine if they contradict each other.

Respond with a JSON object containing:
- "is_conflict": true if the memories contradict each other, false otherwise
- "explanation": brief explanation of why they do or do not conflict

Return ONLY the JSON object, no other text."""

_CONFLICT_USER_PROMPT = """\
Do these two memories contradict each other?

Memory A: {content_a}
Memory B: {content_b}

Return a JSON object with: is_conflict (bool), explanation (str)."""


def find_conflicts(
    store: SQLiteStore,
    vector_cache: VectorCache,
    llm_callback: LLMCallback,
    user_id: str,
    config: OpenMemConfig,
) -> list[ConflictPair]:
    """Find potentially contradictory memory pairs for a user.

    1. Get all active memories.
    2. Find pairs with high embedding similarity (above config.conflict_similarity_threshold).
    3. Use LLM to verify if the pair is actually contradictory.

    Returns:
        List of ConflictPair objects for confirmed conflicts.
    """
    memories = store.get_active_memories(user_id=user_id)
    if len(memories) < 2:
        return []

    # Ensure vector cache is populated for all namespaces the user's memories span
    namespaces = {m.namespace for m in memories}
    for ns in namespaces:
        ensure_user_cache(store, vector_cache, user_id, ns)

    threshold = config.conflict_similarity_threshold
    checked: set[tuple[str, str]] = set()
    conflicts: list[ConflictPair] = []
    memory_map = {m.id: m for m in memories}

    for memory in memories:
        if not memory.embedding:
            continue

        user_key = f"{memory.user_id}:{memory.namespace}"
        results = vector_cache.search(user_key, memory.embedding, top_k=20)

        for mem_id, score in results:
            if mem_id == memory.id or mem_id not in memory_map:
                continue
            if score < threshold:
                continue

            # Avoid checking the same pair twice
            pair_key = tuple(sorted((memory.id, mem_id)))
            if pair_key in checked:
                continue
            checked.add(pair_key)

            other = memory_map[mem_id]

            # Ask LLM to verify conflict
            response = llm_callback(
                LLMRequest(
                    system_prompt=_CONFLICT_SYSTEM_PROMPT,
                    user_prompt=_CONFLICT_USER_PROMPT.format(
                        content_a=memory.content,
                        content_b=other.content,
                    ),
                    expected_format="json",
                )
            )

            try:
                result = json.loads(response.content)
            except (json.JSONDecodeError, TypeError):
                continue

            if result.get("is_conflict"):
                conflicts.append(
                    ConflictPair(
                        memory_id_a=memory.id,
                        memory_id_b=mem_id,
                        similarity_score=score,
                        explanation=result.get("explanation", ""),
                    )
                )

    return conflicts


def resolve_conflict(
    store: SQLiteStore,
    vector_cache: VectorCache,
    memory_a: Memory,
    memory_b: Memory,
    strategy: ConflictStrategy,
) -> None:
    """Resolve a conflict between two memories using the given strategy.

    Args:
        store: The SQLite store.
        vector_cache: The vector cache.
        memory_a: First memory in the conflict.
        memory_b: Second memory in the conflict.
        strategy: How to resolve the conflict.
    """
    now = _now()

    if strategy == ConflictStrategy.KEEP_BOTH:
        return

    def _supersede(winner: Memory, loser: Memory) -> None:
        loser.superseded_by = winner.id
        loser.valid_until = now
        loser.is_active = False
        store.update(loser)
        user_key = f"{loser.user_id}:{loser.namespace}"
        vector_cache.remove_from_user(user_key, loser.id)

    if strategy == ConflictStrategy.SUPERSEDE:
        # Directional: memory_a always supersedes memory_b
        _supersede(memory_a, memory_b)

    elif strategy == ConflictStrategy.KEEP_NEWER:
        # Newer supersedes older
        if memory_a.created_at >= memory_b.created_at:
            _supersede(memory_a, memory_b)
        else:
            _supersede(memory_b, memory_a)

    elif strategy == ConflictStrategy.KEEP_HIGHER_CONFIDENCE:
        if memory_a.confidence >= memory_b.confidence:
            _supersede(memory_a, memory_b)
        else:
            _supersede(memory_b, memory_a)
