"""Memory consolidation — merge related memories into stronger, unified ones.

Two-phase process:
1. propose_consolidations: Find clusters, draft merged content via LLM.
2. apply_consolidations: Create new memories, mark sources as superseded.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

from ..models import Memory
from ..types import (
    ConsolidationProposal,
    EmbeddingCallback,
    LLMCallback,
    LLMRequest,
    MemorySource,
    MemoryType,
)
from .._ulid import generate_ulid
from .._utils import utc_now as _now, content_hash as _content_hash

if TYPE_CHECKING:
    from ..config import OpenMemConfig
    from ..storage.sqlite_store import SQLiteStore
    from ..storage.vector_cache import VectorCache


_CONSOLIDATION_SYSTEM_PROMPT = """\
You are a memory consolidation engine. Given a set of related memories about a user, \
merge them into a single, concise memory that preserves all important information.

Respond with a JSON object containing:
- "content": the merged memory text (concise, factual)
- "memory_type": one of "fact", "preference", "insight", "biographical"
- "confidence": a float between 0.0 and 1.0 reflecting overall confidence
- "reasoning": brief explanation of how you merged them

Return ONLY the JSON object, no other text."""

_CONSOLIDATION_USER_PROMPT = """\
Merge the following related memories into one:

{memories_text}

Return a JSON object with: content, memory_type, confidence, reasoning."""


def _find_clusters(
    memories: list[Memory],
    vector_cache: VectorCache,
    config: OpenMemConfig,
) -> list[list[Memory]]:
    """Find clusters of similar memories using embedding similarity.

    Groups memories where pairwise similarity exceeds threshold (~0.8).
    Simple greedy clustering: iterate memories, try to add to existing cluster.
    """
    if not memories:
        return []

    threshold = 0.8
    clusters: list[list[Memory]] = []
    assigned: set[str] = set()
    memory_map = {m.id: m for m in memories}

    # Build a user_key -> memories mapping for cache lookups
    for memory in memories:
        if memory.id in assigned or not memory.embedding:
            continue

        user_key = f"{memory.user_id}:{memory.namespace}"
        # Search for similar memories
        results = vector_cache.search(user_key, memory.embedding, top_k=50)

        cluster = [memory]
        assigned.add(memory.id)

        for mem_id, score in results:
            if mem_id == memory.id or mem_id in assigned:
                continue
            if score >= threshold and mem_id in memory_map:
                cluster.append(memory_map[mem_id])
                assigned.add(mem_id)

        if len(cluster) >= 2:
            clusters.append(cluster)

    return clusters


def propose_consolidations(
    store: SQLiteStore,
    vector_cache: VectorCache,
    llm_callback: LLMCallback,
    user_id: str,
    config: OpenMemConfig,
) -> list[ConsolidationProposal]:
    """Propose memory consolidations by finding similar clusters and merging via LLM.

    Phase 1 of 2 — returns proposals for human or automated review.
    """
    memories = store.get_active_memories(user_id=user_id)
    # Skip memories that have already been superseded
    memories = [m for m in memories if m.superseded_by is None]

    if len(memories) < 2:
        return []

    # Ensure vector cache is populated
    user_key = f"{user_id}:{config.default_namespace}"
    if not vector_cache.has_user(user_key):
        embeddings = store.get_all_embeddings(user_id, namespace=config.default_namespace)
        vector_cache.build_user_index(user_key, embeddings)

    clusters = _find_clusters(memories, vector_cache, config)
    proposals: list[ConsolidationProposal] = []

    for cluster in clusters:
        memories_text = "\n".join(
            f"- [{m.memory_type.value}] {m.content} (confidence: {m.confidence})"
            for m in cluster
        )

        response = llm_callback(
            LLMRequest(
                system_prompt=_CONSOLIDATION_SYSTEM_PROMPT,
                user_prompt=_CONSOLIDATION_USER_PROMPT.format(memories_text=memories_text),
                expected_format="json",
            )
        )

        try:
            result = json.loads(response.content)
        except (json.JSONDecodeError, TypeError):
            continue

        proposals.append(
            ConsolidationProposal(
                source_memory_ids=[m.id for m in cluster],
                proposed_content=result.get("content", ""),
                reasoning=result.get("reasoning", ""),
                proposed_memory_type=MemoryType(result["memory_type"])
                if result.get("memory_type")
                else None,
                proposed_confidence=result.get("confidence"),
            )
        )

    return proposals


def apply_consolidations(
    store: SQLiteStore,
    vector_cache: VectorCache,
    embedding_callback: EmbeddingCallback | None,
    proposals: list[ConsolidationProposal],
    config: OpenMemConfig,
) -> list[Memory]:
    """Apply approved consolidation proposals.

    Phase 2 of 2 — creates new merged memories and marks sources as superseded.
    """
    now = _now()
    new_memories: list[Memory] = []

    for proposal in proposals:
        if not proposal.source_memory_ids or not proposal.proposed_content:
            continue

        # Load source memories to get user_id, namespace, etc.
        source_memories = []
        for mid in proposal.source_memory_ids:
            mem = store.get(mid)
            if mem is not None:
                source_memories.append(mem)

        if not source_memories:
            continue

        # Use the first source memory for user_id/namespace
        ref = source_memories[0]

        # Generate embedding for new memory
        embedding: list[float] = []
        if embedding_callback is not None:
            embedding = embedding_callback(proposal.proposed_content)

        new_memory = Memory(
            id=generate_ulid(),
            user_id=ref.user_id,
            namespace=ref.namespace,
            content=proposal.proposed_content,
            content_hash=_content_hash(proposal.proposed_content),
            memory_type=proposal.proposed_memory_type or ref.memory_type,
            source=MemorySource.IMPLICIT,
            confidence=proposal.proposed_confidence or ref.confidence,
            strength=1.0,
            created_at=now,
            last_accessed=now,
            access_count=0,
            access_timestamps=[now],
            lifespan=ref.lifespan,
            version=1,
            is_active=True,
            embedding=embedding,
            embedding_model=ref.embedding_model,
            metadata={"consolidated_from": proposal.source_memory_ids},
        )

        store.add(new_memory)

        # Mark source memories as superseded
        for src in source_memories:
            src.superseded_by = new_memory.id
            src.valid_until = now
            src.is_active = False
            store.update(src)

        # Update vector cache
        user_key = f"{ref.user_id}:{ref.namespace}"
        if embedding:
            vector_cache.add_to_user(user_key, new_memory.id, embedding)
        # Remove superseded memories from cache
        for src in source_memories:
            vector_cache.remove_from_user(user_key, src.id)

        new_memories.append(new_memory)

    return new_memories
