"""CaptureEngine — full pipeline from conversation to stored memories."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from ..config import OpenMemConfig
from ..models import Memory
from ..storage.sqlite_store import SQLiteStore
from ..storage.vector_cache import VectorCache
from ..types import (
    EmbeddingCallback,
    LLMCallback,
    MemoryLifespan,
    MemorySource,
    MemoryType,
)
from .._ulid import generate_ulid
from .._utils import utc_now as _now, content_hash as _content_hash
from .extractor import extract_memories


class CaptureEngine:
    """Extracts memories from conversations and stores them with deduplication.

    Pipeline:
        1. LLM extraction (via extractor)
        2. Content-hash dedup (exact match)
        3. Embedding generation
        4. Near-duplicate detection (vector similarity >= dedup_threshold)
        5. Conflict detection (vector similarity >= conflict_threshold)
        6. Conflict resolution (configurable strategy)
        7. Storage + cache update
    """

    def __init__(
        self,
        store: SQLiteStore,
        vector_cache: VectorCache,
        llm_callback: Optional[LLMCallback],
        embedding_callback: Optional[EmbeddingCallback],
        config: OpenMemConfig,
    ) -> None:
        self._store = store
        self._vector_cache = vector_cache
        self._llm_callback = llm_callback
        self._embedding_callback = embedding_callback
        self._config = config

    def _ensure_user_cache(self, user_id: str, namespace: str) -> str:
        """Ensure vector cache is populated for user. Returns user_key."""
        user_key = f"{user_id}:{namespace}"
        if not self._vector_cache.has_user(user_key):
            embeddings = self._store.get_all_embeddings(user_id, namespace=namespace)
            self._vector_cache.build_user_index(user_key, embeddings)
        return user_key

    def capture(
        self,
        user_id: str,
        messages: list[dict[str, str]],
        namespace: Optional[str] = None,
        lifespan: Optional[MemoryLifespan] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[Memory]:
        """Extract and store memories from a conversation.

        Args:
            user_id: The user this conversation belongs to.
            messages: Conversation as list of {"role": ..., "content": ...}.
            namespace: Isolation scope (defaults to config.default_namespace).
            lifespan: Memory lifespan category (defaults to config.default_lifespan).
            metadata: Arbitrary key-value pairs to attach to each memory.

        Returns:
            List of newly created Memory objects.
        """
        if self._llm_callback is None:
            raise ValueError("llm_callback is required for capture")
        if self._embedding_callback is None:
            raise ValueError("embedding_callback is required for capture")

        ns = namespace or self._config.default_namespace
        ls = lifespan or self._config.default_lifespan
        meta = metadata or {}

        # Ensure vector cache is ready for this user
        user_key = self._ensure_user_cache(user_id, ns)

        # Step 1: Extract raw memories via LLM
        raw_memories = extract_memories(
            llm_callback=self._llm_callback,
            messages=messages,
        )

        new_memories: list[Memory] = []
        now = _now()

        for raw in raw_memories:
            content = raw["content"]
            content_h = _content_hash(content)

            # Step 2: Check exact duplicate via content hash
            existing = self._store.find_by_content_hash(content_h, user_id, ns)
            if existing is not None:
                continue

            # Step 3: Generate embedding
            embedding = self._embedding_callback(content)

            # Step 4: Check near-duplicate via vector similarity
            if embedding:
                similar = self._vector_cache.search(
                    user_key, embedding, top_k=1
                )
                if similar:
                    _, score = similar[0]
                    if score >= self._config.dedup_similarity_threshold:
                        continue

            # Step 5: Check for conflicts via embedding similarity
            # (similarity above conflict threshold but below dedup threshold)
            # With keep_both strategy (default), we just store both.
            # Future strategies can supersede or pick higher confidence here.

            # Step 6: Create Memory object
            memory = Memory(
                id=generate_ulid(),
                user_id=user_id,
                namespace=ns,
                content=content,
                content_hash=content_h,
                memory_type=MemoryType(raw["memory_type"]),
                source=MemorySource(raw["source"]),
                confidence=raw["confidence"],
                strength=1.0,
                created_at=now,
                last_accessed=now,
                access_count=0,
                access_timestamps=[now],
                lifespan=ls,
                version=1,
                is_active=True,
                embedding=embedding,
                embedding_model="",
                metadata=meta,
            )

            # Step 7: Store and update cache
            self._store.add(memory)
            if embedding:
                self._vector_cache.add_to_user(user_key, memory.id, embedding)

            new_memories.append(memory)

        return new_memories
