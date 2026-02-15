"""OpenMem client — the main entry point for the library.

The SQLite of AI memory: embedded, zero-dependency, lifecycle-aware.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from .config import OpenMemConfig
from .models import Memory
from .storage.sqlite_store import SQLiteStore
from .storage.vector_cache import VectorCache
from .types import (
    AsyncEmbeddingCallback,
    AsyncLLMCallback,
    ConflictPair,
    ConflictStrategy,
    ConsolidationProposal,
    DecayResult,
    EmbeddingCallback,
    LLMCallback,
    LLMRequest,
    LLMResponse,
    MemoryLifespan,
    MemorySource,
    MemoryStats,
    MemoryType,
)
from ._ulid import generate_ulid
from ._utils import utc_now as _now, content_hash as _content_hash


class OpenMem:
    """The embedded memory engine for AI agents.

    Usage::

        from openmem import OpenMem, LLMRequest, LLMResponse

        def my_llm(req: LLMRequest) -> LLMResponse:
            return LLMResponse(content=call_my_model(req.user_prompt))

        def my_embed(text: str) -> list[float]:
            return get_embedding(text)

        mem = OpenMem(llm_callback=my_llm, embedding_callback=my_embed)

        # Capture memories from conversation
        memories = mem.capture(user_id="user_1", messages=[...])

        # Recall relevant memories
        results = mem.recall(user_id="user_1", query="...")
    """

    def __init__(
        self,
        llm_callback: Optional[LLMCallback] = None,
        embedding_callback: Optional[EmbeddingCallback] = None,
        async_llm_callback: Optional[AsyncLLMCallback] = None,
        async_embedding_callback: Optional[AsyncEmbeddingCallback] = None,
        storage_path: Optional[str] = None,
        config: Optional[OpenMemConfig] = None,
    ) -> None:
        self._config = config or OpenMemConfig()
        if storage_path:
            self._config.storage_path = storage_path

        self._llm_callback = llm_callback
        self._embedding_callback = embedding_callback
        self._async_llm_callback = async_llm_callback
        self._async_embedding_callback = async_embedding_callback

        # Initialize storage
        self._store = SQLiteStore(
            path=self._config.resolved_storage_path,
            busy_timeout_ms=self._config.sqlite_busy_timeout_ms,
        )
        self._vector_cache = VectorCache(
            max_users=self._config.vector_cache_max_users,
        )

        # Lazy imports for engines (set up in respective milestones)
        self._capture_engine = None
        self._recall_engine = None
        self._retention_engine = None

    @classmethod
    def from_simple_callback(
        cls,
        llm_fn: Any,  # Callable[[str], str]
        embed_fn: Any,  # Callable[[str], list[float]]
        **kwargs: Any,
    ) -> OpenMem:
        """Convenience constructor for quick prototyping with simple callbacks.

        Args:
            llm_fn: A function (str) -> str that calls your LLM.
            embed_fn: A function (str) -> list[float] that generates embeddings.
            **kwargs: Passed to OpenMem.__init__.
        """

        def wrapped_llm(request: LLMRequest) -> LLMResponse:
            prompt = f"{request.system_prompt}\n\n{request.user_prompt}"
            result = llm_fn(prompt)
            return LLMResponse(content=result)

        return cls(
            llm_callback=wrapped_llm,
            embedding_callback=embed_fn,
            **kwargs,
        )

    @property
    def config(self) -> OpenMemConfig:
        return self._config

    # --- CRUD Operations ---

    def add(
        self,
        user_id: str,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        source: MemorySource = MemorySource.EXPLICIT,
        confidence: float = 1.0,
        lifespan: Optional[MemoryLifespan] = None,
        namespace: Optional[str] = None,
        ttl: Optional[datetime] = None,
        metadata: Optional[dict[str, Any]] = None,
        embedding: Optional[list[float]] = None,
        embedding_model: str = "",
    ) -> Memory:
        """Add a single memory directly (without LLM extraction).

        If embedding is not provided and an embedding_callback is set,
        the embedding will be generated automatically.

        Returns the created Memory object.
        """
        now = _now()
        ns = namespace or self._config.default_namespace
        ls = lifespan or self._config.default_lifespan

        # Generate embedding if not provided
        if embedding is None and self._embedding_callback is not None:
            embedding = self._embedding_callback(content)

        memory = Memory(
            id=generate_ulid(),
            user_id=user_id,
            namespace=ns,
            content=content,
            content_hash=_content_hash(content),
            memory_type=memory_type,
            source=source,
            confidence=confidence,
            strength=1.0,
            created_at=now,
            last_accessed=now,
            access_count=0,
            access_timestamps=[now],
            lifespan=ls,
            ttl=ttl,
            version=1,
            is_active=True,
            embedding=embedding or [],
            embedding_model=embedding_model,
            metadata=metadata or {},
        )

        self._store.add(memory)

        # Update vector cache
        if memory.embedding:
            user_key = f"{user_id}:{ns}"
            self._vector_cache.add_to_user(user_key, memory.id, memory.embedding)

        return memory

    def get(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a single memory by ID."""
        return self._store.get(memory_id)

    def list(
        self,
        user_id: str,
        namespace: Optional[str] = None,
        memory_types: Optional[list[MemoryType]] = None,
        active_only: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Memory]:
        """List memories for a user with optional filters and pagination."""
        return self._store.list(
            user_id=user_id,
            namespace=namespace,
            memory_types=memory_types,
            active_only=active_only,
            limit=limit,
            offset=offset,
        )

    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        confidence: Optional[float] = None,
        lifespan: Optional[MemoryLifespan] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[Memory]:
        """Update a memory's fields. Re-embeds if content changes.

        Returns the updated Memory, or None if not found.
        """
        memory = self._store.get(memory_id)
        if memory is None:
            return None

        old_content = memory.content
        if content is not None:
            memory.content = content
            memory.content_hash = _content_hash(content)
        if memory_type is not None:
            memory.memory_type = memory_type
        if confidence is not None:
            memory.confidence = confidence
        if lifespan is not None:
            memory.lifespan = lifespan
        if metadata is not None:
            memory.metadata = metadata

        memory.version += 1

        # Re-embed if content changed
        if content is not None and content != old_content and self._embedding_callback:
            memory.embedding = self._embedding_callback(content)
            # Update vector cache
            user_key = f"{memory.user_id}:{memory.namespace}"
            self._vector_cache.remove_from_user(user_key, memory.id)
            if memory.embedding:
                self._vector_cache.add_to_user(user_key, memory.id, memory.embedding)

        self._store.update(memory)
        return memory

    def delete(self, memory_id: str) -> bool:
        """Hard-delete a memory (atomic: SQLite + cache invalidation).

        Returns True if the memory existed and was deleted.
        """
        memory = self._store.get(memory_id)
        if memory is None:
            return False

        user_key = f"{memory.user_id}:{memory.namespace}"
        result = self._store.delete(memory_id)
        if result:
            self._vector_cache.remove_from_user(user_key, memory_id)
        return result

    def delete_all(self, user_id: str) -> int:
        """Delete ALL memories for a user (GDPR right to erasure).

        Returns count of memories deleted.
        """
        # Invalidate all possible namespace caches for this user
        memories = self._store.list(user_id=user_id, active_only=False, limit=100000)
        namespaces = {m.namespace for m in memories}
        for ns in namespaces:
            self._vector_cache.invalidate(f"{user_id}:{ns}")

        return self._store.delete_all(user_id)

    def export(self, user_id: str, format: str = "json") -> str:
        """Export all memories for a user.

        Args:
            user_id: The user whose memories to export.
            format: "json" or "csv".

        Returns:
            Serialized string of all memories.
        """
        memories = self._store.list(
            user_id=user_id, active_only=False, limit=100000
        )

        if format == "json":
            return json.dumps(
                [m.to_dict() for m in memories], indent=2, default=str
            )
        elif format == "csv":
            if not memories:
                return ""
            headers = [
                "id", "user_id", "namespace", "content", "memory_type",
                "source", "confidence", "strength", "created_at",
                "last_accessed", "access_count", "lifespan", "version",
                "is_active",
            ]
            lines = [",".join(headers)]
            for m in memories:
                # Escape content for CSV: double any internal quotes, then wrap in quotes
                escaped_content = m.content.replace('"', '""')
                row = [
                    m.id, m.user_id, m.namespace,
                    f'"{escaped_content}"',
                    m.memory_type.value, m.source.value,
                    str(m.confidence), str(m.strength),
                    m.created_at.isoformat(), m.last_accessed.isoformat(),
                    str(m.access_count), m.lifespan.value,
                    str(m.version), str(m.is_active),
                ]
                lines.append(",".join(row))
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")

    # --- Vector Cache Management ---

    def _ensure_user_cache(self, user_id: str, namespace: Optional[str] = None) -> str:
        """Ensure the vector cache is populated for a user. Returns user_key."""
        ns = namespace or self._config.default_namespace
        user_key = f"{user_id}:{ns}"
        if not self._vector_cache.has_user(user_key):
            embeddings = self._store.get_all_embeddings(user_id, namespace=ns)
            self._vector_cache.build_user_index(user_key, embeddings)
        return user_key

    # --- Placeholder methods for engines (implemented in M2-M4) ---

    def capture(
        self,
        user_id: str,
        messages: list[dict[str, str]],
        namespace: Optional[str] = None,
        lifespan: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[Memory]:
        """Extract memories from a conversation using the LLM callback.

        Requires llm_callback and embedding_callback to be set.
        """
        from .capture.engine import CaptureEngine

        if self._capture_engine is None:
            self._capture_engine = CaptureEngine(
                store=self._store,
                vector_cache=self._vector_cache,
                llm_callback=self._llm_callback,
                embedding_callback=self._embedding_callback,
                config=self._config,
            )
        ls = MemoryLifespan(lifespan) if lifespan else None
        return self._capture_engine.capture(
            user_id=user_id,
            messages=messages,
            namespace=namespace,
            lifespan=ls,
            metadata=metadata,
        )

    def capture_batch(
        self,
        user_id: str,
        conversations: list[list[dict[str, str]]],
        namespace: Optional[str] = None,
        lifespan: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[Memory]:
        """Extract memories from multiple conversations."""
        all_memories: list[Memory] = []
        for messages in conversations:
            all_memories.extend(
                self.capture(
                    user_id=user_id,
                    messages=messages,
                    namespace=namespace,
                    lifespan=lifespan,
                    metadata=metadata,
                )
            )
        return all_memories

    def recall(
        self,
        user_id: str,
        query: str,
        top_k: Optional[int] = None,
        min_confidence: float = 0.0,
        min_strength: float = 0.0,
        memory_types: Optional[list[str]] = None,
        namespace: Optional[str] = None,
    ) -> list[Memory]:
        """Retrieve relevant memories with reinforcement (bumps strength)."""
        from .recall.engine import RecallEngine

        if self._recall_engine is None:
            self._recall_engine = RecallEngine(
                store=self._store,
                vector_cache=self._vector_cache,
                embedding_callback=self._embedding_callback,
                config=self._config,
            )
        types = [MemoryType(t) for t in memory_types] if memory_types else None
        return self._recall_engine.recall(
            user_id=user_id,
            query=query,
            top_k=top_k or self._config.max_memories_per_recall,
            min_confidence=min_confidence,
            min_strength=min_strength,
            memory_types=types,
            namespace=namespace,
            reinforce=True,
        )

    def search(
        self,
        user_id: str,
        query: str,
        top_k: Optional[int] = None,
        min_confidence: float = 0.0,
        min_strength: float = 0.0,
        memory_types: Optional[list[str]] = None,
        namespace: Optional[str] = None,
    ) -> list[Memory]:
        """Read-only query — like recall() but does NOT reinforce memories."""
        from .recall.engine import RecallEngine

        if self._recall_engine is None:
            self._recall_engine = RecallEngine(
                store=self._store,
                vector_cache=self._vector_cache,
                embedding_callback=self._embedding_callback,
                config=self._config,
            )
        types = [MemoryType(t) for t in memory_types] if memory_types else None
        return self._recall_engine.recall(
            user_id=user_id,
            query=query,
            top_k=top_k or self._config.max_memories_per_recall,
            min_confidence=min_confidence,
            min_strength=min_strength,
            memory_types=types,
            namespace=namespace,
            reinforce=False,
        )

    def _get_retention_engine(self):
        """Lazily initialize the retention engine."""
        if self._retention_engine is None:
            from .retention.engine import RetentionEngine

            self._retention_engine = RetentionEngine(
                store=self._store,
                vector_cache=self._vector_cache,
                llm_callback=self._llm_callback,
                embedding_callback=self._embedding_callback,
                config=self._config,
            )
        return self._retention_engine

    def decay(self) -> DecayResult:
        """Run decay on all active memories. Call periodically (e.g., daily)."""
        return self._get_retention_engine().decay()

    def purge(self) -> int:
        """Permanently remove all soft-deleted memories. Returns count purged."""
        count = self._store.purge_inactive()
        self._vector_cache.invalidate_all()
        return count

    def consolidate_propose(self, user_id: str) -> list[ConsolidationProposal]:
        """Propose memory consolidations for review (phase 1 of 2)."""
        return self._get_retention_engine().consolidate_propose(user_id)

    def consolidate_apply(
        self, proposals: list[ConsolidationProposal]
    ) -> list[Memory]:
        """Apply approved consolidation proposals (phase 2 of 2)."""
        return self._get_retention_engine().consolidate_apply(proposals)

    def consolidate(self, user_id: str) -> list[Memory]:
        """One-step consolidation: propose + auto-apply."""
        proposals = self.consolidate_propose(user_id)
        if not proposals:
            return []
        return self.consolidate_apply(proposals)

    def find_conflicts(self, user_id: str) -> list[ConflictPair]:
        """Find potentially contradictory memories for a user."""
        return self._get_retention_engine().find_conflicts(user_id)

    def resolve_conflict(
        self,
        memory_id_a: str,
        memory_id_b: str,
        strategy: Optional[ConflictStrategy] = None,
    ) -> None:
        """Resolve a conflict between two memories.

        Args:
            memory_id_a: First memory ID.
            memory_id_b: Second memory ID.
            strategy: Resolution strategy. Defaults to config.conflict_strategy.
        """
        from .retention.conflict import resolve_conflict as _resolve

        memory_a = self._store.get(memory_id_a)
        memory_b = self._store.get(memory_id_b)
        if memory_a is None or memory_b is None:
            raise ValueError("One or both memory IDs not found")

        strat = strategy or self._config.conflict_strategy
        _resolve(self._store, self._vector_cache, memory_a, memory_b, strat)

    def reembed(self, user_id: str, namespace: Optional[str] = None) -> int:
        """Re-embed all memories for a user using the current embedding callback.

        Use after changing embedding models. Returns count of re-embedded memories.
        """
        if self._embedding_callback is None:
            raise RuntimeError("No embedding_callback configured")

        memories = self._store.list(
            user_id=user_id, namespace=namespace, active_only=True, limit=100000
        )
        count = 0
        for memory in memories:
            memory.embedding = self._embedding_callback(memory.content)
            self._store.update(memory)
            count += 1

        # Rebuild vector cache for this user
        ns = namespace or self._config.default_namespace
        self._vector_cache.invalidate(f"{user_id}:{ns}")

        return count

    def stats(self) -> MemoryStats:
        """Health check — system stats and recommendations."""
        total = self._store.count()
        active = self._store.count(active_only=True)
        inactive = self._store.count(inactive_only=True)

        # Compute average strength of active memories
        all_active = self._store.get_active_memories()
        avg_strength = (
            sum(m.strength for m in all_active) / len(all_active)
            if all_active
            else 0.0
        )
        below_threshold = sum(
            1 for m in all_active
            if m.strength < self._config.strength_threshold
        )

        # Check last decay run
        last_decay = self._store.get_meta("last_decay_run")
        hours_since = None
        if last_decay:
            last_dt = datetime.fromisoformat(last_decay)
            hours_since = (_now() - last_dt).total_seconds() / 3600.0

        # Embedding model check
        model_mismatches = 0
        if self._embedding_callback and all_active:
            # Check first memory's model against others
            models = {m.embedding_model for m in all_active if m.embedding_model}
            if len(models) > 1:
                model_mismatches = len(models) - 1

        recommendations: list[str] = []
        if hours_since is not None and hours_since > 24:
            recommendations.append(
                f"Consider running mem.decay() — last run {hours_since:.1f}h ago"
            )
        elif last_decay is None:
            recommendations.append(
                "No decay runs recorded. Call mem.decay() periodically to manage memory health."
            )
        if below_threshold > 0:
            recommendations.append(
                f"{below_threshold} memories below strength threshold — consider running mem.purge()"
            )
        if model_mismatches > 0:
            recommendations.append(
                f"{model_mismatches + 1} different embedding models detected. "
                "Consider running mem.reembed() for consistency."
            )

        return MemoryStats(
            total_memories=total,
            active_memories=active,
            soft_deleted=inactive,
            avg_strength=round(avg_strength, 4),
            memories_below_threshold=below_threshold,
            last_decay_run=last_decay,
            hours_since_last_decay=round(hours_since, 2) if hours_since else None,
            embedding_model_mismatches=model_mismatches,
            recommendations=recommendations,
        )

    def build_context(
        self,
        user_id: str,
        query: str,
        max_tokens: int = 500,
        namespace: Optional[str] = None,
    ) -> str:
        """Build a context string suitable for LLM system prompts.

        Recalls relevant memories and formats them into a concise string.
        """
        memories = self.recall(
            user_id=user_id,
            query=query,
            namespace=namespace,
        )

        if not memories:
            return ""

        parts: list[str] = []
        approx_tokens = 0
        for m in memories:
            line = f"- {m.content} [{m.memory_type.value}, confidence: {m.confidence:.1f}]"
            # Rough approximation: 1 token ≈ 4 chars
            line_tokens = len(line) // 4
            if approx_tokens + line_tokens > max_tokens:
                break
            parts.append(line)
            approx_tokens += line_tokens

        return "Known about this user:\n" + "\n".join(parts)

    def close(self) -> None:
        """Close the database connection and clean up resources."""
        self._store.close()
        self._vector_cache.invalidate_all()
