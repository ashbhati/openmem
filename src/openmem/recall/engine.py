"""RecallEngine — hybrid retrieval with reinforcement learning."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from ..config import OpenMemConfig
from ..models import Memory
from ..storage.sqlite_store import SQLiteStore
from ..storage.vector_cache import VectorCache
from ..types import EmbeddingCallback, MemoryType
from .ranking import apply_filters, merge_and_rank
from .search import keyword_search, semantic_search

# Cap on stored access timestamps per memory. Very old timestamps contribute
# near-zero to the ACT-R activation sum, so trimming has negligible effect.
MAX_ACCESS_TIMESTAMPS = 500


def _escape_fts5_query(query: str) -> str:
    """Wrap each term in double quotes to avoid FTS5 syntax errors.

    FTS5 treats characters like -, *, (, ) as operators. Quoting each
    token ensures they are treated as literal search terms.
    """
    terms = query.split()
    if not terms:
        return query
    return " ".join(f'"{term}"' for term in terms)


class RecallEngine:
    """Hybrid search engine combining semantic and keyword retrieval.

    Supports optional reinforcement: accessing a memory bumps its
    access_count, access_timestamps, and last_accessed fields, which
    feeds into the ACT-R decay model.
    """

    def __init__(
        self,
        store: SQLiteStore,
        vector_cache: VectorCache,
        embedding_callback: Optional[EmbeddingCallback],
        config: OpenMemConfig,
    ) -> None:
        self._store = store
        self._vector_cache = vector_cache
        self._embedding_callback = embedding_callback
        self._config = config

    def recall(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        min_confidence: float = 0.0,
        min_strength: float = 0.0,
        memory_types: Optional[list[MemoryType]] = None,
        namespace: Optional[str] = None,
        reinforce: bool = True,
    ) -> list[Memory]:
        """Retrieve relevant memories using hybrid search.

        1. Generate query embedding
        2. Run semantic search (vector cosine similarity)
        3. Run keyword search (FTS5 full-text)
        4. Merge and rank results
        5. Apply filters (confidence, strength, type, namespace)
        6. Optionally reinforce accessed memories

        Returns a list of Memory objects sorted by relevance.
        """
        ns = namespace or self._config.default_namespace

        # --- Semantic search ---
        sem_results: list[tuple[Memory, float]] = []
        if self._embedding_callback is not None:
            query_embedding = self._embedding_callback(query)
            sem_results = semantic_search(
                vector_cache=self._vector_cache,
                store=self._store,
                user_id=user_id,
                namespace=ns,
                query_embedding=query_embedding,
                top_k=top_k,
            )

        # --- Keyword search ---
        kw_results: list[tuple[Memory, float]] = []
        escaped_query = _escape_fts5_query(query)
        if escaped_query.strip():
            try:
                kw_results = keyword_search(
                    store=self._store,
                    user_id=user_id,
                    namespace=ns,
                    query=escaped_query,
                    top_k=top_k,
                )
            except Exception:
                # FTS5 queries can still fail on edge cases; degrade gracefully
                kw_results = []

        # --- Merge and rank ---
        merged = merge_and_rank(
            semantic_results=sem_results,
            keyword_results=kw_results,
            semantic_weight=self._config.semantic_weight,
            keyword_weight=self._config.keyword_weight,
            top_k=top_k,
        )

        # --- Apply filters ---
        filtered = apply_filters(
            memories_with_scores=merged,
            min_confidence=min_confidence,
            min_strength=min_strength,
            memory_types=memory_types,
            namespace=ns,
        )

        memories = [memory for memory, _score in filtered]

        # --- Reinforce ---
        if reinforce:
            now = datetime.now(timezone.utc)
            for memory in memories:
                memory.access_timestamps.append(now)
                if len(memory.access_timestamps) > MAX_ACCESS_TIMESTAMPS:
                    memory.access_timestamps = memory.access_timestamps[-MAX_ACCESS_TIMESTAMPS:]
                memory.access_count += 1
                memory.last_accessed = now
                self._store.update(memory)

        return memories
