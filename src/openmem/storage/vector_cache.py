"""In-memory numpy vector cache — per-user, lazily loaded, LRU eviction.

This is a derived cache of SQLite, not a co-equal store. On any inconsistency,
the cache is rebuilt from SQLite (the single source of truth).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional

import numpy as np


class _UserIndex:
    """Embedding index for a single user."""

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.matrix: Optional[np.ndarray] = None  # shape: (n, dim)
        self.lock = threading.RLock()  # RW lock (reentrant for convenience)

    def build(self, items: list[tuple[str, list[float]]]) -> None:
        """Build the index from (id, embedding) pairs."""
        with self.lock:
            if not items:
                self.ids = []
                self.matrix = None
                return
            self.ids = [item[0] for item in items]
            self.matrix = np.array(
                [item[1] for item in items], dtype=np.float32
            )

    def add(self, memory_id: str, embedding: list[float]) -> None:
        """Add a single embedding to the index."""
        with self.lock:
            vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
            if self.matrix is None:
                self.matrix = vec
                self.ids = [memory_id]
            else:
                self.matrix = np.vstack([self.matrix, vec])
                self.ids.append(memory_id)

    def remove(self, memory_id: str) -> bool:
        """Remove an embedding by memory ID. Returns True if found."""
        with self.lock:
            if memory_id not in self.ids:
                return False
            idx = self.ids.index(memory_id)
            self.ids.pop(idx)
            if self.matrix is not None:
                self.matrix = np.delete(self.matrix, idx, axis=0)
                if self.matrix.shape[0] == 0:
                    self.matrix = None
            return True

    def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Cosine similarity search. Returns (memory_id, score) pairs, descending."""
        with self.lock:
            if self.matrix is None or len(self.ids) == 0:
                return []

            query = np.array(query_embedding, dtype=np.float32)
            query_norm = np.linalg.norm(query)
            if query_norm == 0:
                return []
            query = query / query_norm

            # Normalize rows
            norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.maximum(norms, 1e-10)
            normalized = self.matrix / norms

            # Cosine similarity via matrix-vector product
            scores = normalized @ query  # shape: (n,)

            # Get top-k
            k = min(top_k, len(self.ids))
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            return [
                (self.ids[i], float(scores[i]))
                for i in top_indices
            ]

    @property
    def size(self) -> int:
        return len(self.ids)


class VectorCache:
    """Per-user vector cache with LRU eviction.

    Each user's embeddings are loaded lazily from SQLite on first access.
    An LRU policy evicts the least recently accessed user's index when
    the cache exceeds max_users.
    """

    def __init__(self, max_users: int = 50) -> None:
        self._max_users = max_users
        self._cache: OrderedDict[str, _UserIndex] = OrderedDict()
        self._lock = threading.Lock()

    def _evict_if_needed(self) -> None:
        """Evict oldest user if cache is full."""
        while len(self._cache) > self._max_users:
            self._cache.popitem(last=False)

    def get_or_create(self, user_key: str) -> _UserIndex:
        """Get an existing user index or create an empty one.

        The caller is responsible for populating it via build() if new.
        Moves the user to the end of the LRU queue.
        """
        with self._lock:
            if user_key in self._cache:
                self._cache.move_to_end(user_key)
                return self._cache[user_key]

            index = _UserIndex()
            self._cache[user_key] = index
            self._evict_if_needed()
            return index

    def build_user_index(
        self, user_key: str, items: list[tuple[str, list[float]]]
    ) -> _UserIndex:
        """Build (or rebuild) a user's index from (id, embedding) pairs."""
        index = self.get_or_create(user_key)
        index.build(items)
        return index

    def invalidate(self, user_key: str) -> None:
        """Remove a user's index from cache (forces rebuild on next access)."""
        with self._lock:
            self._cache.pop(user_key, None)

    def invalidate_all(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()

    def has_user(self, user_key: str) -> bool:
        """Check if a user's index is currently cached."""
        return user_key in self._cache

    def add_to_user(
        self, user_key: str, memory_id: str, embedding: list[float]
    ) -> None:
        """Add a single embedding to an existing user index.

        If the user is not cached, this is a no-op (will be loaded on next access).
        """
        with self._lock:
            if user_key not in self._cache:
                return
            self._cache[user_key].add(memory_id, embedding)
            self._cache.move_to_end(user_key)

    def remove_from_user(self, user_key: str, memory_id: str) -> None:
        """Remove a single embedding from a user's index."""
        with self._lock:
            if user_key not in self._cache:
                return
            self._cache[user_key].remove(memory_id)

    def search(
        self, user_key: str, query_embedding: list[float], top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Search a user's index. Returns empty if user not cached."""
        with self._lock:
            if user_key not in self._cache:
                return []
            self._cache.move_to_end(user_key)
            index = self._cache[user_key]
        # Release cache lock before doing the actual computation
        return index.search(query_embedding, top_k)

    @property
    def cached_user_count(self) -> int:
        return len(self._cache)
