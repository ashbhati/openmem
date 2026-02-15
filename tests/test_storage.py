"""Tests for SQLite storage and vector cache."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from openmem import Memory, MemoryLifespan, MemorySource, MemoryType
from openmem.storage.sqlite_store import SQLiteStore
from openmem.storage.vector_cache import VectorCache


def _make_memory(
    id: str = "test_id",
    user_id: str = "user_1",
    content: str = "Test content",
    namespace: str = "default",
    embedding: list[float] | None = None,
) -> Memory:
    """Create a Memory object for testing."""
    now = datetime.now(timezone.utc)
    return Memory(
        id=id,
        user_id=user_id,
        namespace=namespace,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        memory_type=MemoryType.FACT,
        source=MemorySource.EXPLICIT,
        confidence=1.0,
        strength=1.0,
        created_at=now,
        last_accessed=now,
        access_count=0,
        access_timestamps=[now],
        lifespan=MemoryLifespan.LONG_TERM,
        version=1,
        is_active=True,
        embedding=embedding or [],
        embedding_model="",
        metadata={},
    )


# ---------------------------------------------------------------------------
# SQLite CRUD tests
# ---------------------------------------------------------------------------


class TestSQLiteCRUD:
    def test_sqlite_crud_operations(self, tmp_path: Path):
        store = SQLiteStore(path=tmp_path / "crud.db")

        # Create
        m = _make_memory(id="m1", content="Hello world")
        store.add(m)

        # Read
        fetched = store.get("m1")
        assert fetched is not None
        assert fetched.content == "Hello world"
        assert fetched.user_id == "user_1"

        # Update
        fetched.content = "Updated content"
        fetched.version = 2
        store.update(fetched)
        refetched = store.get("m1")
        assert refetched.content == "Updated content"
        assert refetched.version == 2

        # Delete
        assert store.delete("m1") is True
        assert store.get("m1") is None
        assert store.delete("m1") is False

        store.close()

    def test_ghost_memory_prevention(self, tmp_path: Path):
        """Deleted memories should not appear in searches or lists."""
        store = SQLiteStore(path=tmp_path / "ghost.db")

        m = _make_memory(id="ghost", content="Ghost memory test")
        store.add(m)

        # Verify it exists in list
        listed = store.list(user_id="user_1")
        assert any(mem.id == "ghost" for mem in listed)

        # Delete it
        store.delete("ghost")

        # Verify it's gone from list
        listed = store.list(user_id="user_1")
        assert not any(mem.id == "ghost" for mem in listed)

        # Verify it's gone from get
        assert store.get("ghost") is None

        store.close()


# ---------------------------------------------------------------------------
# FTS tests
# ---------------------------------------------------------------------------


class TestFTS:
    def test_fts_search_basic(self, tmp_path: Path):
        store = SQLiteStore(path=tmp_path / "fts.db")

        store.add(_make_memory(id="m1", content="User likes Python programming"))
        store.add(_make_memory(id="m2", content="User enjoys JavaScript frameworks"))
        store.add(_make_memory(id="m3", content="User prefers dark mode"))

        results = store.fts_search(query='"Python"', user_id="user_1")
        assert len(results) >= 1
        found_ids = {m.id for m, _ in results}
        assert "m1" in found_ids

        store.close()

    def test_fts_sync_after_update(self, tmp_path: Path):
        """FTS index should reflect updated content."""
        store = SQLiteStore(path=tmp_path / "fts_update.db")

        m = _make_memory(id="m1", content="User likes coffee")
        store.add(m)

        # Verify initial search works
        results = store.fts_search(query='"coffee"', user_id="user_1")
        assert len(results) >= 1

        # Update content
        m.content = "User prefers tea"
        store.update(m)

        # Old term should not match
        results_old = store.fts_search(query='"coffee"', user_id="user_1")
        assert len(results_old) == 0

        # New term should match
        results_new = store.fts_search(query='"tea"', user_id="user_1")
        assert len(results_new) >= 1

        store.close()


# ---------------------------------------------------------------------------
# Vector cache tests
# ---------------------------------------------------------------------------


class TestVectorCache:
    def test_vector_cache_lru_eviction(self):
        """Cache should evict oldest user when full."""
        cache = VectorCache(max_users=2)

        cache.build_user_index("user_a:default", [("m1", [1.0, 0.0])])
        cache.build_user_index("user_b:default", [("m2", [0.0, 1.0])])

        # Both should be present
        assert cache.has_user("user_a:default")
        assert cache.has_user("user_b:default")

        # Adding third user should evict the oldest (user_a)
        cache.build_user_index("user_c:default", [("m3", [1.0, 1.0])])
        assert not cache.has_user("user_a:default")
        assert cache.has_user("user_b:default")
        assert cache.has_user("user_c:default")

    def test_vector_cache_add_remove(self):
        cache = VectorCache()
        cache.build_user_index("u:d", [("m1", [1.0, 0.0, 0.0])])

        # Add
        cache.add_to_user("u:d", "m2", [0.0, 1.0, 0.0])
        results = cache.search("u:d", [0.0, 1.0, 0.0], top_k=5)
        ids = {r[0] for r in results}
        assert "m2" in ids

        # Remove
        cache.remove_from_user("u:d", "m2")
        results = cache.search("u:d", [0.0, 1.0, 0.0], top_k=5)
        ids = {r[0] for r in results}
        assert "m2" not in ids

    def test_vector_cache_search_returns_sorted(self):
        """Search results should be sorted by descending cosine similarity."""
        cache = VectorCache()
        cache.build_user_index("u:d", [
            ("m1", [1.0, 0.0, 0.0]),  # Only matches dim 0
            ("m2", [0.0, 1.0, 0.0]),  # Only matches dim 1
            ("m3", [1.0, 1.0, 0.0]),  # Matches dim 0 and 1
        ])

        # Query close to m3
        results = cache.search("u:d", [1.0, 1.0, 0.0], top_k=3)
        assert len(results) == 3
        # m3 should be first (exact match)
        assert results[0][0] == "m3"
        # Scores should be descending
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i + 1][1]
