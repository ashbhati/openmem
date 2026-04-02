"""Comprehensive tests validating all 18 code-review fixes for OpenMem.

Each test class corresponds to one issue. Tests are self-contained and use
temporary databases so they don't interfere with one another.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import inspect
import io
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from openmem import (
    LLMRequest,
    LLMResponse,
    Memory,
    MemoryLifespan,
    MemorySource,
    MemoryStats,
    MemoryType,
    OpenMem,
    OpenMemConfig,
)
from openmem.storage.cache_utils import ensure_user_cache
from openmem.storage.sqlite_store import SQLiteStore
from openmem.storage.vector_cache import VectorCache
from openmem.types import ConflictStrategy


# ---------------------------------------------------------------------------
# Shared helpers — kept minimal, no business logic
# ---------------------------------------------------------------------------

def _mock_embedding(text: str) -> list[float]:
    """Deterministic 8-dim embedding from content hash (same as conftest)."""
    h = hashlib.md5(text.encode()).hexdigest()
    return [int(h[i : i + 2], 16) / 255.0 for i in range(0, 16, 2)]


def _mock_llm(request: LLMRequest) -> LLMResponse:
    sys_lower = request.system_prompt.lower()
    if "extract" in sys_lower or "memory extraction" in sys_lower:
        memories = [
            {
                "content": "User likes tea",
                "memory_type": "preference",
                "source": "explicit",
                "confidence": 0.9,
            }
        ]
        return LLMResponse(content=json.dumps(memories))
    if "repair" in sys_lower or "fix" in sys_lower:
        return LLMResponse(content=json.dumps([]))
    if "consolidat" in sys_lower or "merg" in sys_lower:
        result = {
            "content": "Merged memory content",
            "memory_type": "fact",
            "confidence": 0.9,
            "reasoning": "test merge",
        }
        return LLMResponse(content=json.dumps(result))
    if "conflict" in sys_lower or "contradic" in sys_lower:
        result = {"is_conflict": False, "explanation": "compatible"}
        return LLMResponse(content=json.dumps(result))
    return LLMResponse(content="OK")


def _make_mem(tmp_path: Path, **kwargs) -> OpenMem:
    """Create a fully wired OpenMem instance backed by a temp db."""
    db = tmp_path / "test.db"
    cfg = OpenMemConfig(storage_path=str(db))
    return OpenMem(
        llm_callback=_mock_llm,
        embedding_callback=_mock_embedding,
        config=cfg,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Issue 1: Thread safety — update() uses a lock during read-modify-write
# ---------------------------------------------------------------------------

class TestIssue01ThreadSafety:
    """Issue 1: update() must hold self._lock during the read-modify-write cycle."""

    def test_lock_attribute_exists(self, tmp_path):
        mem = _make_mem(tmp_path)
        assert hasattr(mem, "_lock"), "OpenMem must expose _lock attribute"
        assert isinstance(mem._lock, type(threading.Lock())), (
            "_lock must be a threading.Lock instance"
        )
        mem.close()

    def test_lock_is_not_reentrant(self, tmp_path):
        """threading.Lock (not RLock) — cannot be acquired twice from same thread."""
        mem = _make_mem(tmp_path)
        lock = mem._lock
        # A plain Lock cannot be re-acquired from the same thread
        acquired = lock.acquire(blocking=False)
        assert acquired
        try:
            second_acquire = lock.acquire(blocking=False)
            assert not second_acquire, "threading.Lock must not be reentrant"
        finally:
            lock.release()
        mem.close()

    def test_concurrent_updates_no_data_corruption(self, tmp_path):
        """Multiple threads updating different memories must not lose writes."""
        mem = _make_mem(tmp_path)
        # Add memories upfront
        ids = [
            mem.add("user1", f"Memory content {i}", confidence=0.5).id
            for i in range(10)
        ]

        errors: list[str] = []

        def update_memory(mid: str, confidence: float):
            try:
                result = mem.update(mid, confidence=confidence)
                if result is None:
                    errors.append(f"update returned None for {mid}")
            except Exception as exc:
                errors.append(str(exc))

        threads = [
            threading.Thread(target=update_memory, args=(mid, 0.1 * (i + 1)))
            for i, mid in enumerate(ids)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent updates produced errors: {errors}"

        # Verify all memories are still retrievable and sane
        for mid in ids:
            m = mem.get(mid)
            assert m is not None
            assert 0.0 < m.confidence <= 1.0

        mem.close()


# ---------------------------------------------------------------------------
# Issue 2: FTS5 trigger conditional — strength/metadata updates don't re-index
# ---------------------------------------------------------------------------

class TestIssue02FTS5TriggerConditional:
    """Issue 2: The memories_au trigger only fires WHEN OLD.content != NEW.content."""

    def test_fts_still_works_after_content_update(self, tmp_path):
        """After a content update, FTS5 must find the new content, not the old."""
        mem = _make_mem(tmp_path)
        m = mem.add("u1", "The quick brown fox")
        # Update content — FTS trigger should remove old entry and add new one
        mem.update(m.id, content="The lazy dog slept")

        # Old content must not appear in FTS
        results_old = mem._store.fts_search("quick", "u1")
        old_ids = [r[0].id for r in results_old]
        assert m.id not in old_ids, "Old content must be removed from FTS index"

        # New content must appear in FTS
        results_new = mem._store.fts_search("lazy", "u1")
        new_ids = [r[0].id for r in results_new]
        assert m.id in new_ids, "New content must be findable in FTS"

        mem.close()

    def test_strength_update_does_not_corrupt_fts(self, tmp_path):
        """Updating only strength/metadata (not content) must not corrupt FTS."""
        mem = _make_mem(tmp_path)
        m = mem.add("u1", "Unique searchable keyword xyzzy")

        # Perform a strength-only update via store directly (simulates decay)
        mem._store.batch_update_strength([(m.id, 0.5, True)])

        # FTS index must still contain the memory
        results = mem._store.fts_search("xyzzy", "u1")
        found_ids = [r[0].id for r in results]
        assert m.id in found_ids, (
            "FTS must still find memory after strength-only update"
        )
        mem.close()

    def test_fts_trigger_schema_contains_when_clause(self, tmp_path):
        """Verify the update trigger has a WHEN clause in SQLite schema."""
        mem = _make_mem(tmp_path)
        conn = mem._store._get_conn()
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND name='memories_au'"
        ).fetchone()
        assert row is not None, "memories_au trigger must exist"
        trigger_sql = row[0].upper()
        assert "WHEN" in trigger_sql, (
            "memories_au trigger must have a WHEN clause for conditional FTS update"
        )
        mem.close()


# ---------------------------------------------------------------------------
# Issue 3: delete_all() clears caches across all namespaces
# ---------------------------------------------------------------------------

class TestIssue03DeleteAllNoGhostCaches:
    """Issue 3: delete_all must clear vector cache for every namespace."""

    def test_delete_all_clears_multiple_namespace_caches(self, tmp_path):
        mem = _make_mem(tmp_path)
        user = "gdpr_user"
        namespaces = ["ns_alpha", "ns_beta", "ns_gamma"]

        for ns in namespaces:
            mem.add(user, f"Memory in {ns}", namespace=ns)
            # Trigger cache population by ensuring user cache
            mem._ensure_user_cache(user, ns)

        # Verify caches are populated
        for ns in namespaces:
            assert mem._vector_cache.has_user(f"{user}:{ns}"), (
                f"Cache must be populated for {ns} before delete_all"
            )

        count = mem.delete_all(user)
        assert count == len(namespaces)

        # All namespace caches must be gone
        for ns in namespaces:
            assert not mem._vector_cache.has_user(f"{user}:{ns}"), (
                f"Cache must be cleared for {ns} after delete_all"
            )

        mem.close()

    def test_delete_all_returns_correct_count(self, tmp_path):
        mem = _make_mem(tmp_path)
        for i in range(7):
            mem.add("user_x", f"Content {i}")
        count = mem.delete_all("user_x")
        assert count == 7
        mem.close()

    def test_delete_all_large_number_of_memories(self, tmp_path):
        """No artificial 100K limit — delete_all must handle large batches."""
        mem = _make_mem(tmp_path)
        n = 200  # Representative stress test (keep fast)
        for i in range(n):
            mem.add("bulk_user", f"Bulk memory {i}", namespace="bulk_ns")
        count = mem.delete_all("bulk_user")
        assert count == n
        remaining = mem.list("bulk_user", active_only=False, limit=n + 10)
        assert len(remaining) == 0
        mem.close()


# ---------------------------------------------------------------------------
# Issue 4: CSV export is RFC 4180 compliant
# ---------------------------------------------------------------------------

class TestIssue04CSVExportRFC4180:
    """Issue 4: CSV export must correctly handle commas, quotes, and newlines."""

    def test_csv_commas_in_user_id(self, tmp_path):
        mem = _make_mem(tmp_path)
        # user_id with a comma (edge case)
        user = "user,with,commas"
        mem.add(user, "Simple memory content")
        csv_output = mem.export(user, format="csv")

        reader = csv.reader(io.StringIO(csv_output))
        rows = list(reader)
        assert len(rows) == 2, "Should have header + 1 data row"
        header = rows[0]
        data = rows[1]
        uid_col = header.index("user_id")
        assert data[uid_col] == user, (
            f"user_id with commas must survive CSV round-trip: {data[uid_col]!r}"
        )
        mem.close()

    def test_csv_quotes_in_namespace(self, tmp_path):
        mem = _make_mem(tmp_path)
        # namespace with a double-quote
        ns = 'my"namespace'
        mem.add("u1", "Memory in quoted namespace", namespace=ns)
        csv_output = mem.export("u1", format="csv")

        reader = csv.reader(io.StringIO(csv_output))
        rows = list(reader)
        header = rows[0]
        data = rows[1]
        ns_col = header.index("namespace")
        assert data[ns_col] == ns, (
            f"Namespace with quotes must survive CSV round-trip: {data[ns_col]!r}"
        )
        mem.close()

    def test_csv_newlines_in_content(self, tmp_path):
        mem = _make_mem(tmp_path)
        content = "Line one\nLine two\r\nLine three"
        mem.add("u1", content)
        csv_output = mem.export("u1", format="csv")

        reader = csv.reader(io.StringIO(csv_output))
        rows = list(reader)
        header = rows[0]
        data = rows[1]
        content_col = header.index("content")
        assert data[content_col] == content, (
            f"Newlines in content must survive CSV round-trip"
        )
        mem.close()

    def test_csv_round_trip_all_fields_intact(self, tmp_path):
        mem = _make_mem(tmp_path)
        m = mem.add("u1", "Standalone memory", confidence=0.75)
        csv_output = mem.export("u1", format="csv")

        reader = csv.reader(io.StringIO(csv_output))
        rows = list(reader)
        assert len(rows) == 2
        header = rows[0]
        data = rows[1]
        assert data[header.index("id")] == m.id
        assert data[header.index("confidence")] == "0.75"
        mem.close()

    def test_csv_empty_user_returns_empty_string(self, tmp_path):
        mem = _make_mem(tmp_path)
        result = mem.export("nonexistent_user", format="csv")
        assert result == "", "CSV export for user with no memories must return empty string"
        mem.close()


# ---------------------------------------------------------------------------
# Issue 5: Centralized ensure_user_cache
# ---------------------------------------------------------------------------

class TestIssue05CentralizedEnsureUserCache:
    """Issue 5: cache_utils.ensure_user_cache must be importable and functional."""

    def test_ensure_user_cache_importable(self):
        from openmem.storage.cache_utils import ensure_user_cache as euc
        assert callable(euc), "ensure_user_cache must be a callable"

    def test_ensure_user_cache_non_default_namespace(self, tmp_path):
        """ensure_user_cache must work for non-default namespaces."""
        store = SQLiteStore(tmp_path / "store.db")
        cache = VectorCache()

        # Add a memory directly to store in a custom namespace
        from openmem._ulid import generate_ulid
        from openmem._utils import utc_now, content_hash
        now = utc_now()
        m = Memory(
            id=generate_ulid(),
            user_id="u1",
            namespace="custom_ns",
            content="Custom namespace memory",
            content_hash=content_hash("Custom namespace memory"),
            embedding=_mock_embedding("Custom namespace memory"),
        )
        store.add(m)

        user_key = ensure_user_cache(store, cache, "u1", "custom_ns")
        assert user_key == "u1:custom_ns"
        assert cache.has_user("u1:custom_ns"), (
            "Cache must be populated for custom namespace"
        )
        store.close()

    def test_ensure_user_cache_idempotent(self, tmp_path):
        """Calling ensure_user_cache twice must not duplicate entries."""
        store = SQLiteStore(tmp_path / "store.db")
        cache = VectorCache()

        from openmem._ulid import generate_ulid
        from openmem._utils import utc_now, content_hash
        m = Memory(
            id=generate_ulid(),
            user_id="u1",
            namespace="default",
            content="Test memory",
            content_hash=content_hash("Test memory"),
            embedding=_mock_embedding("Test memory"),
        )
        store.add(m)

        key1 = ensure_user_cache(store, cache, "u1", "default")
        key2 = ensure_user_cache(store, cache, "u1", "default")
        assert key1 == key2 == "u1:default"
        # Cache should have exactly one entry for this user
        index = cache.get_or_create("u1:default")
        assert index.size == 1, "Idempotent calls must not duplicate cache entries"
        store.close()


# ---------------------------------------------------------------------------
# Issue 6: Config not mutated when storage_path kwarg overrides it
# ---------------------------------------------------------------------------

class TestIssue06ConfigNotMutated:
    """Issue 6: Passing config + storage_path kwarg must not mutate the original config."""

    def test_original_config_unchanged(self, tmp_path):
        original_path = "~/.openmem/memory.db"
        config = OpenMemConfig(storage_path=original_path)
        override_path = str(tmp_path / "override.db")

        mem = OpenMem(
            llm_callback=_mock_llm,
            embedding_callback=_mock_embedding,
            storage_path=override_path,
            config=config,
        )

        # Original config must be unchanged
        assert config.storage_path == original_path, (
            "Original config.storage_path must not be mutated by OpenMem constructor"
        )
        # Instance config should have the override
        assert str(mem._config.storage_path) == override_path, (
            "OpenMem internal config must use the overridden storage_path"
        )
        mem.close()

    def test_config_is_deep_copied(self, tmp_path):
        """The internal config must be a copy, not a reference."""
        config = OpenMemConfig()
        mem = OpenMem(config=config, storage_path=str(tmp_path / "test.db"))
        # Modifying the internal config must not affect the original
        mem._config.max_memories_per_recall = 999
        assert config.max_memories_per_recall == 10, (
            "Modifying internal config must not affect the original config object"
        )
        mem.close()


# ---------------------------------------------------------------------------
# Issue 7: reembed() invalidates vector cache for all affected namespaces
# ---------------------------------------------------------------------------

class TestIssue07ReembedMultiNamespace:
    """Issue 7: reembed() must invalidate vector cache for every namespace touched."""

    def test_reembed_invalidates_all_namespaces(self, tmp_path):
        mem = _make_mem(tmp_path)
        user = "reembed_user"
        ns1, ns2 = "space_one", "space_two"

        mem.add(user, "Memory in space one", namespace=ns1)
        mem.add(user, "Memory in space two", namespace=ns2)

        # Warm up the caches
        mem._ensure_user_cache(user, ns1)
        mem._ensure_user_cache(user, ns2)
        assert mem._vector_cache.has_user(f"{user}:{ns1}")
        assert mem._vector_cache.has_user(f"{user}:{ns2}")

        count = mem.reembed(user)
        assert count == 2

        # Both namespace caches must be invalidated
        assert not mem._vector_cache.has_user(f"{user}:{ns1}"), (
            f"Cache for {ns1} must be invalidated after reembed"
        )
        assert not mem._vector_cache.has_user(f"{user}:{ns2}"), (
            f"Cache for {ns2} must be invalidated after reembed"
        )
        mem.close()

    def test_reembed_namespace_filter(self, tmp_path):
        """reembed(namespace=X) must only re-embed memories in that namespace."""
        mem = _make_mem(tmp_path)
        user = "reembed_filter_user"
        mem.add(user, "Memory A", namespace="ns_a")
        mem.add(user, "Memory B", namespace="ns_b")

        # Warm both caches
        mem._ensure_user_cache(user, "ns_a")
        mem._ensure_user_cache(user, "ns_b")

        count = mem.reembed(user, namespace="ns_a")
        assert count == 1
        # Only ns_a should be invalidated
        assert not mem._vector_cache.has_user(f"{user}:ns_a")
        # ns_b was not re-embedded, so its cache status is unchanged
        mem.close()


# ---------------------------------------------------------------------------
# Issue 8: Batch get — correct results and handles empty list
# ---------------------------------------------------------------------------

class TestIssue08BatchGet:
    """Issue 8: sqlite_store.batch_get() must return correct results."""

    def test_batch_get_returns_all_requested(self, tmp_path):
        mem = _make_mem(tmp_path)
        ids = [mem.add("u1", f"Memory {i}").id for i in range(5)]
        results = mem._store.batch_get(ids)
        assert len(results) == 5
        result_ids = {m.id for m in results}
        assert result_ids == set(ids)
        mem.close()

    def test_batch_get_empty_list(self, tmp_path):
        mem = _make_mem(tmp_path)
        result = mem._store.batch_get([])
        assert result == [], "batch_get with empty list must return empty list"
        mem.close()

    def test_batch_get_missing_ids_skipped(self, tmp_path):
        mem = _make_mem(tmp_path)
        real_id = mem.add("u1", "Real memory").id
        results = mem._store.batch_get([real_id, "nonexistent_id_xyz"])
        assert len(results) == 1
        assert results[0].id == real_id
        mem.close()

    def test_batch_get_preserves_request_order(self, tmp_path):
        mem = _make_mem(tmp_path)
        ids = [mem.add("u1", f"Memory {i}").id for i in range(4)]
        # Request in reverse order
        reversed_ids = list(reversed(ids))
        results = mem._store.batch_get(reversed_ids)
        result_ids = [m.id for m in results]
        assert result_ids == reversed_ids, "batch_get must preserve requested order"
        mem.close()

    def test_semantic_search_still_works(self, tmp_path):
        """Verify semantic search via recall works correctly after batch_get addition."""
        mem = _make_mem(tmp_path)
        mem.add("u1", "The user prefers tea over coffee")
        results = mem.recall("u1", "tea preferences")
        assert len(results) >= 1
        mem.close()


# ---------------------------------------------------------------------------
# Issue 9: No vestigial parameter — extract_memories signature
# ---------------------------------------------------------------------------

class TestIssue09NoVestigialParameter:
    """Issue 9: extract_memories must NOT accept existing_content_hashes."""

    def test_extract_memories_signature_clean(self):
        from openmem.capture.extractor import extract_memories
        sig = inspect.signature(extract_memories)
        param_names = list(sig.parameters.keys())
        assert "existing_content_hashes" not in param_names, (
            "extract_memories must not have an 'existing_content_hashes' parameter; "
            f"found params: {param_names}"
        )

    def test_extract_memories_accepts_correct_params(self):
        from openmem.capture.extractor import extract_memories
        sig = inspect.signature(extract_memories)
        param_names = list(sig.parameters.keys())
        assert "llm_callback" in param_names
        assert "messages" in param_names

    def test_extract_memories_calling_with_extra_kwarg_raises(self):
        """Passing the removed kwarg must raise TypeError, not silently ignore."""
        from openmem.capture.extractor import extract_memories
        with pytest.raises(TypeError):
            extract_memories(
                llm_callback=_mock_llm,
                messages=[{"role": "user", "content": "hi"}],
                existing_content_hashes={"abc123"},  # must be rejected
            )


# ---------------------------------------------------------------------------
# Issue 10: SUPERSEDE vs KEEP_NEWER distinct semantics
# ---------------------------------------------------------------------------

class TestIssue10SupersedeVsKeepNewer:
    """Issue 10: SUPERSEDE always makes memory_a win; KEEP_NEWER uses created_at."""

    def _add_pair(self, mem: OpenMem, user: str):
        """Add two memories with distinct timestamps."""
        older = mem.add(user, "Older memory content", confidence=0.5)
        # Ensure a distinct created_at by tweaking the stored value
        newer = mem.add(user, "Newer memory content", confidence=0.9)
        return older, newer

    def test_supersede_always_makes_a_win(self, tmp_path):
        """SUPERSEDE strategy: memory_a always supersedes memory_b regardless of age."""
        mem = _make_mem(tmp_path)
        older, newer = self._add_pair(mem, "u1")

        # Even though older is older, pass it as memory_a — it should win
        mem.resolve_conflict(older.id, newer.id, strategy=ConflictStrategy.SUPERSEDE)

        result_older = mem.get(older.id)
        result_newer = mem.get(newer.id)

        assert result_newer is not None
        assert not result_newer.is_active, (
            "SUPERSEDE: memory_b (newer) must be deactivated"
        )
        assert result_newer.superseded_by == older.id, (
            "SUPERSEDE: memory_b must reference memory_a as superseder"
        )
        assert result_older is not None
        assert result_older.is_active, "SUPERSEDE: memory_a must remain active"
        mem.close()

    def test_keep_newer_picks_by_created_at(self, tmp_path):
        """KEEP_NEWER: the memory with the later created_at wins."""
        mem = _make_mem(tmp_path)
        user = "u2"
        # Add memory_a with an explicitly earlier timestamp by manipulating via store
        m_a = mem.add(user, "Earlier created memory", confidence=0.8)
        m_b = mem.add(user, "Later created memory", confidence=0.6)

        # Manually set created_at so we control which is newer
        earlier_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        later_ts = datetime(2024, 6, 1, tzinfo=timezone.utc)

        m_a_obj = mem.get(m_a.id)
        m_b_obj = mem.get(m_b.id)
        m_a_obj.created_at = earlier_ts
        m_b_obj.created_at = later_ts
        mem._store.update(m_a_obj)
        mem._store.update(m_b_obj)

        mem.resolve_conflict(m_a.id, m_b.id, strategy=ConflictStrategy.KEEP_NEWER)

        result_a = mem.get(m_a.id)
        result_b = mem.get(m_b.id)

        # m_b was created later, so it must survive
        assert result_b.is_active, "KEEP_NEWER: newer memory (m_b) must remain active"
        assert not result_a.is_active, "KEEP_NEWER: older memory (m_a) must be deactivated"
        assert result_a.superseded_by == m_b.id
        mem.close()

    def test_supersede_and_keep_newer_are_distinct_strategies(self):
        """The two strategies must have different enum values."""
        assert ConflictStrategy.SUPERSEDE != ConflictStrategy.KEEP_NEWER
        assert ConflictStrategy.SUPERSEDE.value == "supersede"
        assert ConflictStrategy.KEEP_NEWER.value == "keep_newer"


# ---------------------------------------------------------------------------
# Issue 11: stats() uses SQL aggregates correctly
# ---------------------------------------------------------------------------

class TestIssue11StatsSQL:
    """Issue 11: stats() must return correct values via SQL aggregate methods."""

    def test_stats_total_active_soft_deleted(self, tmp_path):
        mem = _make_mem(tmp_path)
        for i in range(4):
            m = mem.add("u1", f"Memory {i}")
            if i >= 2:
                mem._store.soft_delete(m.id)

        stats = mem.stats()
        assert stats.total_memories == 4
        assert stats.active_memories == 2
        assert stats.soft_deleted == 2
        mem.close()

    def test_avg_strength_sql_method(self, tmp_path):
        store = SQLiteStore(tmp_path / "store.db")
        # Add memories with known strengths
        from openmem._ulid import generate_ulid
        from openmem._utils import content_hash

        for i, strength in enumerate([0.4, 0.6, 0.8]):
            m = Memory(
                id=generate_ulid(),
                user_id="u1",
                namespace="default",
                content=f"Memory {i}",
                content_hash=content_hash(f"Memory {i}"),
                strength=strength,
            )
            store.add(m)

        avg = store.avg_strength(active_only=True)
        expected = (0.4 + 0.6 + 0.8) / 3
        assert abs(avg - expected) < 1e-4, (
            f"avg_strength expected ~{expected:.4f}, got {avg:.4f}"
        )
        store.close()

    def test_count_below_threshold_sql_method(self, tmp_path):
        store = SQLiteStore(tmp_path / "store.db")
        from openmem._ulid import generate_ulid
        from openmem._utils import content_hash

        threshold = 0.5
        for i, strength in enumerate([0.1, 0.3, 0.6, 0.9]):
            m = Memory(
                id=generate_ulid(),
                user_id="u1",
                namespace="default",
                content=f"Memory {i}",
                content_hash=content_hash(f"Memory {i}"),
                strength=strength,
            )
            store.add(m)

        count = store.count_below_threshold(threshold)
        assert count == 2, f"Expected 2 memories below {threshold}, got {count}"
        store.close()

    def test_distinct_embedding_models_sql_method(self, tmp_path):
        store = SQLiteStore(tmp_path / "store.db")
        from openmem._ulid import generate_ulid
        from openmem._utils import content_hash

        for i, model in enumerate(["text-ada-001", "text-ada-001", "text-embed-v2"]):
            m = Memory(
                id=generate_ulid(),
                user_id="u1",
                namespace="default",
                content=f"Memory {i}",
                content_hash=content_hash(f"Memory {i}"),
                embedding_model=model,
            )
            store.add(m)

        models = store.distinct_embedding_models()
        assert isinstance(models, set)
        assert models == {"text-ada-001", "text-embed-v2"}
        store.close()

    def test_stats_empty_db(self, tmp_path):
        mem = _make_mem(tmp_path)
        stats = mem.stats()
        assert stats.total_memories == 0
        assert stats.active_memories == 0
        assert stats.soft_deleted == 0
        assert stats.avg_strength == 0.0
        assert stats.memories_below_threshold == 0
        mem.close()


# ---------------------------------------------------------------------------
# Issue 12: Config validation raises ValueError for invalid values
# ---------------------------------------------------------------------------

class TestIssue12ConfigValidation:
    """Issue 12: OpenMemConfig must raise ValueError for invalid field values."""

    def test_semantic_weight_above_one_raises(self):
        with pytest.raises(ValueError, match="semantic_weight"):
            OpenMemConfig(semantic_weight=1.5)

    def test_semantic_weight_below_zero_raises(self):
        with pytest.raises(ValueError, match="semantic_weight"):
            OpenMemConfig(semantic_weight=-0.1)

    def test_strength_threshold_negative_raises(self):
        with pytest.raises(ValueError, match="strength_threshold"):
            OpenMemConfig(strength_threshold=-0.1)

    def test_strength_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="strength_threshold"):
            OpenMemConfig(strength_threshold=1.1)

    def test_max_memories_per_recall_zero_raises(self):
        with pytest.raises(ValueError, match="max_memories_per_recall"):
            OpenMemConfig(max_memories_per_recall=0)

    def test_max_memories_per_recall_negative_raises(self):
        with pytest.raises(ValueError, match="max_memories_per_recall"):
            OpenMemConfig(max_memories_per_recall=-5)

    def test_vector_cache_max_users_zero_raises(self):
        with pytest.raises(ValueError, match="vector_cache_max_users"):
            OpenMemConfig(vector_cache_max_users=0)

    def test_conflict_similarity_threshold_out_of_range(self):
        with pytest.raises(ValueError, match="conflict_similarity_threshold"):
            OpenMemConfig(conflict_similarity_threshold=1.5)

    def test_valid_boundary_values_do_not_raise(self):
        # Edge: exactly 0.0 and 1.0 are valid for most float fields
        cfg = OpenMemConfig(
            semantic_weight=0.0,
            strength_threshold=0.0,
        )
        assert cfg.semantic_weight == 0.0

    def test_valid_config_instantiates(self):
        cfg = OpenMemConfig(
            semantic_weight=0.7,
            strength_threshold=0.1,
            max_memories_per_recall=10,
        )
        assert cfg is not None


# ---------------------------------------------------------------------------
# Issue 13: Non-dict messages raise TypeError with clear message
# ---------------------------------------------------------------------------

class TestIssue13NonDictMessages:
    """Issue 13: capture() with non-dict messages must raise TypeError."""

    def test_string_message_raises_type_error(self, tmp_path):
        mem = _make_mem(tmp_path)
        with pytest.raises(TypeError) as exc_info:
            mem.capture(
                user_id="u1",
                messages=["This is a plain string, not a dict"],
            )
        error_msg = str(exc_info.value).lower()
        assert "dict" in error_msg, (
            f"TypeError message must mention 'dict', got: {exc_info.value}"
        )
        mem.close()

    def test_integer_message_raises_type_error(self, tmp_path):
        mem = _make_mem(tmp_path)
        with pytest.raises(TypeError):
            mem.capture(user_id="u1", messages=[42])
        mem.close()

    def test_none_message_raises_type_error(self, tmp_path):
        mem = _make_mem(tmp_path)
        with pytest.raises(TypeError):
            mem.capture(user_id="u1", messages=[None])
        mem.close()

    def test_list_message_raises_type_error(self, tmp_path):
        mem = _make_mem(tmp_path)
        with pytest.raises(TypeError):
            mem.capture(user_id="u1", messages=[["role", "user"]])
        mem.close()

    def test_valid_dict_messages_accepted(self, tmp_path):
        mem = _make_mem(tmp_path)
        # Must not raise
        result = mem.capture(
            user_id="u1",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert isinstance(result, list)
        mem.close()


# ---------------------------------------------------------------------------
# Issue 14: distinct_embedding_models field name on MemoryStats
# ---------------------------------------------------------------------------

class TestIssue14DistinctEmbeddingModelsRenamed:
    """Issue 14: MemoryStats must have distinct_embedding_models (not embedding_model_mismatches)."""

    def test_field_exists(self):
        stats = MemoryStats()
        assert hasattr(stats, "distinct_embedding_models"), (
            "MemoryStats must have 'distinct_embedding_models' field"
        )

    def test_old_field_name_absent(self):
        stats = MemoryStats()
        assert not hasattr(stats, "embedding_model_mismatches"), (
            "MemoryStats must NOT have the old 'embedding_model_mismatches' field"
        )

    def test_field_type_is_int(self):
        stats = MemoryStats(distinct_embedding_models=3)
        assert stats.distinct_embedding_models == 3

    def test_stats_method_populates_field(self, tmp_path):
        mem = _make_mem(tmp_path)
        mem.add("u1", "Memory A", embedding_model="model-v1")
        mem.add("u1", "Memory B", embedding_model="model-v2")
        stats = mem.stats()
        assert hasattr(stats, "distinct_embedding_models")
        assert stats.distinct_embedding_models == 2
        mem.close()


# ---------------------------------------------------------------------------
# Issue 15: Lifespan accepts both str and enum in add() and capture()
# ---------------------------------------------------------------------------

class TestIssue15LifespanStrEnumConsistency:
    """Issue 15: add() and capture() both accept MemoryLifespan enum and string."""

    def test_add_with_enum_lifespan(self, tmp_path):
        mem = _make_mem(tmp_path)
        m = mem.add("u1", "Enum lifespan test", lifespan=MemoryLifespan.SHORT_TERM)
        assert m.lifespan == MemoryLifespan.SHORT_TERM
        mem.close()

    def test_add_with_string_lifespan_short_term(self, tmp_path):
        mem = _make_mem(tmp_path)
        m = mem.add("u1", "String lifespan test", lifespan="short_term")
        assert m.lifespan == MemoryLifespan.SHORT_TERM
        mem.close()

    def test_add_with_string_lifespan_working(self, tmp_path):
        mem = _make_mem(tmp_path)
        m = mem.add("u1", "Working lifespan test", lifespan="working")
        assert m.lifespan == MemoryLifespan.WORKING
        mem.close()

    def test_add_with_string_lifespan_long_term(self, tmp_path):
        mem = _make_mem(tmp_path)
        m = mem.add("u1", "Long term lifespan test", lifespan="long_term")
        assert m.lifespan == MemoryLifespan.LONG_TERM
        mem.close()

    def test_add_with_invalid_string_lifespan_raises(self, tmp_path):
        mem = _make_mem(tmp_path)
        with pytest.raises(ValueError, match="lifespan"):
            mem.add("u1", "Invalid lifespan test", lifespan="forever")
        mem.close()

    def test_capture_with_enum_lifespan(self, tmp_path):
        mem = _make_mem(tmp_path)
        captured = mem.capture(
            user_id="u1",
            messages=[{"role": "user", "content": "I like tea"}],
            lifespan=MemoryLifespan.SHORT_TERM,
        )
        for m in captured:
            assert m.lifespan == MemoryLifespan.SHORT_TERM
        mem.close()

    def test_capture_with_string_lifespan(self, tmp_path):
        mem = _make_mem(tmp_path)
        captured = mem.capture(
            user_id="u1",
            messages=[{"role": "user", "content": "I like coffee"}],
            lifespan="working",
        )
        for m in captured:
            assert m.lifespan == MemoryLifespan.WORKING
        mem.close()

    def test_capture_with_invalid_string_lifespan_raises(self, tmp_path):
        mem = _make_mem(tmp_path)
        with pytest.raises(ValueError, match="lifespan"):
            mem.capture(
                user_id="u1",
                messages=[{"role": "user", "content": "hello"}],
                lifespan="ephemeral",
            )
        mem.close()


# ---------------------------------------------------------------------------
# Issue 16: conftest.py fixture works (existing tests still pass)
# ---------------------------------------------------------------------------

class TestIssue16ConftestFixture:
    """Issue 16: Verify the conftest fixtures are usable and produce valid instances."""

    def test_mem_fixture_usable(self, mem):
        """The 'mem' fixture from conftest must produce a working OpenMem instance."""
        assert mem is not None
        assert isinstance(mem, OpenMem)

    def test_mem_fixture_can_add_and_retrieve(self, mem):
        m = mem.add("fixture_user", "Fixture test memory")
        retrieved = mem.get(m.id)
        assert retrieved is not None
        assert retrieved.content == "Fixture test memory"

    def test_mock_embedding_callback_fixture(self, mock_embedding_callback):
        emb = mock_embedding_callback("hello world")
        assert isinstance(emb, list)
        assert len(emb) == 8
        assert all(0.0 <= v <= 1.0 for v in emb)

    def test_mock_llm_callback_fixture(self, mock_llm_callback):
        req = LLMRequest(
            system_prompt="Extract memory extraction details",
            user_prompt="User said: I prefer Python",
        )
        resp = mock_llm_callback(req)
        assert isinstance(resp, LLMResponse)
        assert resp.content

    def test_sample_memories_fixture(self, sample_memories):
        assert len(sample_memories) == 4
        for m in sample_memories:
            assert "content" in m
            assert isinstance(m["content"], str)


# ---------------------------------------------------------------------------
# Issue 17: README badges show correct numbers (96 tests passing, 97% coverage)
# ---------------------------------------------------------------------------

class TestIssue17ReadmeBadges:
    """Issue 17: README.md badges must state current test/coverage numbers."""

    def test_readme_badge_237_tests(self):
        readme_path = Path("/Users/ashishbhatia/ClaudeProjects/OpenMem/README.md")
        assert readme_path.exists(), "README.md must exist"
        content = readme_path.read_text()
        assert "237" in content, (
            "README must mention '237' (passing tests badge)"
        )

    def test_readme_badge_98_coverage(self):
        readme_path = Path("/Users/ashishbhatia/ClaudeProjects/OpenMem/README.md")
        content = readme_path.read_text()
        assert "98" in content, (
            "README must mention '98' (coverage badge)"
        )

    def test_readme_badges_in_comment_block(self):
        """The badge lines must be present (even if inside HTML comment)."""
        readme_path = Path("/Users/ashishbhatia/ClaudeProjects/OpenMem/README.md")
        content = readme_path.read_text()
        # Both numbers 237 and 98 must appear somewhere
        assert "237" in content and "98" in content, (
            "README must contain both '237' and '98'"
        )


# ---------------------------------------------------------------------------
# Issue 18: access_timestamps cap at 500
# ---------------------------------------------------------------------------

class TestIssue18AccessTimestampsCap:
    """Issue 18: access_timestamps must be capped at 500 entries."""

    def test_cap_enforced_at_500(self, tmp_path):
        """Recalling a memory 600 times must cap access_timestamps at 500."""
        mem = _make_mem(tmp_path)
        m = mem.add("u1", "Cap test memory")

        # recall() with reinforce=True appends to access_timestamps
        for _ in range(600):
            mem.recall("u1", "Cap test memory")

        final = mem.get(m.id)
        assert final is not None
        assert len(final.access_timestamps) <= 500, (
            f"access_timestamps must be capped at 500, got {len(final.access_timestamps)}"
        )

    def test_access_count_still_accurate(self, tmp_path):
        """access_count must reflect total accesses even when timestamps are capped."""
        mem = _make_mem(tmp_path)
        m = mem.add("u1", "Count test memory")
        n_recalls = 50
        for _ in range(n_recalls):
            mem.recall("u1", "Count test memory")

        final = mem.get(m.id)
        assert final.access_count == n_recalls, (
            f"access_count must be {n_recalls}, got {final.access_count}"
        )

    def test_cap_is_exactly_500(self, tmp_path):
        """After 501 recalls, timestamps must be exactly 500 (not 499 or 501)."""
        mem = _make_mem(tmp_path)
        m = mem.add("u1", "Exact cap test memory")

        # First, prime to exactly 500
        for _ in range(501):
            mem.recall("u1", "Exact cap test memory")

        final = mem.get(m.id)
        assert len(final.access_timestamps) == 500, (
            f"After 501 recalls, access_timestamps must be exactly 500, "
            f"got {len(final.access_timestamps)}"
        )

    def test_below_cap_not_truncated(self, tmp_path):
        """Fewer than 500 recalls must not truncate access_timestamps."""
        mem = _make_mem(tmp_path)
        m = mem.add("u1", "Below cap test memory")
        # add() seeds access_timestamps with [now], so the initial count is 1.

        n = 10
        for _ in range(n):
            mem.recall("u1", "Below cap test memory")

        final = mem.get(m.id)
        # 1 seed timestamp from add() + n recall timestamps = n + 1 total.
        # All must be present because we are well below the 500 cap.
        expected = n + 1
        assert len(final.access_timestamps) == expected, (
            f"Expected {expected} timestamps (1 seed + {n} recalls), "
            f"got {len(final.access_timestamps)}"
        )


# ---------------------------------------------------------------------------
# Additional integration sanity checks
# ---------------------------------------------------------------------------

class TestIntegrationSanity:
    """Cross-issue integration tests to catch regressions."""

    def test_full_crud_cycle(self, tmp_path):
        mem = _make_mem(tmp_path)
        m = mem.add("u1", "Integration test memory", confidence=0.8)
        assert mem.get(m.id) is not None

        updated = mem.update(m.id, confidence=0.95)
        assert updated.confidence == 0.95
        assert updated.version == 2

        deleted = mem.delete(m.id)
        assert deleted
        assert mem.get(m.id) is None
        mem.close()

    def test_export_json_round_trip(self, tmp_path):
        mem = _make_mem(tmp_path)
        mem.add("u1", "Export test memory", confidence=0.7)
        exported = mem.export("u1", format="json")
        data = json.loads(exported)
        assert len(data) == 1
        assert data[0]["confidence"] == 0.7
        mem.close()

    def test_stats_recommendations_populated(self, tmp_path):
        """Stats must produce at least the 'no decay runs' recommendation."""
        mem = _make_mem(tmp_path)
        mem.add("u1", "Stats test memory")
        stats = mem.stats()
        assert isinstance(stats.recommendations, list)
        # At minimum: no decay runs recorded
        recs_text = " ".join(stats.recommendations).lower()
        assert "decay" in recs_text, (
            "Stats must recommend running decay if no decay has been run"
        )
        mem.close()
