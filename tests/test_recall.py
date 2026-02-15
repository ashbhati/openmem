"""Tests for the recall engine."""

from __future__ import annotations

import pytest

from openmem import (
    MemoryType,
    OpenMem,
)


USER = "test_user"


def _add_test_memories(mem: OpenMem) -> list[str]:
    """Add several test memories and return their IDs."""
    ids = []
    ids.append(mem.add(USER, "User prefers Python over JavaScript",
                       memory_type=MemoryType.PREFERENCE, confidence=0.95).id)
    ids.append(mem.add(USER, "User works at Acme Corp",
                       memory_type=MemoryType.FACT, confidence=1.0).id)
    ids.append(mem.add(USER, "User has two children",
                       memory_type=MemoryType.BIOGRAPHICAL, confidence=1.0).id)
    ids.append(mem.add(USER, "User tends to work late at night",
                       memory_type=MemoryType.INSIGHT, confidence=0.7).id)
    return ids


class TestRecallBasic:
    def test_recall_returns_relevant_memories(self, mem: OpenMem):
        _add_test_memories(mem)
        results = mem.recall(user_id=USER, query="Python programming")
        assert len(results) > 0

    def test_recall_reinforces_memories(self, mem: OpenMem):
        """recall() should increment access_count and append access_timestamps."""
        m = mem.add(USER, "User likes coffee")
        assert m.access_count == 0
        initial_ts_count = len(m.access_timestamps)

        # Recall should reinforce
        results = mem.recall(user_id=USER, query="coffee")
        assert len(results) > 0

        # Re-fetch from store to verify persistence
        updated = mem.get(results[0].id)
        assert updated is not None
        assert updated.access_count >= 1
        assert len(updated.access_timestamps) > initial_ts_count

    def test_search_does_not_reinforce(self, mem: OpenMem):
        """search() is read-only and should NOT bump access_count."""
        m = mem.add(USER, "User likes tea")
        original_count = m.access_count

        results = mem.search(user_id=USER, query="tea")
        assert len(results) > 0

        # Re-fetch to verify no reinforcement
        updated = mem.get(results[0].id)
        assert updated is not None
        assert updated.access_count == original_count


class TestRecallFilters:
    def test_recall_filters_by_confidence(self, mem: OpenMem):
        mem.add(USER, "High confidence fact", confidence=0.9)
        mem.add(USER, "Low confidence guess", confidence=0.3)

        results = mem.recall(user_id=USER, query="fact guess", min_confidence=0.5)
        for m in results:
            assert m.confidence >= 0.5

    def test_recall_filters_by_strength(self, mem: OpenMem):
        m1 = mem.add(USER, "Strong memory")
        m2 = mem.add(USER, "Weak memory")
        # Manually weaken one through the store
        m2_fetched = mem.get(m2.id)
        m2_fetched.strength = 0.05
        mem._store.update(m2_fetched)

        results = mem.recall(user_id=USER, query="memory", min_strength=0.1)
        ids = {m.id for m in results}
        assert m1.id in ids
        assert m2.id not in ids

    def test_recall_filters_by_type(self, mem: OpenMem):
        mem.add(USER, "User prefers dark mode", memory_type=MemoryType.PREFERENCE)
        mem.add(USER, "User was born in 1990", memory_type=MemoryType.BIOGRAPHICAL)

        results = mem.recall(
            user_id=USER,
            query="user info",
            memory_types=["preference"],
        )
        for m in results:
            assert m.memory_type == MemoryType.PREFERENCE

    def test_recall_filters_by_namespace(self, mem: OpenMem):
        mem.add(USER, "Memory in ns_a", namespace="ns_a")
        mem.add(USER, "Memory in ns_b", namespace="ns_b")

        results = mem.recall(user_id=USER, query="Memory", namespace="ns_a")
        for m in results:
            assert m.namespace == "ns_a"


class TestHybridRanking:
    def test_hybrid_ranking_combines_semantic_and_keyword(self, mem: OpenMem):
        """Both semantic and keyword results should contribute to final ranking."""
        mem.add(USER, "User enjoys hiking in the mountains")
        mem.add(USER, "User likes swimming at the beach")
        mem.add(USER, "User prefers outdoor activities")

        results = mem.recall(user_id=USER, query="hiking mountains outdoor")
        assert len(results) > 0
        # The query contains keywords from the first and third memories,
        # so those should appear in results
        contents = [m.content for m in results]
        assert any("hiking" in c for c in contents) or any("outdoor" in c for c in contents)
