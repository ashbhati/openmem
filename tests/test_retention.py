"""Tests for the retention engine (decay, consolidation, conflict detection)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from openmem import (
    ConflictPair,
    ConflictStrategy,
    ConsolidationProposal,
    DecayResult,
    LLMRequest,
    LLMResponse,
    Memory,
    MemoryLifespan,
    MemoryType,
    OpenMem,
    OpenMemConfig,
)


USER = "test_user"


# ---------------------------------------------------------------------------
# Decay tests
# ---------------------------------------------------------------------------


class TestDecay:
    def test_decay_reduces_strength_of_unused_memories(self, mem: OpenMem):
        """Memories with old access timestamps should have reduced strength after decay."""
        m = mem.add(USER, "Old memory")
        fetched = mem.get(m.id)

        # Backdate the access timestamps to simulate an old memory
        old_time = datetime.now(timezone.utc) - timedelta(days=30)
        fetched.access_timestamps = [old_time]
        fetched.last_accessed = old_time
        fetched.created_at = old_time
        mem._store.update(fetched)

        result = mem.decay()
        assert isinstance(result, DecayResult)
        assert result.evaluated >= 1

        updated = mem.get(m.id)
        assert updated is not None
        assert updated.strength < 1.0

    def test_decay_preserves_frequently_accessed_memories(self, mem: OpenMem):
        """Memories accessed many times recently should retain high strength."""
        m = mem.add(USER, "Frequently accessed memory")
        fetched = mem.get(m.id)

        # Add many recent access timestamps
        now = datetime.now(timezone.utc)
        fetched.access_timestamps = [now - timedelta(seconds=i) for i in range(20)]
        fetched.access_count = 20
        mem._store.update(fetched)

        result = mem.decay()
        updated = mem.get(m.id)
        assert updated is not None
        # Frequently accessed memory should still be active
        assert updated.is_active is True
        assert updated.strength > 0.5

    def test_decay_soft_deletes_below_threshold(self, mem: OpenMem):
        """Memories with very old, single access should be soft-deleted."""
        m = mem.add(USER, "Very old memory")
        fetched = mem.get(m.id)

        # Make access timestamp very old (200 days ago)
        old_time = datetime.now(timezone.utc) - timedelta(days=200)
        fetched.access_timestamps = [old_time]
        fetched.last_accessed = old_time
        fetched.created_at = old_time
        mem._store.update(fetched)

        result = mem.decay()
        assert result.soft_deleted >= 1

        updated = mem.get(m.id)
        assert updated is not None
        assert updated.is_active is False

    def test_decay_respects_ttl(self, mem: OpenMem):
        """Memories past their TTL should be soft-deleted regardless of strength."""
        past_ttl = datetime.now(timezone.utc) - timedelta(hours=1)
        m = mem.add(USER, "Expiring memory", ttl=past_ttl)

        result = mem.decay()
        assert result.soft_deleted >= 1

        updated = mem.get(m.id)
        assert updated is not None
        assert updated.is_active is False
        assert updated.strength == 0.0


class TestPurge:
    def test_purge_removes_soft_deleted(self, mem: OpenMem):
        m = mem.add(USER, "Soon to be purged")
        # Soft-delete via store
        mem._store.soft_delete(m.id)

        count = mem.purge()
        assert count >= 1
        assert mem.get(m.id) is None


# ---------------------------------------------------------------------------
# Consolidation tests
# ---------------------------------------------------------------------------


class TestConsolidate:
    def _create_similar_memories(self, mem: OpenMem) -> list[str]:
        """Create memories with identical embeddings to form a cluster."""
        ids = []
        # Use very similar content so embeddings are close
        ids.append(mem.add(USER, "User likes Python programming").id)
        ids.append(mem.add(USER, "User likes Python coding").id)
        return ids

    def test_consolidate_propose_finds_similar_clusters(self, tmp_path: Path):
        """propose should find clusters of similar memories."""
        def embed_fn(text: str) -> list[float]:
            # Return identical embedding for all text to force clustering
            return [0.5] * 8

        def llm_fn(req: LLMRequest) -> LLMResponse:
            sys = req.system_prompt.lower()
            if "extract" in sys:
                return LLMResponse(content="[]")
            return LLMResponse(content=json.dumps({
                "content": "User is a Python enthusiast",
                "memory_type": "preference",
                "confidence": 0.95,
                "reasoning": "Merged Python-related preferences",
            }))

        config = OpenMemConfig(storage_path=str(tmp_path / "consol.db"))
        m = OpenMem(llm_callback=llm_fn, embedding_callback=embed_fn, config=config)

        m.add(USER, "User likes Python")
        m.add(USER, "User prefers Python")

        proposals = m.consolidate_propose(USER)
        assert len(proposals) >= 1
        assert proposals[0].proposed_content
        assert len(proposals[0].source_memory_ids) >= 2
        m.close()

    def test_consolidate_apply_creates_merged_memory(self, tmp_path: Path):
        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        def llm_fn(req: LLMRequest) -> LLMResponse:
            return LLMResponse(content=json.dumps({
                "content": "User is a Python enthusiast",
                "memory_type": "preference",
                "confidence": 0.95,
                "reasoning": "Merged Python-related preferences",
            }))

        config = OpenMemConfig(storage_path=str(tmp_path / "consol2.db"))
        m = OpenMem(llm_callback=llm_fn, embedding_callback=embed_fn, config=config)

        m1 = m.add(USER, "User likes Python")
        m2 = m.add(USER, "User prefers Python")

        proposals = m.consolidate_propose(USER)
        if proposals:
            new_memories = m.consolidate_apply(proposals)
            assert len(new_memories) >= 1
            assert new_memories[0].content == "User is a Python enthusiast"
        m.close()

    def test_consolidate_marks_sources_as_superseded(self, tmp_path: Path):
        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        def llm_fn(req: LLMRequest) -> LLMResponse:
            return LLMResponse(content=json.dumps({
                "content": "User is a Python enthusiast",
                "memory_type": "preference",
                "confidence": 0.95,
                "reasoning": "Merged",
            }))

        config = OpenMemConfig(storage_path=str(tmp_path / "consol3.db"))
        m = OpenMem(llm_callback=llm_fn, embedding_callback=embed_fn, config=config)

        m1 = m.add(USER, "User likes Python")
        m2 = m.add(USER, "User prefers Python")

        proposals = m.consolidate_propose(USER)
        if proposals:
            new_memories = m.consolidate_apply(proposals)
            assert len(new_memories) >= 1

            # Source memories should be marked as superseded
            src1 = m.get(m1.id)
            src2 = m.get(m2.id)
            assert src1.is_active is False
            assert src2.is_active is False
            assert src1.superseded_by == new_memories[0].id
            assert src2.superseded_by == new_memories[0].id
        m.close()

    def test_consolidate_idempotent(self, tmp_path: Path):
        """Superseded memories should not be re-proposed."""
        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        def llm_fn(req: LLMRequest) -> LLMResponse:
            return LLMResponse(content=json.dumps({
                "content": "User is a Python enthusiast",
                "memory_type": "preference",
                "confidence": 0.95,
                "reasoning": "Merged",
            }))

        config = OpenMemConfig(storage_path=str(tmp_path / "consol4.db"))
        m = OpenMem(llm_callback=llm_fn, embedding_callback=embed_fn, config=config)

        m.add(USER, "User likes Python")
        m.add(USER, "User prefers Python")

        # First consolidation
        m.consolidate(USER)

        # Second consolidation should return nothing (sources superseded)
        proposals = m.consolidate_propose(USER)
        # The consolidated memory is alone now, so no new clusters
        assert len(proposals) == 0
        m.close()


# ---------------------------------------------------------------------------
# Conflict tests
# ---------------------------------------------------------------------------


class TestConflict:
    def test_find_conflicts_detects_contradictions(self, tmp_path: Path):
        """Conflicting memories should be detected."""
        def embed_fn(text: str) -> list[float]:
            # Very similar embeddings to exceed conflict threshold
            return [0.5] * 8

        def llm_fn(req: LLMRequest) -> LLMResponse:
            sys = req.system_prompt.lower()
            if "conflict" in sys or "contradic" in sys:
                return LLMResponse(content=json.dumps({
                    "is_conflict": True,
                    "explanation": "These memories contradict each other",
                }))
            return LLMResponse(content="[]")

        config = OpenMemConfig(
            storage_path=str(tmp_path / "conflict.db"),
            conflict_similarity_threshold=0.3,  # Low threshold to trigger conflict check
        )
        m = OpenMem(llm_callback=llm_fn, embedding_callback=embed_fn, config=config)

        m.add(USER, "User prefers Python")
        m.add(USER, "User dislikes Python")

        conflicts = m.find_conflicts(USER)
        assert len(conflicts) >= 1
        assert isinstance(conflicts[0], ConflictPair)
        assert conflicts[0].explanation
        m.close()

    def test_conflict_keep_both_strategy(self, tmp_path: Path):
        """KEEP_BOTH strategy should leave both memories active."""
        from openmem.retention.conflict import resolve_conflict

        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        config = OpenMemConfig(
            storage_path=str(tmp_path / "keep_both.db"),
            conflict_strategy=ConflictStrategy.KEEP_BOTH,
        )
        m = OpenMem(embedding_callback=embed_fn, config=config)

        m1 = m.add(USER, "User prefers Python")
        m2 = m.add(USER, "User dislikes Python")

        resolve_conflict(
            store=m._store,
            vector_cache=m._vector_cache,
            memory_a=m.get(m1.id),
            memory_b=m.get(m2.id),
            strategy=ConflictStrategy.KEEP_BOTH,
        )

        # Both should still be active
        assert m.get(m1.id).is_active is True
        assert m.get(m2.id).is_active is True
        m.close()

    def test_resolve_conflict_supersede_strategy(self, tmp_path: Path):
        """SUPERSEDE strategy should supersede the older memory."""
        from openmem.retention.conflict import resolve_conflict
        from datetime import timedelta

        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "supersede.db"))
        m = OpenMem(embedding_callback=embed_fn, config=config)

        m1 = m.add(USER, "User prefers Python")
        m2 = m.add(USER, "User dislikes Python")

        # Make m1 older
        mem1 = m.get(m1.id)
        mem1.created_at = mem1.created_at - timedelta(days=1)
        m._store.update(mem1)

        resolve_conflict(
            store=m._store,
            vector_cache=m._vector_cache,
            memory_a=m.get(m1.id),
            memory_b=m.get(m2.id),
            strategy=ConflictStrategy.SUPERSEDE,
        )

        # Newer (m2) should win, older (m1) should be superseded
        assert m.get(m1.id).is_active is False
        assert m.get(m1.id).superseded_by == m2.id
        assert m.get(m2.id).is_active is True
        m.close()

    def test_resolve_conflict_keep_newer(self, tmp_path: Path):
        """KEEP_NEWER strategy should supersede the older memory."""
        from openmem.retention.conflict import resolve_conflict
        from datetime import timedelta

        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "keep_newer.db"))
        m = OpenMem(embedding_callback=embed_fn, config=config)

        m1 = m.add(USER, "User prefers Python")
        m2 = m.add(USER, "User dislikes Python")

        # Make m1 older
        mem1 = m.get(m1.id)
        mem1.created_at = mem1.created_at - timedelta(days=1)
        m._store.update(mem1)

        resolve_conflict(
            store=m._store,
            vector_cache=m._vector_cache,
            memory_a=m.get(m1.id),
            memory_b=m.get(m2.id),
            strategy=ConflictStrategy.KEEP_NEWER,
        )

        # Newer (m2) should win, older (m1) should be superseded
        assert m.get(m1.id).is_active is False
        assert m.get(m1.id).superseded_by == m2.id
        assert m.get(m2.id).is_active is True
        m.close()

    def test_resolve_conflict_keep_higher_confidence(self, tmp_path: Path):
        """KEEP_HIGHER_CONFIDENCE should supersede the lower-confidence memory."""
        from openmem.retention.conflict import resolve_conflict

        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "keep_conf.db"))
        m = OpenMem(embedding_callback=embed_fn, config=config)

        m1 = m.add(USER, "User prefers Python", confidence=0.9)
        m2 = m.add(USER, "User dislikes Python", confidence=0.5)

        resolve_conflict(
            store=m._store,
            vector_cache=m._vector_cache,
            memory_a=m.get(m1.id),
            memory_b=m.get(m2.id),
            strategy=ConflictStrategy.KEEP_HIGHER_CONFIDENCE,
        )

        # Higher confidence (m1) should win, lower (m2) should be superseded
        assert m.get(m1.id).is_active is True
        assert m.get(m2.id).is_active is False
        assert m.get(m2.id).superseded_by == m1.id
        m.close()

    def test_decay_no_access_timestamps(self, tmp_path: Path):
        """Memory with empty access_timestamps should still be handled."""
        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "no_access.db"))
        m = OpenMem(embedding_callback=embed_fn, config=config)

        mem = m.add(USER, "Memory with no access history")
        fetched = m.get(mem.id)
        fetched.access_timestamps = []
        m._store.update(fetched)

        result = m.decay()
        assert isinstance(result, DecayResult)
        assert result.evaluated >= 1
        # With empty timestamps, activation is 0, strength via sigmoid is 0.5
        updated = m.get(mem.id)
        assert updated is not None
        m.close()

    def test_retention_engine_no_llm_callback(self, tmp_path: Path):
        """RetentionEngine without llm_callback should return empty for LLM-dependent ops."""
        from openmem.retention.engine import RetentionEngine

        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "no_llm.db"))
        m = OpenMem(embedding_callback=embed_fn, config=config)

        m.add(USER, "Memory one")
        m.add(USER, "Memory two")

        engine = RetentionEngine(
            store=m._store,
            vector_cache=m._vector_cache,
            llm_callback=None,
            embedding_callback=embed_fn,
            config=config,
        )

        # consolidate_propose should return [] without LLM
        proposals = engine.consolidate_propose(USER)
        assert proposals == []

        # find_conflicts should return [] without LLM
        conflicts = engine.find_conflicts(USER)
        assert conflicts == []
        m.close()
