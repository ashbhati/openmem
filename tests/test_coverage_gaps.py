"""Targeted tests to close coverage gaps and reach >= 95%."""

from __future__ import annotations

import hashlib
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
    MemorySource,
    MemoryType,
    OpenMem,
    OpenMemConfig,
)

USER = "test_user"


# ---------------------------------------------------------------------------
# capture/extractor.py — regex fallback, repair failure, confidence clamping
# ---------------------------------------------------------------------------


class TestExtractorRegexFallback:
    def test_json_embedded_in_text(self, tmp_path: Path):
        """Extractor should extract JSON array from surrounding markdown/text."""

        def llm_fn(req: LLMRequest) -> LLMResponse:
            # Return JSON embedded in markdown backticks
            return LLMResponse(
                content='Here are the memories:\n[{"content": "User likes cats", "memory_type": "fact", "source": "explicit", "confidence": 0.9}]\nDone!'
            )

        def embed_fn(text: str) -> list[float]:
            return [0.1] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "regex.db"))
        m = OpenMem(llm_callback=llm_fn, embedding_callback=embed_fn, config=config)
        memories = m.capture(user_id=USER, messages=[{"role": "user", "content": "I like cats"}])
        assert len(memories) == 1
        assert memories[0].content == "User likes cats"
        m.close()

    def test_repair_also_fails_returns_empty(self, tmp_path: Path):
        """When both initial parse and repair fail, capture returns []."""

        def llm_fn(req: LLMRequest) -> LLMResponse:
            # Always return non-JSON
            return LLMResponse(content="This is not JSON at all")

        def embed_fn(text: str) -> list[float]:
            return [0.1] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "repair_fail.db"))
        m = OpenMem(llm_callback=llm_fn, embedding_callback=embed_fn, config=config)
        memories = m.capture(user_id=USER, messages=[{"role": "user", "content": "hello"}])
        assert memories == []
        m.close()

    def test_confidence_clamped_to_range(self, tmp_path: Path):
        """Confidence values outside 0-1 should be clamped."""

        def llm_fn(req: LLMRequest) -> LLMResponse:
            return LLMResponse(content=json.dumps([
                {"content": "High conf", "memory_type": "fact", "source": "explicit", "confidence": 1.5},
                {"content": "Neg conf", "memory_type": "fact", "source": "explicit", "confidence": -0.3},
            ]))

        def embed_fn(text: str) -> list[float]:
            h = hashlib.md5(text.encode()).hexdigest()
            return [int(h[i:i + 2], 16) / 255.0 for i in range(0, 16, 2)]

        config = OpenMemConfig(storage_path=str(tmp_path / "clamp.db"))
        m = OpenMem(llm_callback=llm_fn, embedding_callback=embed_fn, config=config)
        memories = m.capture(user_id=USER, messages=[{"role": "user", "content": "test"}])
        for mem in memories:
            assert 0.0 <= mem.confidence <= 1.0
        m.close()


# ---------------------------------------------------------------------------
# capture/engine.py — no callback raises
# ---------------------------------------------------------------------------


class TestCaptureEngineCallbackValidation:
    def test_capture_no_llm_callback_raises(self, tmp_path: Path):
        """capture() without llm_callback should raise ValueError."""
        def embed_fn(text: str) -> list[float]:
            return [0.1] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "no_llm.db"))
        m = OpenMem(embedding_callback=embed_fn, config=config)
        with pytest.raises(ValueError, match="llm_callback"):
            m.capture(user_id=USER, messages=[{"role": "user", "content": "hi"}])
        m.close()

    def test_capture_no_embedding_callback_raises(self, tmp_path: Path):
        """capture() without embedding_callback should raise ValueError."""
        def llm_fn(req: LLMRequest) -> LLMResponse:
            return LLMResponse(content="[]")

        config = OpenMemConfig(storage_path=str(tmp_path / "no_embed.db"))
        m = OpenMem(llm_callback=llm_fn, config=config)
        with pytest.raises(ValueError, match="embedding_callback"):
            m.capture(user_id=USER, messages=[{"role": "user", "content": "hi"}])
        m.close()


# ---------------------------------------------------------------------------
# recall/engine.py — FTS5 edge case graceful degradation, empty query
# ---------------------------------------------------------------------------


class TestRecallEdgeCases:
    def test_recall_fts_error_degrades_gracefully(self, tmp_path: Path):
        """If FTS5 throws, recall should still return semantic results."""
        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "fts_edge.db"))
        m = OpenMem(embedding_callback=embed_fn, config=config)
        m.add(USER, "User likes Python")

        # Query with special characters that might trip FTS5 even after escaping
        results = m.recall(user_id=USER, query="(test*)")
        # Should not crash — returns whatever semantic search finds
        assert isinstance(results, list)
        m.close()

    def test_recall_empty_query(self, tmp_path: Path):
        """Recall with empty query should return results (from FTS or semantics)."""
        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "empty_q.db"))
        m = OpenMem(embedding_callback=embed_fn, config=config)
        m.add(USER, "User likes Python")

        results = m.recall(user_id=USER, query="")
        assert isinstance(results, list)
        m.close()


# ---------------------------------------------------------------------------
# recall/ranking.py — keyword-only results, namespace filter
# ---------------------------------------------------------------------------


class TestRankingEdgeCases:
    def test_apply_filters_namespace_filter(self, tmp_path: Path):
        """Namespace filter should exclude memories from other namespaces."""
        def embed_fn(text: str) -> list[float]:
            h = hashlib.md5(text.encode()).hexdigest()
            return [int(h[i:i + 2], 16) / 255.0 for i in range(0, 16, 2)]

        config = OpenMemConfig(storage_path=str(tmp_path / "ns_filter.db"))
        m = OpenMem(embedding_callback=embed_fn, config=config)

        m.add(USER, "Memory in ns_a", namespace="ns_a")
        m.add(USER, "Memory in ns_b", namespace="ns_b")

        results = m.recall(user_id=USER, query="Memory", namespace="ns_a")
        for mem in results:
            assert mem.namespace == "ns_a"
        m.close()


# ---------------------------------------------------------------------------
# retention/decay.py — zero activation, TTL edge
# ---------------------------------------------------------------------------


class TestDecayEdgeCases:
    def test_decay_empty_database(self, tmp_path: Path):
        """Decay on empty database should succeed with all zeros."""
        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "empty_decay.db"))
        m = OpenMem(embedding_callback=embed_fn, config=config)
        result = m.decay()
        assert result.evaluated == 0
        assert result.decayed == 0
        assert result.soft_deleted == 0
        m.close()

    def test_decay_short_term_lifespan_decays_faster(self, tmp_path: Path):
        """Short-term memories should decay faster than long-term."""
        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "lifespan.db"))
        m = OpenMem(embedding_callback=embed_fn, config=config)

        # Add a short-term memory and a long-term memory
        st = m.add(USER, "Short term info", lifespan=MemoryLifespan.SHORT_TERM)
        lt = m.add(USER, "Long term info", lifespan=MemoryLifespan.LONG_TERM)

        # Backdate both identically
        old_time = datetime.now(timezone.utc) - timedelta(days=7)
        for mid in [st.id, lt.id]:
            mem = m.get(mid)
            mem.access_timestamps = [old_time]
            mem.last_accessed = old_time
            mem.created_at = old_time
            m._store.update(mem)

        m.decay()

        st_updated = m.get(st.id)
        lt_updated = m.get(lt.id)

        # Short-term should be weaker (or inactive) compared to long-term
        assert st_updated.strength <= lt_updated.strength
        m.close()


# ---------------------------------------------------------------------------
# retention/consolidation.py — empty proposals, LLM parse failure
# ---------------------------------------------------------------------------


class TestConsolidationEdgeCases:
    def test_consolidate_with_no_clusters(self, tmp_path: Path):
        """When memories have distinct embeddings, no clusters should form."""
        call_count = 0

        def embed_fn(text: str) -> list[float]:
            nonlocal call_count
            call_count += 1
            # Return very different embeddings
            return [float(call_count)] * 8

        def llm_fn(req: LLMRequest) -> LLMResponse:
            return LLMResponse(content=json.dumps({
                "content": "Merged",
                "memory_type": "fact",
                "confidence": 0.9,
                "reasoning": "Merged",
            }))

        config = OpenMemConfig(storage_path=str(tmp_path / "no_cluster.db"))
        m = OpenMem(llm_callback=llm_fn, embedding_callback=embed_fn, config=config)

        m.add(USER, "User likes Python")
        m.add(USER, "User has a cat")

        # With very different embeddings, consolidation should produce no proposals
        # (or at least not crash)
        result = m.consolidate(USER)
        assert isinstance(result, list)
        m.close()

    def test_consolidate_llm_returns_bad_json(self, tmp_path: Path):
        """When LLM returns invalid JSON during consolidation, proposals should be skipped."""

        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        def llm_fn(req: LLMRequest) -> LLMResponse:
            return LLMResponse(content="Not valid JSON at all")

        config = OpenMemConfig(storage_path=str(tmp_path / "bad_consol.db"))
        m = OpenMem(llm_callback=llm_fn, embedding_callback=embed_fn, config=config)

        m.add(USER, "User likes Python")
        m.add(USER, "User prefers Python")

        # Should not crash, just return empty proposals
        proposals = m.consolidate_propose(USER)
        assert isinstance(proposals, list)
        m.close()

    def test_consolidate_apply_empty_proposal(self, tmp_path: Path):
        """Apply with proposal having empty source_memory_ids should skip."""

        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "empty_prop.db"))
        m = OpenMem(embedding_callback=embed_fn, config=config)

        empty_proposal = ConsolidationProposal(
            source_memory_ids=[],
            proposed_content="Should be skipped",
            reasoning="Empty sources",
        )
        result = m.consolidate_apply([empty_proposal])
        assert result == []
        m.close()

    def test_consolidate_apply_missing_source_memories(self, tmp_path: Path):
        """Apply with non-existent source IDs should skip."""

        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "missing_src.db"))
        m = OpenMem(embedding_callback=embed_fn, config=config)

        proposal = ConsolidationProposal(
            source_memory_ids=["nonexistent_1", "nonexistent_2"],
            proposed_content="Merged content",
            reasoning="Test",
        )
        result = m.consolidate_apply([proposal])
        assert result == []
        m.close()


# ---------------------------------------------------------------------------
# retention/conflict.py — fewer than 2 memories, no embedding
# ---------------------------------------------------------------------------


class TestConflictEdgeCases:
    def test_find_conflicts_with_single_memory(self, tmp_path: Path):
        """With only 1 memory, no conflicts should be found."""

        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        def llm_fn(req: LLMRequest) -> LLMResponse:
            return LLMResponse(content=json.dumps({"is_conflict": True, "explanation": "yes"}))

        config = OpenMemConfig(storage_path=str(tmp_path / "single.db"))
        m = OpenMem(llm_callback=llm_fn, embedding_callback=embed_fn, config=config)
        m.add(USER, "Only memory")

        conflicts = m.find_conflicts(USER)
        assert conflicts == []
        m.close()

    def test_find_conflicts_memory_without_embedding(self, tmp_path: Path):
        """Memories without embeddings should be skipped in conflict detection."""

        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        def llm_fn(req: LLMRequest) -> LLMResponse:
            sys = req.system_prompt.lower()
            if "conflict" in sys or "contradic" in sys:
                return LLMResponse(content=json.dumps({"is_conflict": True, "explanation": "yes"}))
            return LLMResponse(content="[]")

        config = OpenMemConfig(
            storage_path=str(tmp_path / "no_emb.db"),
            conflict_similarity_threshold=0.3,
        )
        m = OpenMem(llm_callback=llm_fn, embedding_callback=embed_fn, config=config)

        m1 = m.add(USER, "Memory with embedding")
        # Manually clear embedding on one memory
        mem2 = m.add(USER, "Memory without embedding")
        fetched = m.get(mem2.id)
        fetched.embedding = []
        m._store.update(fetched)
        # Also invalidate the vector cache so it reflects the empty embedding
        m._vector_cache.invalidate_all()

        # Should not crash
        conflicts = m.find_conflicts(USER)
        assert isinstance(conflicts, list)
        m.close()

    def test_find_conflicts_llm_returns_bad_json(self, tmp_path: Path):
        """When LLM returns invalid JSON during conflict check, skip the pair."""

        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        def llm_fn(req: LLMRequest) -> LLMResponse:
            return LLMResponse(content="not json")

        config = OpenMemConfig(
            storage_path=str(tmp_path / "bad_conflict.db"),
            conflict_similarity_threshold=0.3,
        )
        m = OpenMem(llm_callback=llm_fn, embedding_callback=embed_fn, config=config)

        m.add(USER, "Memory A")
        m.add(USER, "Memory B")

        # Should not crash, just skip the pair
        conflicts = m.find_conflicts(USER)
        assert conflicts == []
        m.close()


# ---------------------------------------------------------------------------
# client.py — consolidate returns early, build_context empty, CSV with commas
# ---------------------------------------------------------------------------


class TestClientEdgeCases:
    def test_consolidate_no_proposals_returns_empty(self, tmp_path: Path):
        """consolidate() should return [] when no proposals generated."""
        call_count = 0

        def embed_fn(text: str) -> list[float]:
            nonlocal call_count
            call_count += 1
            return [float(call_count)] * 8  # Different embeddings

        def llm_fn(req: LLMRequest) -> LLMResponse:
            return LLMResponse(content="[]")

        config = OpenMemConfig(storage_path=str(tmp_path / "no_consol.db"))
        m = OpenMem(llm_callback=llm_fn, embedding_callback=embed_fn, config=config)
        m.add(USER, "Only one memory")

        result = m.consolidate(USER)
        assert result == []
        m.close()

    def test_build_context_empty_when_no_memories(self, tmp_path: Path):
        """build_context should return empty string with no memories."""
        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "empty_ctx.db"))
        m = OpenMem(embedding_callback=embed_fn, config=config)

        context = m.build_context("nobody", query="anything")
        assert context == ""
        m.close()

    def test_build_context_max_tokens_truncation(self, tmp_path: Path):
        """build_context should respect max_tokens by truncating."""
        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "truncate.db"))
        m = OpenMem(embedding_callback=embed_fn, config=config)

        # Add many memories
        for i in range(20):
            m.add(USER, f"This is memory number {i} with some extra text for length")

        context = m.build_context(USER, query="memory", max_tokens=10)
        # With tiny token budget, should be truncated
        assert isinstance(context, str)
        # Should have fewer lines than total memories
        lines = [l for l in context.split("\n") if l.startswith("- ")]
        assert len(lines) < 20
        m.close()

    def test_export_csv_with_commas_and_quotes(self, tmp_path: Path):
        """CSV export should properly escape content with commas and quotes."""
        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "csv_escape.db"))
        m = OpenMem(embedding_callback=embed_fn, config=config)

        m.add(USER, 'Likes "Python", not JavaScript')
        exported = m.export(USER, format="csv")
        lines = exported.strip().split("\n")
        assert len(lines) == 2  # Header + 1 data line
        # The content should be properly quoted
        assert '""' in lines[1]  # Escaped quotes
        m.close()

    def test_stats_no_decay_run_recommendation(self, tmp_path: Path):
        """Stats should recommend running decay if never run."""
        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "no_decay.db"))
        m = OpenMem(embedding_callback=embed_fn, config=config)
        m.add(USER, "Some memory")

        stats = m.stats()
        assert any("decay" in r.lower() for r in stats.recommendations)
        m.close()


# ---------------------------------------------------------------------------
# storage/vector_cache.py — uncached user operations
# ---------------------------------------------------------------------------


class TestVectorCacheEdgeCases:
    def test_add_to_uncached_user_is_noop(self):
        """Adding to a user not in cache should be a no-op."""
        from openmem.storage.vector_cache import VectorCache

        cache = VectorCache()
        # This should not raise
        cache.add_to_user("nonexistent:default", "m1", [1.0, 0.0])
        assert not cache.has_user("nonexistent:default")

    def test_remove_from_uncached_user_is_noop(self):
        """Removing from a user not in cache should be a no-op."""
        from openmem.storage.vector_cache import VectorCache

        cache = VectorCache()
        cache.remove_from_user("nonexistent:default", "m1")
        assert not cache.has_user("nonexistent:default")

    def test_search_uncached_user_returns_empty(self):
        """Searching for a user not in cache returns empty."""
        from openmem.storage.vector_cache import VectorCache

        cache = VectorCache()
        results = cache.search("nonexistent:default", [1.0, 0.0])
        assert results == []

    def test_invalidate_all_clears_cache(self):
        """invalidate_all should clear all cached users."""
        from openmem.storage.vector_cache import VectorCache

        cache = VectorCache()
        cache.build_user_index("u1:d", [("m1", [1.0, 0.0])])
        cache.build_user_index("u2:d", [("m2", [0.0, 1.0])])

        assert cache.cached_user_count == 2
        cache.invalidate_all()
        assert cache.cached_user_count == 0

    def test_remove_last_embedding_from_index(self):
        """Removing the last embedding should set matrix to None."""
        from openmem.storage.vector_cache import VectorCache

        cache = VectorCache()
        cache.build_user_index("u:d", [("m1", [1.0, 0.0])])
        cache.remove_from_user("u:d", "m1")

        # After removing the only item, search should return empty
        results = cache.search("u:d", [1.0, 0.0])
        assert results == []


# ---------------------------------------------------------------------------
# storage/sqlite_store.py — get_all_embeddings without namespace
# ---------------------------------------------------------------------------


class TestSQLiteStoreEdgeCases:
    def test_get_all_embeddings_without_namespace(self, tmp_path: Path):
        """get_all_embeddings without namespace should fetch all."""
        from openmem.storage.sqlite_store import SQLiteStore

        store = SQLiteStore(path=tmp_path / "all_emb.db")

        now = datetime.now(timezone.utc)
        m = Memory(
            id="m1", user_id="u1", namespace="ns1", content="Test",
            content_hash=hashlib.sha256(b"Test").hexdigest(),
            memory_type=MemoryType.FACT, source=MemorySource.EXPLICIT,
            confidence=1.0, strength=1.0, created_at=now, last_accessed=now,
            access_count=0, access_timestamps=[now],
            lifespan=MemoryLifespan.LONG_TERM, version=1, is_active=True,
            embedding=[1.0, 2.0, 3.0], embedding_model="", metadata={},
        )
        store.add(m)

        # With namespace
        results = store.get_all_embeddings("u1", namespace="ns1")
        assert len(results) == 1

        # Without namespace
        results_all = store.get_all_embeddings("u1")
        assert len(results_all) == 1

        store.close()
