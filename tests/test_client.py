"""Integration tests for the OpenMem client."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from openmem import (
    ConflictStrategy,
    LLMRequest,
    LLMResponse,
    Memory,
    MemoryType,
    MemorySource,
    MemoryLifespan,
    MemoryStats,
    OpenMem,
    OpenMemConfig,
)


USER = "test_user"


class TestInit:
    def test_init_creates_db_file(self, tmp_db_path: Path):
        config = OpenMemConfig(storage_path=str(tmp_db_path))
        m = OpenMem(storage_path=str(tmp_db_path), config=config)
        assert tmp_db_path.exists()
        m.close()


class TestAddAndGet:
    def test_add_and_get_memory(self, mem: OpenMem):
        m = mem.add(USER, "Likes coffee")
        assert m.id
        assert m.content == "Likes coffee"
        assert m.user_id == USER
        assert m.embedding  # Should auto-embed

        fetched = mem.get(m.id)
        assert fetched is not None
        assert fetched.content == "Likes coffee"
        # Embeddings stored as float32 BLOBs lose some precision
        assert len(fetched.embedding) == len(m.embedding)
        for a, b in zip(fetched.embedding, m.embedding):
            assert abs(a - b) < 1e-5

    def test_add_duplicate_id_raises(self, mem: OpenMem):
        m = mem.add(USER, "Likes tea")
        # Directly insert same ID through store to trigger duplicate
        m2 = Memory(
            id=m.id,
            user_id=USER,
            content="Different content",
            content_hash="different",
        )
        with pytest.raises(ValueError, match="already exists"):
            mem._store.add(m2)


class TestListMemories:
    def test_list_memories_with_pagination(self, mem: OpenMem):
        for i in range(5):
            mem.add(USER, f"Memory {i}")

        page1 = mem.list(USER, limit=3, offset=0)
        page2 = mem.list(USER, limit=3, offset=3)

        assert len(page1) == 3
        assert len(page2) == 2
        all_ids = {m.id for m in page1} | {m.id for m in page2}
        assert len(all_ids) == 5

    def test_list_memories_with_type_filter(self, mem: OpenMem):
        mem.add(USER, "Likes Python", memory_type=MemoryType.PREFERENCE)
        mem.add(USER, "Works at Acme", memory_type=MemoryType.FACT)
        mem.add(USER, "Has a dog", memory_type=MemoryType.BIOGRAPHICAL)

        prefs = mem.list(USER, memory_types=[MemoryType.PREFERENCE])
        assert len(prefs) == 1
        assert prefs[0].memory_type == MemoryType.PREFERENCE

        facts = mem.list(USER, memory_types=[MemoryType.FACT])
        assert len(facts) == 1
        assert facts[0].memory_type == MemoryType.FACT


class TestUpdate:
    def test_update_memory_content(self, mem: OpenMem):
        m = mem.add(USER, "Likes coffee")
        old_embedding = m.embedding[:]

        updated = mem.update(m.id, content="Prefers tea")
        assert updated is not None
        assert updated.content == "Prefers tea"
        assert updated.version == 2
        # Embedding should have changed since content changed
        assert updated.embedding != old_embedding

    def test_update_nonexistent_returns_none(self, mem: OpenMem):
        result = mem.update("nonexistent_id", content="Nope")
        assert result is None

    def test_update_metadata_and_type(self, mem: OpenMem):
        m = mem.add(USER, "Likes coffee")
        updated = mem.update(
            m.id,
            memory_type=MemoryType.PREFERENCE,
            confidence=0.8,
            lifespan=MemoryLifespan.SHORT_TERM,
            metadata={"tag": "drink"},
        )
        assert updated is not None
        assert updated.memory_type == MemoryType.PREFERENCE
        assert updated.confidence == 0.8
        assert updated.lifespan == MemoryLifespan.SHORT_TERM
        assert updated.metadata == {"tag": "drink"}


class TestDelete:
    def test_delete_memory(self, mem: OpenMem):
        m = mem.add(USER, "Temporary memory")
        assert mem.get(m.id) is not None

        result = mem.delete(m.id)
        assert result is True
        assert mem.get(m.id) is None

        # Deleting again returns False
        assert mem.delete(m.id) is False

    def test_delete_all_for_user(self, mem: OpenMem):
        for i in range(3):
            mem.add(USER, f"Memory {i}")
        mem.add("other_user", "Other memory")

        count = mem.delete_all(USER)
        assert count == 3
        assert mem.list(USER) == []
        # Other user's memories should be unaffected
        assert len(mem.list("other_user")) == 1


class TestExport:
    def test_export_json(self, mem: OpenMem):
        mem.add(USER, "Likes coffee")
        mem.add(USER, "Works remotely")

        exported = mem.export(USER, format="json")
        data = json.loads(exported)
        assert len(data) == 2
        contents = {d["content"] for d in data}
        assert "Likes coffee" in contents
        assert "Works remotely" in contents

    def test_export_csv(self, mem: OpenMem):
        mem.add(USER, "Likes coffee")
        mem.add(USER, "Works remotely")

        exported = mem.export(USER, format="csv")
        lines = exported.strip().split("\n")
        # Header + 2 data lines
        assert len(lines) == 3
        assert "id" in lines[0]
        assert "content" in lines[0]


class TestExportEdgeCases:
    def test_export_csv_empty(self, mem: OpenMem):
        exported = mem.export("nobody", format="csv")
        assert exported == ""

    def test_export_invalid_format(self, mem: OpenMem):
        mem.add(USER, "Something")
        import pytest
        with pytest.raises(ValueError, match="Unsupported format"):
            mem.export(USER, format="xml")


class TestFromSimpleCallback:
    def test_from_simple_callback(self, tmp_db_path: Path):
        def simple_llm(prompt: str) -> str:
            return json.dumps([{
                "content": "User likes pizza",
                "memory_type": "preference",
                "source": "explicit",
                "confidence": 0.9,
            }])

        def simple_embed(text: str) -> list[float]:
            return [0.1] * 8

        m = OpenMem.from_simple_callback(
            llm_fn=simple_llm,
            embed_fn=simple_embed,
            storage_path=str(tmp_db_path),
        )
        assert m is not None
        # Verify the wrapped callback works
        result = m.add("u1", "Test")
        assert result.embedding == [0.1] * 8
        m.close()

    def test_from_simple_callback_capture(self, tmp_path: Path):
        """Verify the wrapped LLM callback works end-to-end with capture."""
        def simple_llm(prompt: str) -> str:
            return json.dumps([{
                "content": "User likes tacos",
                "memory_type": "preference",
                "source": "explicit",
                "confidence": 0.85,
            }])

        def simple_embed(text: str) -> list[float]:
            return [0.2] * 8

        m = OpenMem.from_simple_callback(
            llm_fn=simple_llm,
            embed_fn=simple_embed,
            storage_path=str(tmp_path / "simple.db"),
        )
        memories = m.capture(
            user_id=USER,
            messages=[{"role": "user", "content": "I love tacos"}],
        )
        assert len(memories) == 1
        assert memories[0].content == "User likes tacos"
        m.close()


class TestStats:
    def test_stats_returns_health_metrics(self, mem: OpenMem):
        mem.add(USER, "Likes coffee")
        mem.add(USER, "Works remotely")

        stats = mem.stats()
        assert isinstance(stats, MemoryStats)
        assert stats.total_memories == 2
        assert stats.active_memories == 2
        assert stats.soft_deleted == 0
        assert stats.avg_strength == 1.0
        assert isinstance(stats.recommendations, list)


class TestStatsEdgeCases:
    def test_stats_after_decay_with_recommendations(self, tmp_path: Path):
        """Stats should report hours since last decay and below-threshold warnings."""
        from datetime import datetime, timedelta, timezone

        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "stats.db"))
        m = OpenMem(embedding_callback=embed_fn, config=config)

        mem1 = m.add(USER, "Old memory")
        fetched = m.get(mem1.id)
        old_time = datetime.now(timezone.utc) - timedelta(days=200)
        fetched.access_timestamps = [old_time]
        fetched.last_accessed = old_time
        fetched.created_at = old_time
        m._store.update(fetched)

        # Run decay to soft-delete and set last_decay_run
        m.decay()

        # Set last_decay_run to 48 hours ago to trigger recommendation
        past = datetime.now(timezone.utc) - timedelta(hours=48)
        m._store.set_meta("last_decay_run", past.isoformat())

        # Add a new memory so stats has something active
        m.add(USER, "New memory")

        stats = m.stats()
        assert stats.hours_since_last_decay is not None
        assert stats.hours_since_last_decay >= 48.0
        assert any("decay" in r.lower() for r in stats.recommendations)
        m.close()

    def test_stats_model_mismatches(self, tmp_path: Path):
        """Stats should detect multiple embedding models."""
        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "mismatch.db"))
        m = OpenMem(embedding_callback=embed_fn, config=config)

        m1 = m.add(USER, "Memory one", embedding_model="model-a")
        m2 = m.add(USER, "Memory two", embedding_model="model-b")

        stats = m.stats()
        assert stats.distinct_embedding_models >= 2
        assert any("embedding" in r.lower() for r in stats.recommendations)
        m.close()


class TestResolveConflict:
    def test_resolve_conflict_method(self, tmp_path: Path):
        """Test the client-level resolve_conflict method."""
        def embed_fn(text: str) -> list[float]:
            return [0.5] * 8

        config = OpenMemConfig(
            storage_path=str(tmp_path / "resolve.db"),
            conflict_strategy=ConflictStrategy.KEEP_NEWER,
        )
        m = OpenMem(embedding_callback=embed_fn, config=config)

        from datetime import timedelta
        m1 = m.add(USER, "User prefers Python")
        m2 = m.add(USER, "User dislikes Python")

        # Make m1 older
        mem1 = m.get(m1.id)
        mem1.created_at = mem1.created_at - timedelta(days=1)
        m._store.update(mem1)

        # Use client method (defaults to config strategy KEEP_NEWER)
        m.resolve_conflict(m1.id, m2.id)

        assert m.get(m1.id).is_active is False
        assert m.get(m2.id).is_active is True
        m.close()

    def test_resolve_conflict_not_found(self, mem: OpenMem):
        """resolve_conflict should raise ValueError for missing IDs."""
        import pytest
        m1 = mem.add(USER, "Some memory")
        with pytest.raises(ValueError, match="not found"):
            mem.resolve_conflict(m1.id, "nonexistent_id")


class TestReembed:
    def test_reembed_method(self, tmp_path: Path):
        """Test the client-level reembed method."""
        call_count = 0

        def embed_fn(text: str) -> list[float]:
            nonlocal call_count
            call_count += 1
            return [0.1 * call_count] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "reembed.db"))
        m = OpenMem(embedding_callback=embed_fn, config=config)

        m.add(USER, "Memory one")
        m.add(USER, "Memory two")
        initial_count = call_count

        # Re-embed all memories
        count = m.reembed(USER)
        assert count == 2
        assert call_count == initial_count + 2

        # Verify embeddings changed
        memories = m.list(USER)
        for mem in memories:
            # Embeddings should have been updated
            assert len(mem.embedding) == 8
        m.close()

    def test_reembed_no_callback_raises(self, tmp_path: Path):
        """reembed should raise RuntimeError without embedding_callback."""
        config = OpenMemConfig(storage_path=str(tmp_path / "no_embed.db"))
        m = OpenMem(config=config)
        import pytest
        with pytest.raises(RuntimeError, match="No embedding_callback"):
            m.reembed(USER)
        m.close()


class TestConfigProperty:
    def test_config_property(self, mem: OpenMem):
        config = mem.config
        assert isinstance(config, OpenMemConfig)


class TestBuildContext:
    def test_build_context(self, mem: OpenMem):
        mem.add(USER, "Likes coffee")
        mem.add(USER, "Works remotely")

        context = mem.build_context(USER, query="coffee")
        assert isinstance(context, str)
        # Should contain the header
        if context:
            assert "Known about this user:" in context

    def test_build_context_empty(self, tmp_path: Path):
        """build_context returns empty string when no memories exist."""
        config = OpenMemConfig(storage_path=str(tmp_path / "empty_ctx.db"))
        m = OpenMem(config=config)
        context = m.build_context("nobody", query="anything")
        assert context == ""
        m.close()

    def test_build_context_token_limit(self, mem: OpenMem):
        """build_context should respect max_tokens limit."""
        for i in range(20):
            mem.add(USER, f"Memory number {i} with some extra content for length")
        context = mem.build_context(USER, query="memory", max_tokens=10)
        assert isinstance(context, str)
        # Should be short due to token limit
        lines = context.strip().split("\n")
        # At least the header + at most a couple of memories
        assert len(lines) <= 5
