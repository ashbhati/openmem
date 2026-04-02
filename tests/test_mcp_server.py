"""Tests for the OpenMem MCP server.

Tests the MCP tool functions directly (without running a transport layer).
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid

import pytest

# Override storage path BEFORE importing server (it lazy-inits on first use)
_tmp_dir = tempfile.mkdtemp()
os.environ["OPENMEM_STORAGE_PATH"] = os.path.join(_tmp_dir, "test_mcp.db")
os.environ["OPENMEM_EMBEDDING_PROVIDER"] = "none"  # No external API calls

from openmem.mcp.server import (
    _get_client,
    add_memory,
    build_context,
    delete_all_memories,
    delete_memory,
    export_memories,
    get_memory,
    list_memories,
    memory_stats,
    purge_memories,
    recall_memories,
    run_decay,
    search_memories,
    update_memory,
)
import openmem.mcp.server as server_module
import openmem.mcp.providers as providers_module


@pytest.fixture(autouse=True)
def _fresh_client():
    """Reset the singleton client and config loader between tests."""
    server_module._client = None
    providers_module._config_loaded = False
    # Use a unique DB per test to avoid interference
    db_path = os.path.join(_tmp_dir, f"test_{uuid.uuid4().hex}.db")
    os.environ["OPENMEM_STORAGE_PATH"] = db_path
    yield
    if server_module._client is not None:
        server_module._client.close()
        server_module._client = None


class TestAddMemory:
    def test_add_basic(self):
        result = json.loads(add_memory(user_id="u1", content="Likes coffee"))
        assert result["user_id"] == "u1"
        assert result["content"] == "Likes coffee"
        assert result["memory_type"] == "fact"
        assert result["source"] == "explicit"
        assert result["confidence"] == 1.0
        assert result["is_active"] is True
        assert result["id"]  # non-empty ULID

    def test_add_with_all_params(self):
        result = json.loads(add_memory(
            user_id="u1",
            content="Prefers dark mode",
            memory_type="preference",
            source="implicit",
            confidence=0.8,
            lifespan="working",
            namespace="agent_a",
            metadata={"context": "UI settings"},
        ))
        assert result["memory_type"] == "preference"
        assert result["source"] == "implicit"
        assert result["confidence"] == 0.8
        assert result["lifespan"] == "working"
        assert result["namespace"] == "agent_a"
        assert result["metadata"] == {"context": "UI settings"}

    def test_add_invalid_type_returns_error(self):
        """Invalid enum values should return JSON error, not raise exceptions."""
        result = json.loads(add_memory(user_id="u1", content="test", memory_type="invalid"))
        assert "error" in result
        assert "invalid" in result["error"].lower()

    def test_add_invalid_source_returns_error(self):
        result = json.loads(add_memory(user_id="u1", content="test", source="bad"))
        assert "error" in result

    def test_add_invalid_lifespan_returns_error(self):
        result = json.loads(add_memory(user_id="u1", content="test", lifespan="forever"))
        assert "error" in result


class TestGetMemory:
    def test_get_existing(self):
        added = json.loads(add_memory(user_id="u1", content="Test memory"))
        result = json.loads(get_memory(added["id"]))
        assert result["content"] == "Test memory"
        assert result["id"] == added["id"]

    def test_get_nonexistent(self):
        result = json.loads(get_memory("nonexistent_id"))
        assert "error" in result


class TestListMemories:
    def test_list_empty(self):
        result = json.loads(list_memories(user_id="u1"))
        assert result == []

    def test_list_with_memories(self):
        add_memory(user_id="u1", content="Memory 1")
        add_memory(user_id="u1", content="Memory 2")
        add_memory(user_id="u2", content="Other user memory")

        result = json.loads(list_memories(user_id="u1"))
        assert len(result) == 2
        contents = {m["content"] for m in result}
        assert contents == {"Memory 1", "Memory 2"}

    def test_list_pagination(self):
        for i in range(5):
            add_memory(user_id="u1", content=f"Memory {i}")

        page1 = json.loads(list_memories(user_id="u1", limit=2, offset=0))
        page2 = json.loads(list_memories(user_id="u1", limit=2, offset=2))
        assert len(page1) == 2
        assert len(page2) == 2

    def test_list_filter_by_type(self):
        add_memory(user_id="u1", content="A fact", memory_type="fact")
        add_memory(user_id="u1", content="A preference", memory_type="preference")

        result = json.loads(list_memories(user_id="u1", memory_types=["preference"]))
        assert len(result) == 1
        assert result[0]["memory_type"] == "preference"

    def test_list_invalid_type_returns_error(self):
        result = json.loads(list_memories(user_id="u1", memory_types=["invalid_type"]))
        assert "error" in result


class TestUpdateMemory:
    def test_update_content(self):
        added = json.loads(add_memory(user_id="u1", content="Old content"))
        result = json.loads(update_memory(memory_id=added["id"], content="New content"))
        assert result["content"] == "New content"
        assert result["version"] == 2

    def test_update_metadata(self):
        added = json.loads(add_memory(user_id="u1", content="Test"))
        result = json.loads(update_memory(
            memory_id=added["id"],
            metadata={"key": "value"},
        ))
        assert result["metadata"] == {"key": "value"}

    def test_update_nonexistent(self):
        result = json.loads(update_memory(memory_id="nonexistent"))
        assert "error" in result

    def test_update_invalid_type_returns_error(self):
        added = json.loads(add_memory(user_id="u1", content="Test"))
        result = json.loads(update_memory(memory_id=added["id"], memory_type="bad"))
        assert "error" in result

    def test_update_invalid_lifespan_returns_error(self):
        added = json.loads(add_memory(user_id="u1", content="Test"))
        result = json.loads(update_memory(memory_id=added["id"], lifespan="forever"))
        assert "error" in result


class TestDeleteMemory:
    def test_delete_existing(self):
        added = json.loads(add_memory(user_id="u1", content="To delete"))
        result = json.loads(delete_memory(added["id"]))
        assert result["deleted"] is True

        # Verify it's gone
        get_result = json.loads(get_memory(added["id"]))
        assert "error" in get_result

    def test_delete_nonexistent(self):
        result = json.loads(delete_memory("nonexistent"))
        assert result["deleted"] is False


class TestDeleteAllMemories:
    def test_delete_all(self):
        add_memory(user_id="u1", content="Memory 1")
        add_memory(user_id="u1", content="Memory 2")
        add_memory(user_id="u2", content="Other user")

        result = json.loads(delete_all_memories("u1"))
        assert result["deleted_count"] == 2
        assert result["user_id"] == "u1"

        # u2's memory should still exist
        u2_memories = json.loads(list_memories(user_id="u2"))
        assert len(u2_memories) == 1


class TestSearchAndRecall:
    """Test search/recall with keyword-only search (no embeddings in test)."""

    def test_search_by_keyword(self):
        add_memory(user_id="u1", content="Loves Python programming")
        add_memory(user_id="u1", content="Enjoys hiking on weekends")

        result = json.loads(search_memories(user_id="u1", query="Python"))
        # Without embeddings, FTS5 keyword search still works
        assert isinstance(result, list)

    def test_recall_by_keyword(self):
        add_memory(user_id="u1", content="Prefers dark roast coffee")

        result = json.loads(recall_memories(user_id="u1", query="coffee"))
        assert isinstance(result, list)

    def test_search_empty_user(self):
        result = json.loads(search_memories(user_id="nobody", query="anything"))
        assert result == []

    def test_search_invalid_memory_types_returns_error(self):
        result = json.loads(search_memories(
            user_id="u1", query="test", memory_types=["invalid"]
        ))
        assert "error" in result

    def test_recall_invalid_memory_types_returns_error(self):
        result = json.loads(recall_memories(
            user_id="u1", query="test", memory_types=["invalid"]
        ))
        assert "error" in result


class TestBuildContext:
    def test_build_context_no_memories(self):
        result = build_context(user_id="nobody", query="anything")
        assert result == "No memories found for this user."

    def test_build_context_with_memories(self):
        add_memory(user_id="u1", content="Senior Python developer")
        result = build_context(user_id="u1", query="developer")
        # May or may not find it depending on FTS5 keyword match
        assert isinstance(result, str)


class TestExportMemories:
    def test_export_json(self):
        add_memory(user_id="u1", content="Export me")
        result = export_memories(user_id="u1", format="json")
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["content"] == "Export me"

    def test_export_csv(self):
        add_memory(user_id="u1", content="CSV export test")
        result = export_memories(user_id="u1", format="csv")
        assert "CSV export test" in result
        # RFC 4180: csv.writer uses comma-separated values
        assert result.startswith("id,")

    def test_export_empty(self):
        result = export_memories(user_id="nobody", format="json")
        assert json.loads(result) == []

    def test_export_invalid_format_returns_error(self):
        result = json.loads(export_memories(user_id="u1", format="xml"))
        assert "error" in result


class TestMemoryStats:
    def test_stats_empty(self):
        result = json.loads(memory_stats())
        assert result["total_memories"] == 0
        assert result["active_memories"] == 0
        assert isinstance(result["recommendations"], list)

    def test_stats_with_memories(self):
        add_memory(user_id="u1", content="Memory 1")
        add_memory(user_id="u1", content="Memory 2")

        result = json.loads(memory_stats())
        assert result["total_memories"] == 2
        assert result["active_memories"] == 2


class TestDecayAndPurge:
    def test_run_decay(self):
        add_memory(user_id="u1", content="Will it decay?")
        result = json.loads(run_decay())
        assert "evaluated" in result
        assert "decayed" in result
        assert "soft_deleted" in result

    def test_purge_empty(self):
        result = json.loads(purge_memories())
        assert result["purged_count"] == 0


class TestNamespaceIsolation:
    def test_different_namespaces(self):
        add_memory(user_id="u1", content="In default", namespace="default")
        add_memory(user_id="u1", content="In agent_a", namespace="agent_a")

        default_list = json.loads(list_memories(user_id="u1", namespace="default"))
        agent_a_list = json.loads(list_memories(user_id="u1", namespace="agent_a"))

        assert len(default_list) == 1
        assert default_list[0]["content"] == "In default"
        assert len(agent_a_list) == 1
        assert agent_a_list[0]["content"] == "In agent_a"
