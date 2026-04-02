"""OpenMem MCP Server — expose OpenMem as a Model Context Protocol server.

Run with:
    openmem-mcp              # if installed via pip
    python -m openmem.mcp    # direct invocation

Environment variables:
    OPENMEM_STORAGE_PATH          - SQLite database path (default: ~/.openmem/memory.db)
    OPENMEM_EMBEDDING_PROVIDER    - "openai" (default) or "none"
    OPENMEM_EMBEDDING_MODEL       - embedding model (default: "text-embedding-3-small")
    OPENMEM_EMBEDDING_API_KEY     - API key (falls back to OPENAI_API_KEY)
    OPENMEM_EMBEDDING_BASE_URL    - base URL (default: "https://api.openai.com/v1")
    OPENMEM_EMBEDDING_DIMENSIONS  - optional dimension override
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from openmem import OpenMem, OpenMemConfig
from openmem.types import (
    MemoryLifespan,
    MemorySource,
    MemoryType,
)
from .providers import get_embedding_callback, load_config_env

logger = logging.getLogger("openmem.mcp")

# --- Server setup ---

mcp = FastMCP(
    "OpenMem",
    instructions=(
        "OpenMem is an AI memory engine. Use it to store, search, and manage "
        "memories about users across conversations. Memories persist in a local "
        "SQLite database and support semantic search via embeddings."
    ),
)

# Lazily initialized OpenMem instance
_client: Optional[OpenMem] = None


def _get_client() -> OpenMem:
    """Get or create the OpenMem client singleton."""
    global _client
    if _client is None:
        load_config_env()
        storage_path = os.environ.get("OPENMEM_STORAGE_PATH", "")
        config = OpenMemConfig()
        if storage_path:
            config.storage_path = storage_path

        embedding_callback = get_embedding_callback()

        _client = OpenMem(
            embedding_callback=embedding_callback,
            config=config,
        )
        logger.info("OpenMem initialized (db=%s)", config.resolved_storage_path)
    return _client


def _memory_to_result(m) -> dict[str, Any]:
    """Convert a Memory object to a clean dict for tool results."""
    return {
        "id": m.id,
        "user_id": m.user_id,
        "namespace": m.namespace,
        "content": m.content,
        "memory_type": m.memory_type.value,
        "source": m.source.value,
        "confidence": m.confidence,
        "strength": round(m.strength, 4),
        "created_at": m.created_at.isoformat(),
        "last_accessed": m.last_accessed.isoformat(),
        "access_count": m.access_count,
        "lifespan": m.lifespan.value,
        "version": m.version,
        "is_active": m.is_active,
        "metadata": m.metadata,
    }


# --- MCP Tools ---


@mcp.tool()
def add_memory(
    user_id: str,
    content: str,
    memory_type: str = "fact",
    source: str = "explicit",
    confidence: float = 1.0,
    lifespan: str = "long_term",
    namespace: str = "default",
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Store a new memory about a user.

    Use this to persist facts, preferences, insights, or biographical details
    that should be remembered across conversations.

    Args:
        user_id: Who this memory belongs to.
        content: The memory text (e.g., "Prefers Python over JavaScript").
        memory_type: One of: fact, preference, insight, biographical.
        source: "explicit" (user stated) or "implicit" (you inferred).
        confidence: How confident you are, 0.0-1.0 (default: 1.0).
        lifespan: Decay rate — "short_term", "working", or "long_term".
        namespace: Isolation scope (default: "default").
        metadata: Optional key-value pairs for extra context.
    """
    client = _get_client()
    memory = client.add(
        user_id=user_id,
        content=content,
        memory_type=MemoryType(memory_type),
        source=MemorySource(source),
        confidence=confidence,
        lifespan=MemoryLifespan(lifespan),
        namespace=namespace,
        metadata=metadata,
    )
    return json.dumps(_memory_to_result(memory), indent=2)


@mcp.tool()
def search_memories(
    user_id: str,
    query: str,
    top_k: int = 10,
    namespace: str = "default",
    memory_types: Optional[list[str]] = None,
    min_confidence: float = 0.0,
    min_strength: float = 0.0,
) -> str:
    """Search for relevant memories (read-only, does not affect memory strength).

    Use this when you want to look up information without reinforcing it.

    Args:
        user_id: Whose memories to search.
        query: Natural language query.
        top_k: Maximum results to return (default: 10).
        namespace: Memory namespace (default: "default").
        memory_types: Filter by types (e.g., ["fact", "preference"]).
        min_confidence: Minimum confidence threshold (0.0-1.0).
        min_strength: Minimum strength threshold (0.0-1.0).
    """
    client = _get_client()
    memories = client.search(
        user_id=user_id,
        query=query,
        top_k=top_k,
        namespace=namespace,
        memory_types=memory_types,
        min_confidence=min_confidence,
        min_strength=min_strength,
    )
    results = [_memory_to_result(m) for m in memories]
    return json.dumps(results, indent=2)


@mcp.tool()
def recall_memories(
    user_id: str,
    query: str,
    top_k: int = 10,
    namespace: str = "default",
    memory_types: Optional[list[str]] = None,
    min_confidence: float = 0.0,
    min_strength: float = 0.0,
) -> str:
    """Recall relevant memories and reinforce them (increases their strength).

    Use this during active conversations — retrieved memories get stronger,
    modeling how human memory works (frequently accessed = better retained).

    Args:
        user_id: Whose memories to recall.
        query: Natural language query.
        top_k: Maximum results to return (default: 10).
        namespace: Memory namespace (default: "default").
        memory_types: Filter by types (e.g., ["fact", "preference"]).
        min_confidence: Minimum confidence threshold (0.0-1.0).
        min_strength: Minimum strength threshold (0.0-1.0).
    """
    client = _get_client()
    memories = client.recall(
        user_id=user_id,
        query=query,
        top_k=top_k,
        namespace=namespace,
        memory_types=memory_types,
        min_confidence=min_confidence,
        min_strength=min_strength,
    )
    results = [_memory_to_result(m) for m in memories]
    return json.dumps(results, indent=2)


@mcp.tool()
def build_context(
    user_id: str,
    query: str,
    max_tokens: int = 500,
    namespace: str = "default",
) -> str:
    """Build a formatted context string from relevant memories.

    Returns a ready-to-use text block summarizing what you know about the user,
    suitable for injecting into system prompts.

    Args:
        user_id: Whose memories to use.
        query: What the conversation is about (used to find relevant memories).
        max_tokens: Approximate token budget for the context (default: 500).
        namespace: Memory namespace (default: "default").
    """
    client = _get_client()
    context = client.build_context(
        user_id=user_id,
        query=query,
        max_tokens=max_tokens,
        namespace=namespace,
    )
    return context if context else "No memories found for this user."


@mcp.tool()
def get_memory(memory_id: str) -> str:
    """Retrieve a single memory by its ID.

    Args:
        memory_id: The memory's unique identifier.
    """
    client = _get_client()
    memory = client.get(memory_id)
    if memory is None:
        return json.dumps({"error": f"Memory '{memory_id}' not found"})
    return json.dumps(_memory_to_result(memory), indent=2)


@mcp.tool()
def list_memories(
    user_id: str,
    namespace: str = "default",
    memory_types: Optional[list[str]] = None,
    active_only: bool = True,
    limit: int = 50,
    offset: int = 0,
) -> str:
    """List memories for a user with optional filters and pagination.

    Args:
        user_id: Whose memories to list.
        namespace: Memory namespace (default: "default").
        memory_types: Filter by types (e.g., ["fact", "preference"]).
        active_only: Only show active memories (default: true).
        limit: Maximum results (default: 50).
        offset: Skip first N results for pagination.
    """
    client = _get_client()
    types = [MemoryType(t) for t in memory_types] if memory_types else None
    memories = client.list(
        user_id=user_id,
        namespace=namespace,
        memory_types=types,
        active_only=active_only,
        limit=limit,
        offset=offset,
    )
    results = [_memory_to_result(m) for m in memories]
    return json.dumps(results, indent=2)


@mcp.tool()
def update_memory(
    memory_id: str,
    content: Optional[str] = None,
    memory_type: Optional[str] = None,
    confidence: Optional[float] = None,
    lifespan: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Update an existing memory's fields.

    Only provide the fields you want to change. If content changes,
    the embedding is automatically regenerated.

    Args:
        memory_id: The memory to update.
        content: New content text.
        memory_type: New type (fact, preference, insight, biographical).
        confidence: New confidence value (0.0-1.0).
        lifespan: New lifespan (short_term, working, long_term).
        metadata: New metadata (replaces existing).
    """
    client = _get_client()
    mt = MemoryType(memory_type) if memory_type else None
    ls = MemoryLifespan(lifespan) if lifespan else None
    memory = client.update(
        memory_id=memory_id,
        content=content,
        memory_type=mt,
        confidence=confidence,
        lifespan=ls,
        metadata=metadata,
    )
    if memory is None:
        return json.dumps({"error": f"Memory '{memory_id}' not found"})
    return json.dumps(_memory_to_result(memory), indent=2)


@mcp.tool()
def delete_memory(memory_id: str) -> str:
    """Permanently delete a single memory.

    Args:
        memory_id: The memory to delete.
    """
    client = _get_client()
    deleted = client.delete(memory_id)
    return json.dumps({
        "deleted": deleted,
        "memory_id": memory_id,
    })


@mcp.tool()
def delete_all_memories(user_id: str) -> str:
    """Delete ALL memories for a user (GDPR right to erasure).

    This is irreversible. Use with caution.

    Args:
        user_id: The user whose memories to delete.
    """
    client = _get_client()
    count = client.delete_all(user_id)
    return json.dumps({
        "deleted_count": count,
        "user_id": user_id,
    })


@mcp.tool()
def export_memories(user_id: str, format: str = "json") -> str:
    """Export all memories for a user.

    Args:
        user_id: Whose memories to export.
        format: "json" or "csv".
    """
    client = _get_client()
    return client.export(user_id, format=format)


@mcp.tool()
def memory_stats() -> str:
    """Get system health stats and recommendations.

    Returns total/active/deleted counts, average strength,
    and actionable recommendations.
    """
    client = _get_client()
    stats = client.stats()
    return json.dumps(dataclasses.asdict(stats), indent=2)


@mcp.tool()
def run_decay() -> str:
    """Run memory decay on all active memories.

    Applies ACT-R power-law decay — memories that haven't been accessed
    recently lose strength. Call periodically (e.g., daily).
    """
    client = _get_client()
    result = client.decay()
    return json.dumps(dataclasses.asdict(result), indent=2)


@mcp.tool()
def purge_memories() -> str:
    """Permanently remove all soft-deleted (decayed) memories.

    Returns the count of purged memories.
    """
    client = _get_client()
    count = client.purge()
    return json.dumps({"purged_count": count})


# --- Entry point ---


def main():
    """Run the OpenMem MCP server over stdio."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    mcp.run()


if __name__ == "__main__":
    main()
