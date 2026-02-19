"""Shared utility for ensuring user vector cache is populated.

Eliminates duplication across client, capture engine, consolidation, and conflict modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sqlite_store import SQLiteStore
    from .vector_cache import VectorCache


def ensure_user_cache(
    store: SQLiteStore,
    vector_cache: VectorCache,
    user_id: str,
    namespace: str,
) -> str:
    """Ensure vector cache is populated for a user+namespace. Returns user_key.

    Args:
        store: The SQLite store to load embeddings from.
        vector_cache: The vector cache to populate.
        user_id: The user ID.
        namespace: The namespace (must be the actual namespace, not a default).

    Returns:
        The cache key string "{user_id}:{namespace}".
    """
    user_key = f"{user_id}:{namespace}"
    if not vector_cache.has_user(user_key):
        embeddings = store.get_all_embeddings(user_id, namespace=namespace)
        vector_cache.build_user_index(user_key, embeddings)
    return user_key
