"""SQLite structured storage — single source of truth for all memory data.

Stores structured fields, FTS5 full-text index, and embeddings as BLOBs.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..models import Memory
from ..types import MemoryLifespan, MemorySource, MemoryType

_SCHEMA_VERSION = 1


def _serialize_embedding(embedding: list[float]) -> bytes:
    """Serialize a float list to a compact binary BLOB."""
    if not embedding:
        return b""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _deserialize_embedding(blob: bytes) -> list[float]:
    """Deserialize a binary BLOB back to a float list."""
    if not blob:
        return []
    n = len(blob) // 4  # float32 = 4 bytes
    return list(struct.unpack(f"{n}f", blob))


def _serialize_timestamps(timestamps: list[datetime]) -> str:
    """Serialize datetime list to JSON string."""
    return json.dumps([t.isoformat() for t in timestamps])


def _deserialize_timestamps(s: str) -> list[datetime]:
    """Deserialize JSON string to datetime list."""
    if not s:
        return []
    return [datetime.fromisoformat(t) for t in json.loads(s)]


def _dt_to_str(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat()


def _str_to_dt(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    return datetime.fromisoformat(s)


def _row_to_memory(row: sqlite3.Row) -> Memory:
    """Convert a database row to a Memory object."""
    return Memory(
        id=row["id"],
        user_id=row["user_id"],
        namespace=row["namespace"],
        content=row["content"],
        content_hash=row["content_hash"],
        memory_type=MemoryType(row["memory_type"]),
        source=MemorySource(row["source"]),
        confidence=row["confidence"],
        strength=row["strength"],
        created_at=datetime.fromisoformat(row["created_at"]),
        last_accessed=datetime.fromisoformat(row["last_accessed"]),
        access_count=row["access_count"],
        access_timestamps=_deserialize_timestamps(row["access_timestamps"]),
        lifespan=MemoryLifespan(row["lifespan"]),
        ttl=_str_to_dt(row["ttl"]),
        version=row["version"],
        superseded_by=row["superseded_by"],
        valid_from=_str_to_dt(row["valid_from"]),
        valid_until=_str_to_dt(row["valid_until"]),
        is_active=bool(row["is_active"]),
        embedding=_deserialize_embedding(row["embedding"]),
        embedding_model=row["embedding_model"],
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
    )


class SQLiteStore:
    """SQLite storage backend for OpenMem.

    All memory data lives here — structured fields, FTS5 index, and embedding
    BLOBs. This is the single source of truth; the numpy vector cache is
    a derived view rebuilt from this store.
    """

    def __init__(self, path: Path, busy_timeout_ms: int = 5000) -> None:
        self._path = path
        self._busy_timeout_ms = busy_timeout_ms
        self._write_lock = threading.Lock()
        self._local = threading.local()

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self._path),
                timeout=self._busy_timeout_ms / 1000.0,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(f"PRAGMA busy_timeout={int(self._busy_timeout_ms)}")
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        conn = self._get_conn()
        with self._write_lock:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    strength REAL NOT NULL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    access_timestamps TEXT NOT NULL DEFAULT '[]',
                    lifespan TEXT NOT NULL DEFAULT 'long_term',
                    ttl TEXT,
                    version INTEGER NOT NULL DEFAULT 1,
                    superseded_by TEXT,
                    valid_from TEXT,
                    valid_until TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    embedding BLOB,
                    embedding_model TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_memories_user_ns
                    ON memories(user_id, namespace);
                CREATE INDEX IF NOT EXISTS idx_memories_content_hash
                    ON memories(content_hash);
                CREATE INDEX IF NOT EXISTS idx_memories_active
                    ON memories(is_active);
                CREATE INDEX IF NOT EXISTS idx_memories_user_active
                    ON memories(user_id, namespace, is_active);

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content,
                    content='memories',
                    content_rowid='rowid'
                );

                -- Triggers to keep FTS index synchronized
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content)
                    VALUES (new.rowid, new.content);
                END;

                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content)
                    VALUES ('delete', old.rowid, old.content);
                END;

                DROP TRIGGER IF EXISTS memories_au;
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories
                WHEN OLD.content != NEW.content BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content)
                    VALUES ('delete', old.rowid, old.content);
                    INSERT INTO memories_fts(rowid, content)
                    VALUES (new.rowid, new.content);
                END;

                CREATE TABLE IF NOT EXISTS openmem_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
            """)
            # Store schema version
            conn.execute(
                "INSERT OR REPLACE INTO openmem_meta (key, value) VALUES (?, ?)",
                ("schema_version", str(_SCHEMA_VERSION)),
            )
            conn.commit()

    def add(self, memory: Memory) -> None:
        """Insert a new memory. Raises ValueError if ID already exists."""
        conn = self._get_conn()
        with self._write_lock:
            try:
                conn.execute(
                    """INSERT INTO memories (
                        id, user_id, namespace, content, content_hash,
                        memory_type, source, confidence, strength,
                        created_at, last_accessed, access_count, access_timestamps,
                        lifespan, ttl, version, superseded_by,
                        valid_from, valid_until, is_active,
                        embedding, embedding_model, metadata
                    ) VALUES (
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?
                    )""",
                    (
                        memory.id,
                        memory.user_id,
                        memory.namespace,
                        memory.content,
                        memory.content_hash,
                        memory.memory_type.value,
                        memory.source.value,
                        memory.confidence,
                        memory.strength,
                        _dt_to_str(memory.created_at),
                        _dt_to_str(memory.last_accessed),
                        memory.access_count,
                        _serialize_timestamps(memory.access_timestamps),
                        memory.lifespan.value,
                        _dt_to_str(memory.ttl),
                        memory.version,
                        memory.superseded_by,
                        _dt_to_str(memory.valid_from),
                        _dt_to_str(memory.valid_until),
                        int(memory.is_active),
                        _serialize_embedding(memory.embedding),
                        memory.embedding_model,
                        json.dumps(memory.metadata),
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError as e:
                conn.rollback()
                raise ValueError(f"Memory with id '{memory.id}' already exists") from e

    def get(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a single memory by ID. Returns None if not found."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_memory(row)

    def list(
        self,
        user_id: str,
        namespace: Optional[str] = None,
        memory_types: Optional[list[MemoryType]] = None,
        active_only: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Memory]:
        """List memories for a user with optional filters."""
        conn = self._get_conn()
        conditions = ["user_id = ?"]
        params: list[Any] = [user_id]

        if namespace is not None:
            conditions.append("namespace = ?")
            params.append(namespace)
        if memory_types:
            placeholders = ",".join("?" for _ in memory_types)
            conditions.append(f"memory_type IN ({placeholders})")
            params.extend(t.value for t in memory_types)
        if active_only:
            conditions.append("is_active = 1")

        where = " AND ".join(conditions)
        params.extend([limit, offset])

        rows = conn.execute(
            f"SELECT * FROM memories WHERE {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [_row_to_memory(row) for row in rows]

    def update(self, memory: Memory) -> None:
        """Update an existing memory (full replacement)."""
        conn = self._get_conn()
        with self._write_lock:
            conn.execute(
                """UPDATE memories SET
                    user_id=?, namespace=?, content=?, content_hash=?,
                    memory_type=?, source=?, confidence=?, strength=?,
                    created_at=?, last_accessed=?, access_count=?, access_timestamps=?,
                    lifespan=?, ttl=?, version=?, superseded_by=?,
                    valid_from=?, valid_until=?, is_active=?,
                    embedding=?, embedding_model=?, metadata=?
                WHERE id=?""",
                (
                    memory.user_id,
                    memory.namespace,
                    memory.content,
                    memory.content_hash,
                    memory.memory_type.value,
                    memory.source.value,
                    memory.confidence,
                    memory.strength,
                    _dt_to_str(memory.created_at),
                    _dt_to_str(memory.last_accessed),
                    memory.access_count,
                    _serialize_timestamps(memory.access_timestamps),
                    memory.lifespan.value,
                    _dt_to_str(memory.ttl),
                    memory.version,
                    memory.superseded_by,
                    _dt_to_str(memory.valid_from),
                    _dt_to_str(memory.valid_until),
                    int(memory.is_active),
                    _serialize_embedding(memory.embedding),
                    memory.embedding_model,
                    json.dumps(memory.metadata),
                    memory.id,
                ),
            )
            conn.commit()

    def delete(self, memory_id: str) -> bool:
        """Hard-delete a memory. Returns True if a row was actually deleted."""
        conn = self._get_conn()
        with self._write_lock:
            cursor = conn.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_distinct_namespaces(self, user_id: str) -> list[str]:
        """Return distinct namespaces for a user (lightweight query)."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT namespace FROM memories WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        return [row["namespace"] for row in rows]

    def delete_all(self, user_id: str) -> int:
        """Hard-delete ALL memories for a user. Returns count deleted."""
        conn = self._get_conn()
        with self._write_lock:
            cursor = conn.execute(
                "DELETE FROM memories WHERE user_id = ?", (user_id,)
            )
            conn.commit()
            return cursor.rowcount

    def soft_delete(self, memory_id: str) -> bool:
        """Mark a memory as inactive (soft-delete). Returns True if updated."""
        conn = self._get_conn()
        with self._write_lock:
            cursor = conn.execute(
                "UPDATE memories SET is_active = 0 WHERE id = ? AND is_active = 1",
                (memory_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def purge_inactive(self) -> int:
        """Permanently remove all soft-deleted (inactive) memories."""
        conn = self._get_conn()
        with self._write_lock:
            cursor = conn.execute("DELETE FROM memories WHERE is_active = 0")
            conn.commit()
            return cursor.rowcount

    def find_by_content_hash(
        self, content_hash: str, user_id: str, namespace: str
    ) -> Optional[Memory]:
        """Find an active memory by content hash (exact dedup)."""
        conn = self._get_conn()
        row = conn.execute(
            """SELECT * FROM memories
            WHERE content_hash = ? AND user_id = ? AND namespace = ? AND is_active = 1
            LIMIT 1""",
            (content_hash, user_id, namespace),
        ).fetchone()
        if row is None:
            return None
        return _row_to_memory(row)

    def fts_search(
        self, query: str, user_id: str, namespace: Optional[str] = None, limit: int = 20
    ) -> list[tuple[Memory, float]]:
        """Full-text search via FTS5. Returns (memory, rank) pairs."""
        conn = self._get_conn()
        conditions = ["m.user_id = ?", "m.is_active = 1"]
        params: list[Any] = [user_id]
        if namespace is not None:
            conditions.append("m.namespace = ?")
            params.append(namespace)

        where = " AND ".join(conditions)
        # FTS5 rank is negative (more negative = better match)
        rows = conn.execute(
            f"""SELECT m.*, fts.rank as fts_rank
            FROM memories_fts fts
            JOIN memories m ON m.rowid = fts.rowid
            WHERE memories_fts MATCH ? AND {where}
            ORDER BY fts.rank
            LIMIT ?""",
            (query, *params, limit),
        ).fetchall()

        results = []
        for row in rows:
            memory = _row_to_memory(row)
            # Normalize FTS5 rank to 0-1 (higher = better)
            rank = -row["fts_rank"] if row["fts_rank"] else 0.0
            results.append((memory, rank))
        return results

    def get_all_embeddings(
        self, user_id: str, namespace: Optional[str] = None
    ) -> list[tuple[str, list[float]]]:
        """Load all (id, embedding) pairs for a user. Used to build vector cache."""
        conn = self._get_conn()
        if namespace is not None:
            rows = conn.execute(
                """SELECT id, embedding FROM memories
                WHERE user_id = ? AND namespace = ? AND is_active = 1
                AND embedding IS NOT NULL""",
                (user_id, namespace),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT id, embedding FROM memories
                WHERE user_id = ? AND is_active = 1
                AND embedding IS NOT NULL""",
                (user_id,),
            ).fetchall()
        return [
            (row["id"], _deserialize_embedding(row["embedding"]))
            for row in rows
            if row["embedding"]
        ]

    def get_active_memories(
        self,
        user_id: Optional[str] = None,
    ) -> list[Memory]:
        """Get all active memories, optionally filtered by user."""
        conn = self._get_conn()
        if user_id:
            rows = conn.execute(
                "SELECT * FROM memories WHERE user_id = ? AND is_active = 1",
                (user_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM memories WHERE is_active = 1"
            ).fetchall()
        return [_row_to_memory(row) for row in rows]

    def batch_update_strength(
        self, updates: list[tuple[str, float, bool]]
    ) -> None:
        """Batch update strength and is_active for multiple memories.

        Args:
            updates: List of (memory_id, new_strength, new_is_active) tuples.
        """
        conn = self._get_conn()
        with self._write_lock:
            conn.executemany(
                "UPDATE memories SET strength = ?, is_active = ? WHERE id = ?",
                [(strength, int(active), mid) for mid, strength, active in updates],
            )
            conn.commit()

    def count(
        self,
        user_id: Optional[str] = None,
        active_only: bool = False,
        inactive_only: bool = False,
    ) -> int:
        """Count memories with optional filters."""
        conn = self._get_conn()
        conditions: list[str] = []
        params: list[Any] = []

        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if active_only:
            conditions.append("is_active = 1")
        if inactive_only:
            conditions.append("is_active = 0")

        where = " AND ".join(conditions) if conditions else "1=1"
        row = conn.execute(
            f"SELECT COUNT(*) as cnt FROM memories WHERE {where}", params
        ).fetchone()
        return row["cnt"]

    def get_meta(self, key: str) -> Optional[str]:
        """Get a metadata value."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM openmem_meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        """Set a metadata value."""
        conn = self._get_conn()
        with self._write_lock:
            conn.execute(
                "INSERT OR REPLACE INTO openmem_meta (key, value) VALUES (?, ?)",
                (key, value),
            )
            conn.commit()

    def batch_get(self, ids: list[str]) -> list[Memory]:
        """Retrieve multiple memories by ID in a single query.

        Returns memories in the order of the provided IDs.
        Missing IDs are silently skipped.
        """
        if not ids:
            return []
        conn = self._get_conn()
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", ids
        ).fetchall()
        row_map = {row["id"]: _row_to_memory(row) for row in rows}
        return [row_map[mid] for mid in ids if mid in row_map]

    def avg_strength(self, active_only: bool = True) -> float:
        """Compute average strength via SQL aggregate."""
        conn = self._get_conn()
        condition = "WHERE is_active = 1" if active_only else ""
        row = conn.execute(
            f"SELECT AVG(strength) as avg_str FROM memories {condition}"
        ).fetchone()
        return row["avg_str"] if row["avg_str"] is not None else 0.0

    def count_below_threshold(self, threshold: float) -> int:
        """Count active memories with strength below threshold."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE is_active = 1 AND strength < ?",
            (threshold,),
        ).fetchone()
        return row["cnt"]

    def distinct_embedding_models(self) -> set[str]:
        """Return distinct non-empty embedding models from active memories."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT embedding_model FROM memories WHERE is_active = 1 AND embedding_model != ''"
        ).fetchall()
        return {row["embedding_model"] for row in rows}

    def close(self) -> None:
        """Close the thread-local connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
