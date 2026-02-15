"""Memory data model — the atomic unit of OpenMem."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .types import MemoryLifespan, MemorySource, MemoryType
from ._utils import utc_now as _now


@dataclass
class Memory:
    """A single piece of information the system retains about a user.

    Attributes:
        id: Unique identifier (ULID — sortable, globally unique).
        user_id: Who this memory belongs to.
        namespace: Isolation scope (default: "default"). Allows multiple agents
            to share a .db without intermingling memories.
        content: Human-readable text (e.g., "Prefers morning meetings").
        content_hash: SHA-256 of content for O(1) exact deduplication.
        memory_type: Classification: fact, insight, preference, biographical.
        source: How acquired: explicit (user stated) or implicit (inferred).
        confidence: 0.0–1.0 (facts ~1.0, insights lower).
        strength: 0.0–1.0, decays over time, reinforced on access.
        created_at: When this memory was first created.
        last_accessed: When this memory was last retrieved.
        access_count: Total number of times accessed.
        access_timestamps: Full access history for ACT-R power-law decay.
        lifespan: Decay category: short_term, working, long_term.
        ttl: Hard expiry (absolute timestamp). Takes precedence over decay.
        version: Incremented on update; starts at 1.
        superseded_by: ID of the memory that replaced this one (if any).
        valid_from: When this memory became true.
        valid_until: When this memory stopped being true (set on supersede).
        is_active: False if soft-deleted by decay or superseded.
        embedding: Vector representation for semantic search.
        embedding_model: Identifier for the embedding model used.
        metadata: Arbitrary developer-defined key-value pairs.
    """

    id: str = ""
    user_id: str = ""
    namespace: str = "default"
    content: str = ""
    content_hash: str = ""
    memory_type: MemoryType = MemoryType.FACT
    source: MemorySource = MemorySource.EXPLICIT
    confidence: float = 1.0

    # Lifecycle
    strength: float = 1.0
    created_at: datetime = field(default_factory=_now)
    last_accessed: datetime = field(default_factory=_now)
    access_count: int = 0
    access_timestamps: list[datetime] = field(default_factory=list)
    lifespan: MemoryLifespan = MemoryLifespan.LONG_TERM
    ttl: Optional[datetime] = None

    # Versioning & conflict resolution
    version: int = 1
    superseded_by: Optional[str] = None
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    is_active: bool = True

    # Storage
    embedding: list[float] = field(default_factory=list)
    embedding_model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.valid_from is None:
            self.valid_from = self.created_at

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary (for export/JSON)."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "namespace": self.namespace,
            "content": self.content,
            "content_hash": self.content_hash,
            "memory_type": self.memory_type.value,
            "source": self.source.value,
            "confidence": self.confidence,
            "strength": self.strength,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "access_timestamps": [t.isoformat() for t in self.access_timestamps],
            "lifespan": self.lifespan.value,
            "ttl": self.ttl.isoformat() if self.ttl else None,
            "version": self.version,
            "superseded_by": self.superseded_by,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "is_active": self.is_active,
            "embedding_model": self.embedding_model,
            "metadata": self.metadata,
        }
