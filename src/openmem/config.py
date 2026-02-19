"""Configuration for OpenMem."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .types import ConflictStrategy, MemoryLifespan


# Default decay parameters (d) per lifespan — ACT-R power-law model
DEFAULT_DECAY_PARAMS: dict[MemoryLifespan, float] = {
    MemoryLifespan.SHORT_TERM: 0.8,
    MemoryLifespan.WORKING: 0.5,
    MemoryLifespan.LONG_TERM: 0.3,
}


@dataclass
class OpenMemConfig:
    """Configuration for an OpenMem instance.

    All fields are optional with sensible defaults.
    """

    # Storage
    storage_path: str = "~/.openmem/memory.db"

    # Decay
    strength_threshold: float = 0.1  # Soft-delete below this
    decay_params: dict[MemoryLifespan, float] = field(
        default_factory=lambda: dict(DEFAULT_DECAY_PARAMS)
    )

    # Recall
    max_memories_per_recall: int = 10
    semantic_weight: float = 0.7  # vs keyword weight (1 - semantic_weight)

    # Defaults
    default_lifespan: MemoryLifespan = MemoryLifespan.LONG_TERM
    default_namespace: str = "default"

    # Conflict resolution
    conflict_strategy: ConflictStrategy = ConflictStrategy.KEEP_BOTH
    conflict_similarity_threshold: float = 0.85  # Cosine sim above this triggers check

    # Near-duplicate detection
    dedup_similarity_threshold: float = 0.95  # Cosine sim above this = near-duplicate

    # Vector cache
    vector_cache_max_users: int = 50  # LRU cache size

    # SQLite
    sqlite_busy_timeout_ms: int = 5000

    def __post_init__(self) -> None:
        if not (0.0 <= self.semantic_weight <= 1.0):
            raise ValueError(f"semantic_weight must be in [0, 1], got {self.semantic_weight}")
        if not (0.0 <= self.strength_threshold <= 1.0):
            raise ValueError(f"strength_threshold must be in [0, 1], got {self.strength_threshold}")
        if not (0.0 <= self.conflict_similarity_threshold <= 1.0):
            raise ValueError(f"conflict_similarity_threshold must be in [0, 1], got {self.conflict_similarity_threshold}")
        if not (0.0 <= self.dedup_similarity_threshold <= 1.0):
            raise ValueError(f"dedup_similarity_threshold must be in [0, 1], got {self.dedup_similarity_threshold}")
        if self.max_memories_per_recall <= 0:
            raise ValueError(f"max_memories_per_recall must be > 0, got {self.max_memories_per_recall}")
        if self.vector_cache_max_users <= 0:
            raise ValueError(f"vector_cache_max_users must be > 0, got {self.vector_cache_max_users}")
        if self.sqlite_busy_timeout_ms <= 0:
            raise ValueError(f"sqlite_busy_timeout_ms must be > 0, got {self.sqlite_busy_timeout_ms}")

    @property
    def resolved_storage_path(self) -> Path:
        """Return the storage path with ~ expanded."""
        return Path(self.storage_path).expanduser()

    @property
    def keyword_weight(self) -> float:
        return 1.0 - self.semantic_weight
