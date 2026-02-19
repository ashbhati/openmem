"""Callback type definitions and enums for OpenMem."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Optional


class MemoryType(str, Enum):
    """Classification of memory content."""

    FACT = "fact"
    PREFERENCE = "preference"
    INSIGHT = "insight"
    BIOGRAPHICAL = "biographical"


class MemorySource(str, Enum):
    """How the memory was acquired."""

    EXPLICIT = "explicit"  # User directly stated
    IMPLICIT = "implicit"  # System inferred


class MemoryLifespan(str, Enum):
    """Expected duration category, determines decay rate."""

    SHORT_TERM = "short_term"
    WORKING = "working"
    LONG_TERM = "long_term"


class ConflictStrategy(str, Enum):
    """How to handle contradictory memories."""

    KEEP_BOTH = "keep_both"
    SUPERSEDE = "supersede"
    KEEP_NEWER = "keep_newer"
    KEEP_HIGHER_CONFIDENCE = "keep_higher_confidence"


# --- LLM Callback Types ---


@dataclass
class LLMRequest:
    """Structured request to the developer's LLM callback."""

    system_prompt: str
    user_prompt: str
    expected_format: str = "json"  # "json" | "text"
    max_tokens: Optional[int] = None


@dataclass
class LLMResponse:
    """Structured response from the developer's LLM callback."""

    content: str
    tokens_used: Optional[int] = None
    model: Optional[str] = None


# Callback type aliases
LLMCallback = Callable[[LLMRequest], LLMResponse]
AsyncLLMCallback = Callable[[LLMRequest], Awaitable[LLMResponse]]
EmbeddingCallback = Callable[[str], list[float]]
AsyncEmbeddingCallback = Callable[[str], Awaitable[list[float]]]


# --- Result Types ---


@dataclass
class DecayResult:
    """Result of running mem.decay()."""

    evaluated: int = 0
    decayed: int = 0
    soft_deleted: int = 0


@dataclass
class ConsolidationProposal:
    """A proposed merge of related memories."""

    source_memory_ids: list[str] = field(default_factory=list)
    proposed_content: str = ""
    reasoning: str = ""
    proposed_memory_type: Optional[MemoryType] = None
    proposed_confidence: Optional[float] = None


@dataclass
class ConflictPair:
    """A pair of potentially contradictory memories."""

    memory_id_a: str = ""
    memory_id_b: str = ""
    similarity_score: float = 0.0
    explanation: str = ""


@dataclass
class MemoryStats:
    """Health metrics for the memory system."""

    total_memories: int = 0
    active_memories: int = 0
    soft_deleted: int = 0
    avg_strength: float = 0.0
    memories_below_threshold: int = 0
    last_decay_run: Optional[str] = None
    hours_since_last_decay: Optional[float] = None
    distinct_embedding_models: int = 0
    recommendations: list[str] = field(default_factory=list)
