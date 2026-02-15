"""OpenMem — The SQLite of AI memory.

Embedded, zero-dependency, lifecycle-aware memory for AI agents.

Usage::

    from openmem import OpenMem, LLMRequest, LLMResponse

    def my_llm(req: LLMRequest) -> LLMResponse:
        return LLMResponse(content=call_my_model(req.user_prompt))

    def my_embed(text: str) -> list[float]:
        return get_embedding(text)

    mem = OpenMem(llm_callback=my_llm, embedding_callback=my_embed)
    memories = mem.capture(user_id="user_1", messages=[...])
    results = mem.recall(user_id="user_1", query="...")
"""

from .client import OpenMem
from .config import OpenMemConfig
from .models import Memory
from .types import (
    ConflictPair,
    ConflictStrategy,
    ConsolidationProposal,
    DecayResult,
    LLMCallback,
    LLMRequest,
    LLMResponse,
    MemoryLifespan,
    MemorySource,
    MemoryStats,
    MemoryType,
)

__version__ = "0.1.0"

__all__ = [
    "OpenMem",
    "OpenMemConfig",
    "Memory",
    "MemoryType",
    "MemorySource",
    "MemoryLifespan",
    "ConflictStrategy",
    "LLMRequest",
    "LLMResponse",
    "LLMCallback",
    "DecayResult",
    "ConsolidationProposal",
    "ConflictPair",
    "MemoryStats",
]
