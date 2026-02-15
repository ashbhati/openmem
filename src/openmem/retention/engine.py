"""RetentionEngine — orchestrates decay, consolidation, and conflict resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..types import (
    ConflictPair,
    ConsolidationProposal,
    DecayResult,
    EmbeddingCallback,
    LLMCallback,
)
from ..models import Memory
from . import decay as decay_mod
from . import conflict as conflict_mod
from . import consolidation as consolidation_mod

if TYPE_CHECKING:
    from ..config import OpenMemConfig
    from ..storage.sqlite_store import SQLiteStore
    from ..storage.vector_cache import VectorCache


class RetentionEngine:
    """Orchestrator for memory lifecycle operations.

    Delegates to decay, consolidation, and conflict modules.
    """

    def __init__(
        self,
        store: SQLiteStore,
        vector_cache: VectorCache,
        llm_callback: Optional[LLMCallback] = None,
        embedding_callback: Optional[EmbeddingCallback] = None,
        config: Optional[OpenMemConfig] = None,
    ) -> None:
        self._store = store
        self._vector_cache = vector_cache
        self._llm_callback = llm_callback
        self._embedding_callback = embedding_callback

        if config is None:
            from ..config import OpenMemConfig
            config = OpenMemConfig()
        self._config = config

    def decay(self) -> DecayResult:
        """Run ACT-R power-law decay on all active memories."""
        return decay_mod.run_decay(self._store, self._config)

    def consolidate_propose(self, user_id: str) -> list[ConsolidationProposal]:
        """Find clusters of similar memories and propose merges."""
        if self._llm_callback is None:
            return []
        return consolidation_mod.propose_consolidations(
            store=self._store,
            vector_cache=self._vector_cache,
            llm_callback=self._llm_callback,
            user_id=user_id,
            config=self._config,
        )

    def consolidate_apply(self, proposals: list[ConsolidationProposal]) -> list[Memory]:
        """Apply approved consolidation proposals."""
        return consolidation_mod.apply_consolidations(
            store=self._store,
            vector_cache=self._vector_cache,
            embedding_callback=self._embedding_callback,
            proposals=proposals,
            config=self._config,
        )

    def find_conflicts(self, user_id: str) -> list[ConflictPair]:
        """Find potentially contradictory memories for a user."""
        if self._llm_callback is None:
            return []
        return conflict_mod.find_conflicts(
            store=self._store,
            vector_cache=self._vector_cache,
            llm_callback=self._llm_callback,
            user_id=user_id,
            config=self._config,
        )
