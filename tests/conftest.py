"""Shared fixtures for OpenMem tests."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from openmem import (
    LLMRequest,
    LLMResponse,
    Memory,
    MemoryLifespan,
    MemorySource,
    MemoryType,
    OpenMem,
    OpenMemConfig,
)


# ---------------------------------------------------------------------------
# Mock callbacks
# ---------------------------------------------------------------------------

def _mock_embedding(text: str) -> list[float]:
    """Deterministic 8-dim embedding based on content hash."""
    h = hashlib.md5(text.encode()).hexdigest()
    return [int(h[i : i + 2], 16) / 255.0 for i in range(0, 16, 2)]


def _mock_llm(request: LLMRequest) -> LLMResponse:
    """Deterministic LLM callback that handles extraction, consolidation, and conflict prompts."""
    sys = request.system_prompt.lower()
    user = request.user_prompt.lower()

    # --- Extraction ---
    if "extract" in sys or "memory extraction" in sys:
        memories = [
            {
                "content": "User prefers Python over JavaScript",
                "memory_type": "preference",
                "source": "explicit",
                "confidence": 0.95,
            },
            {
                "content": "User works at Acme Corp",
                "memory_type": "fact",
                "source": "explicit",
                "confidence": 1.0,
            },
        ]
        return LLMResponse(content=json.dumps(memories))

    # --- JSON repair ---
    if "repair" in sys or "fix" in sys:
        memories = [
            {
                "content": "User prefers Python over JavaScript",
                "memory_type": "preference",
                "source": "explicit",
                "confidence": 0.95,
            },
        ]
        return LLMResponse(content=json.dumps(memories))

    # --- Consolidation / merge ---
    if "consolidat" in sys or "merg" in sys:
        result = {
            "content": "User is a Python developer at Acme Corp who prefers dark mode",
            "memory_type": "fact",
            "confidence": 0.9,
            "reasoning": "Merged related facts about user's work and preferences",
        }
        return LLMResponse(content=json.dumps(result))

    # --- Conflict detection ---
    if "conflict" in sys or "contradic" in sys:
        if "contradict" in user:
            result = {
                "is_conflict": True,
                "explanation": "The two memories contain contradictory information",
            }
        else:
            result = {
                "is_conflict": False,
                "explanation": "The memories are compatible",
            }
        return LLMResponse(content=json.dumps(result))

    # --- Default ---
    return LLMResponse(content="OK")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_callback():
    """Return the mock LLM callback."""
    return _mock_llm


@pytest.fixture
def mock_embedding_callback():
    """Return the mock embedding callback."""
    return _mock_embedding


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Return a temporary database path."""
    return tmp_path / "test_memory.db"


@pytest.fixture
def mem(tmp_db_path: Path) -> OpenMem:
    """Pre-configured OpenMem instance with mock callbacks and temp db."""
    config = OpenMemConfig(storage_path=str(tmp_db_path))
    instance = OpenMem(
        llm_callback=_mock_llm,
        embedding_callback=_mock_embedding,
        storage_path=str(tmp_db_path),
        config=config,
    )
    yield instance
    instance.close()


@pytest.fixture
def sample_memories() -> list[dict]:
    """Return sample memory dicts for testing."""
    return [
        {
            "content": "User prefers Python over JavaScript",
            "memory_type": MemoryType.PREFERENCE,
            "source": MemorySource.EXPLICIT,
            "confidence": 0.95,
        },
        {
            "content": "User works at Acme Corp",
            "memory_type": MemoryType.FACT,
            "source": MemorySource.EXPLICIT,
            "confidence": 1.0,
        },
        {
            "content": "User tends to work late at night",
            "memory_type": MemoryType.INSIGHT,
            "source": MemorySource.IMPLICIT,
            "confidence": 0.7,
        },
        {
            "content": "User has two children",
            "memory_type": MemoryType.BIOGRAPHICAL,
            "source": MemorySource.EXPLICIT,
            "confidence": 1.0,
        },
    ]
