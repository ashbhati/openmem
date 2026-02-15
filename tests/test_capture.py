"""Tests for the capture engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmem import (
    LLMRequest,
    LLMResponse,
    OpenMem,
    OpenMemConfig,
)


USER = "test_user"

SAMPLE_MESSAGES = [
    {"role": "user", "content": "I prefer Python over JavaScript"},
    {"role": "assistant", "content": "Good to know! Python is great."},
    {"role": "user", "content": "I work at Acme Corp"},
]


class TestCaptureBasic:
    def test_capture_extracts_memories_from_conversation(self, mem: OpenMem):
        memories = mem.capture(user_id=USER, messages=SAMPLE_MESSAGES)
        assert len(memories) >= 1
        # Each memory should have content and embedding
        for m in memories:
            assert m.content
            assert m.user_id == USER
            assert m.embedding
            assert m.is_active is True
            assert m.strength == 1.0

    def test_capture_deduplicates_by_content_hash(self, mem: OpenMem):
        """Running capture twice with same conversation should not duplicate."""
        first = mem.capture(user_id=USER, messages=SAMPLE_MESSAGES)
        assert len(first) >= 1

        second = mem.capture(user_id=USER, messages=SAMPLE_MESSAGES)
        assert len(second) == 0  # All duplicates should be filtered

    def test_capture_deduplicates_by_embedding_similarity(self, tmp_path: Path):
        """Near-duplicate content (different wording, same meaning) should be deduped."""
        call_count = 0

        def embedding_fn(text: str) -> list[float]:
            # Return very similar embeddings for any text to trigger dedup
            return [0.5] * 8

        def llm_fn(req: LLMRequest) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(content=json.dumps([
                    {"content": "User prefers Python", "memory_type": "preference",
                     "source": "explicit", "confidence": 0.9},
                ]))
            else:
                return LLMResponse(content=json.dumps([
                    {"content": "User likes Python language", "memory_type": "preference",
                     "source": "explicit", "confidence": 0.9},
                ]))

        config = OpenMemConfig(
            storage_path=str(tmp_path / "dedup.db"),
            dedup_similarity_threshold=0.95,
        )
        m = OpenMem(
            llm_callback=llm_fn,
            embedding_callback=embedding_fn,
            config=config,
        )
        first = m.capture(user_id=USER, messages=SAMPLE_MESSAGES)
        assert len(first) == 1

        # Second capture: content differs but embedding is identical => deduped
        second = m.capture(user_id=USER, messages=SAMPLE_MESSAGES)
        assert len(second) == 0
        m.close()


class TestCaptureNamespace:
    def test_capture_with_namespace_isolation(self, mem: OpenMem):
        msgs_a = [{"role": "user", "content": "I like cats"}]
        msgs_b = [{"role": "user", "content": "I like dogs"}]

        mem_a = mem.capture(user_id=USER, messages=msgs_a, namespace="agent_a")
        mem_b = mem.capture(user_id=USER, messages=msgs_b, namespace="agent_b")

        list_a = mem.list(USER, namespace="agent_a")
        list_b = mem.list(USER, namespace="agent_b")

        # Each namespace has its own memories
        assert len(list_a) >= 1
        assert len(list_b) >= 1
        ids_a = {m.id for m in list_a}
        ids_b = {m.id for m in list_b}
        assert ids_a.isdisjoint(ids_b)


class TestCaptureParseFailure:
    def test_capture_handles_parse_failure_with_retry(self, tmp_path: Path):
        """When the LLM returns invalid JSON, capture retries with repair prompt."""
        call_count = 0

        def llm_fn(req: LLMRequest) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Return invalid JSON first time
                return LLMResponse(content="This is not valid JSON [broken")
            else:
                # Repair prompt: return valid JSON
                return LLMResponse(content=json.dumps([
                    {"content": "User likes pizza", "memory_type": "preference",
                     "source": "explicit", "confidence": 0.8},
                ]))

        def embed_fn(text: str) -> list[float]:
            return [0.1] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "retry.db"))
        m = OpenMem(llm_callback=llm_fn, embedding_callback=embed_fn, config=config)
        memories = m.capture(user_id=USER, messages=SAMPLE_MESSAGES)
        assert len(memories) == 1
        assert memories[0].content == "User likes pizza"
        assert call_count == 2  # initial + repair
        m.close()


class TestExtractorEdgeCases:
    def test_extractor_handles_non_dict_items(self, tmp_path: Path):
        """Non-dict items in LLM response should be silently skipped."""
        def llm_fn(req: LLMRequest) -> LLMResponse:
            # Array with a non-dict item mixed in
            return LLMResponse(content=json.dumps([
                "just a string",
                {"content": "Valid memory", "memory_type": "fact",
                 "source": "explicit", "confidence": 0.9},
            ]))

        def embed_fn(text: str) -> list[float]:
            return [0.1] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "non_dict.db"))
        m = OpenMem(llm_callback=llm_fn, embedding_callback=embed_fn, config=config)
        memories = m.capture(user_id=USER, messages=SAMPLE_MESSAGES)
        # Only the valid dict should produce a memory
        assert len(memories) == 1
        assert memories[0].content == "Valid memory"
        m.close()

    def test_extractor_handles_empty_content(self, tmp_path: Path):
        """Memory with empty content should be filtered out."""
        def llm_fn(req: LLMRequest) -> LLMResponse:
            return LLMResponse(content=json.dumps([
                {"content": "", "memory_type": "fact",
                 "source": "explicit", "confidence": 0.9},
                {"content": "Real memory", "memory_type": "fact",
                 "source": "explicit", "confidence": 0.9},
            ]))

        def embed_fn(text: str) -> list[float]:
            return [0.1] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "empty_content.db"))
        m = OpenMem(llm_callback=llm_fn, embedding_callback=embed_fn, config=config)
        memories = m.capture(user_id=USER, messages=SAMPLE_MESSAGES)
        assert len(memories) == 1
        assert memories[0].content == "Real memory"
        m.close()

    def test_extractor_handles_invalid_type(self, tmp_path: Path):
        """Memory with invalid memory_type should default to 'fact'."""
        def llm_fn(req: LLMRequest) -> LLMResponse:
            return LLMResponse(content=json.dumps([
                {"content": "Some memory", "memory_type": "invalid_type",
                 "source": "explicit", "confidence": 0.9},
            ]))

        def embed_fn(text: str) -> list[float]:
            return [0.1] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "invalid_type.db"))
        m = OpenMem(llm_callback=llm_fn, embedding_callback=embed_fn, config=config)
        memories = m.capture(user_id=USER, messages=SAMPLE_MESSAGES)
        assert len(memories) == 1
        assert memories[0].memory_type.value == "fact"
        m.close()

    def test_extractor_handles_invalid_confidence(self, tmp_path: Path):
        """Memory with string confidence should default to 0.5."""
        def llm_fn(req: LLMRequest) -> LLMResponse:
            return LLMResponse(content=json.dumps([
                {"content": "Some memory", "memory_type": "fact",
                 "source": "explicit", "confidence": "high"},
            ]))

        def embed_fn(text: str) -> list[float]:
            return [0.1] * 8

        config = OpenMemConfig(storage_path=str(tmp_path / "invalid_conf.db"))
        m = OpenMem(llm_callback=llm_fn, embedding_callback=embed_fn, config=config)
        memories = m.capture(user_id=USER, messages=SAMPLE_MESSAGES)
        assert len(memories) == 1
        assert memories[0].confidence == 0.5
        m.close()


class TestCaptureBatch:
    def test_capture_batch_multiple_conversations(self, tmp_path: Path):
        call_count = 0

        def llm_fn(req: LLMRequest) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            # Return different memories for each call
            if call_count == 1:
                return LLMResponse(content=json.dumps([
                    {"content": "User likes morning meetings", "memory_type": "preference",
                     "source": "explicit", "confidence": 0.9},
                ]))
            else:
                return LLMResponse(content=json.dumps([
                    {"content": "User uses VS Code", "memory_type": "fact",
                     "source": "explicit", "confidence": 1.0},
                ]))

        def embed_fn(text: str) -> list[float]:
            import hashlib
            h = hashlib.md5(text.encode()).hexdigest()
            return [int(h[i:i+2], 16) / 255.0 for i in range(0, 16, 2)]

        config = OpenMemConfig(storage_path=str(tmp_path / "batch.db"))
        m = OpenMem(llm_callback=llm_fn, embedding_callback=embed_fn, config=config)

        convos = [
            [{"role": "user", "content": "I like morning meetings"}],
            [{"role": "user", "content": "I use VS Code"}],
        ]
        memories = m.capture_batch(user_id=USER, conversations=convos)
        assert len(memories) == 2
        contents = {mem.content for mem in memories}
        assert "User likes morning meetings" in contents
        assert "User uses VS Code" in contents
        m.close()
