"""OpenMem Quickstart — demonstrates core features with mock callbacks.

Run: python examples/quickstart.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

# Ensure the src directory is on the path for running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openmem import LLMRequest, LLMResponse, OpenMem, OpenMemConfig


# ---------------------------------------------------------------------------
# Mock callbacks (no real LLM/embedding API needed)
# ---------------------------------------------------------------------------

def mock_embedding(text: str) -> list[float]:
    """Deterministic 8-dim embedding from content hash."""
    h = hashlib.md5(text.encode()).hexdigest()
    return [int(h[i : i + 2], 16) / 255.0 for i in range(0, 16, 2)]


def mock_llm(request: LLMRequest) -> LLMResponse:
    """Mock LLM that handles extraction, consolidation, and conflict prompts."""
    sys_prompt = request.system_prompt.lower()

    if "extract" in sys_prompt or "memory extraction" in sys_prompt:
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
            {
                "content": "User has a golden retriever named Max",
                "memory_type": "biographical",
                "source": "explicit",
                "confidence": 0.9,
            },
        ]
        return LLMResponse(content=json.dumps(memories))

    if "consolidat" in sys_prompt or "merg" in sys_prompt:
        return LLMResponse(content=json.dumps({
            "content": "User is a Python developer at Acme Corp",
            "memory_type": "fact",
            "confidence": 0.95,
            "reasoning": "Merged work and language preference",
        }))

    if "conflict" in sys_prompt or "contradic" in sys_prompt:
        return LLMResponse(content=json.dumps({
            "is_conflict": False,
            "explanation": "Memories are compatible",
        }))

    return LLMResponse(content="OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Use a temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "quickstart.db")
        config = OpenMemConfig(storage_path=db_path)

        mem = OpenMem(
            llm_callback=mock_llm,
            embedding_callback=mock_embedding,
            config=config,
        )

        user_id = "demo_user"

        # --- Step 1: Add memories directly ---
        print("=== Step 1: Adding memories directly ===")
        m1 = mem.add(user_id, "Likes morning coffee", confidence=1.0)
        m2 = mem.add(user_id, "Prefers dark mode in all editors")
        print(f"  Added: {m1.content} (id={m1.id[:8]}...)")
        print(f"  Added: {m2.content} (id={m2.id[:8]}...)")

        # --- Step 2: Capture from a conversation ---
        print("\n=== Step 2: Capturing from conversation ===")
        conversation = [
            {"role": "user", "content": "I prefer Python over JavaScript."},
            {"role": "assistant", "content": "I'll remember that!"},
            {"role": "user", "content": "I work at Acme Corp and I have a golden retriever named Max."},
        ]
        captured = mem.capture(user_id=user_id, messages=conversation)
        print(f"  Captured {len(captured)} memories:")
        for m in captured:
            print(f"    - {m.content} [{m.memory_type.value}, conf={m.confidence}]")

        # --- Step 3: Recall relevant memories ---
        print("\n=== Step 3: Recalling memories ===")
        results = mem.recall(user_id=user_id, query="programming language preferences")
        print(f"  Query: 'programming language preferences'")
        print(f"  Found {len(results)} relevant memories:")
        for m in results:
            print(f"    - {m.content} [strength={m.strength:.2f}]")

        # --- Step 4: Build context string ---
        print("\n=== Step 4: Building context for LLM ===")
        context = mem.build_context(user_id=user_id, query="user background")
        print(f"  Context:\n{context}")

        # --- Step 5: Run decay ---
        print("\n=== Step 5: Running decay ===")
        decay_result = mem.decay()
        print(f"  Evaluated: {decay_result.evaluated}")
        print(f"  Decayed: {decay_result.decayed}")
        print(f"  Soft-deleted: {decay_result.soft_deleted}")

        # --- Step 6: Check stats ---
        print("\n=== Step 6: Memory health stats ===")
        stats = mem.stats()
        print(f"  Total memories: {stats.total_memories}")
        print(f"  Active: {stats.active_memories}")
        print(f"  Avg strength: {stats.avg_strength}")
        if stats.recommendations:
            print(f"  Recommendations:")
            for r in stats.recommendations:
                print(f"    - {r}")

        # --- Step 7: Export ---
        print("\n=== Step 7: Export (JSON) ===")
        exported = mem.export(user_id, format="json")
        data = json.loads(exported)
        print(f"  Exported {len(data)} memories")

        mem.close()
        print("\nDone!")


if __name__ == "__main__":
    main()
