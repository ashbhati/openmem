"""Chatbot with Memory — demonstrates how an AI chatbot uses OpenMem.

This example shows a simple interactive loop where a chatbot:
1. Recalls context before generating each response
2. Captures memories from each exchange
3. Reports memory health via stats()

Run: python examples/chatbot_with_memory.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openmem import LLMRequest, LLMResponse, OpenMem, OpenMemConfig


# ---------------------------------------------------------------------------
# Mock callbacks
# ---------------------------------------------------------------------------

# Track conversation for extraction
_conversation_counter = 0


def mock_embedding(text: str) -> list[float]:
    """Deterministic 8-dim embedding."""
    h = hashlib.md5(text.encode()).hexdigest()
    return [int(h[i : i + 2], 16) / 255.0 for i in range(0, 16, 2)]


def mock_llm(request: LLMRequest) -> LLMResponse:
    """Mock LLM for extraction and chat."""
    global _conversation_counter
    sys_prompt = request.system_prompt.lower()

    if "extract" in sys_prompt or "memory extraction" in sys_prompt:
        user_text = request.user_prompt
        # Parse out simple facts from the conversation text
        memories = []
        if "python" in user_text.lower():
            memories.append({
                "content": "User is interested in Python",
                "memory_type": "preference",
                "source": "explicit",
                "confidence": 0.9,
            })
        if "coffee" in user_text.lower():
            memories.append({
                "content": "User enjoys coffee",
                "memory_type": "preference",
                "source": "explicit",
                "confidence": 0.85,
            })
        if "work" in user_text.lower() or "job" in user_text.lower():
            memories.append({
                "content": "User discussed their work",
                "memory_type": "fact",
                "source": "implicit",
                "confidence": 0.7,
            })
        if "name" in user_text.lower():
            memories.append({
                "content": "User shared their name",
                "memory_type": "biographical",
                "source": "explicit",
                "confidence": 0.95,
            })
        if not memories:
            memories.append({
                "content": f"User engaged in conversation (exchange #{_conversation_counter})",
                "memory_type": "insight",
                "source": "implicit",
                "confidence": 0.5,
            })
        _conversation_counter += 1
        return LLMResponse(content=json.dumps(memories))

    if "repair" in sys_prompt or "fix" in sys_prompt:
        return LLMResponse(content="[]")

    return LLMResponse(content="OK")


def generate_response(user_input: str, context: str) -> str:
    """Simulate a chatbot response (no real LLM needed)."""
    if context:
        return f"[With memory context] I remember some things about you. Regarding '{user_input}' - let me help!"
    return f"[No memory context] Thanks for sharing about '{user_input}'!"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "chatbot.db")
        config = OpenMemConfig(storage_path=db_path)

        mem = OpenMem(
            llm_callback=mock_llm,
            embedding_callback=mock_embedding,
            config=config,
        )

        user_id = "chatbot_user"

        # Simulate a multi-turn conversation
        exchanges = [
            "Hi, my name is Alice and I love Python programming!",
            "I work at a startup and drink lots of coffee.",
            "What do you remember about me?",
            "I also enjoy hiking on weekends.",
        ]

        print("=== Chatbot with Memory Demo ===\n")

        for i, user_input in enumerate(exchanges, 1):
            print(f"--- Turn {i} ---")
            print(f"User: {user_input}")

            # 1. Recall context before responding
            context = mem.build_context(user_id=user_id, query=user_input)
            if context:
                print(f"[Memory context retrieved: {len(context)} chars]")

            # 2. Generate response (using context)
            response = generate_response(user_input, context)
            print(f"Bot: {response}")

            # 3. Capture memories from this exchange
            messages = [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response},
            ]
            captured = mem.capture(user_id=user_id, messages=messages)
            if captured:
                print(f"[Captured {len(captured)} new memories]")
                for m in captured:
                    print(f"  -> {m.content}")
            print()

        # Show final memory stats
        print("=== Memory Health Report ===")
        stats = mem.stats()
        print(f"Total memories: {stats.total_memories}")
        print(f"Active memories: {stats.active_memories}")
        print(f"Average strength: {stats.avg_strength:.2f}")
        if stats.recommendations:
            print("Recommendations:")
            for r in stats.recommendations:
                print(f"  - {r}")

        # Show all stored memories
        print("\n=== All Stored Memories ===")
        all_memories = mem.list(user_id=user_id)
        for m in all_memories:
            print(f"  [{m.memory_type.value}] {m.content} "
                  f"(conf={m.confidence:.1f}, strength={m.strength:.2f})")

        # Run decay
        print("\n=== Running Decay ===")
        decay_result = mem.decay()
        print(f"Evaluated: {decay_result.evaluated}, "
              f"Decayed: {decay_result.decayed}, "
              f"Soft-deleted: {decay_result.soft_deleted}")

        mem.close()
        print("\nDone!")


if __name__ == "__main__":
    main()
