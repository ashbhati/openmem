"""Memory extraction via LLM callback."""

from __future__ import annotations

import json
import re
from typing import Any

from ..types import LLMCallback, LLMRequest
from .prompts import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_PROMPT_TEMPLATE, REPAIR_PROMPT


def _format_conversation(messages: list[dict[str, str]]) -> str:
    """Format a list of message dicts into a readable conversation string."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _parse_json_array(text: str) -> list[dict[str, Any]] | None:
    """Try to parse a JSON array from LLM output. Returns None on failure."""
    text = text.strip()
    # Try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        return None
    except json.JSONDecodeError:
        pass

    # Try to extract JSON array from surrounding text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return None


def _validate_memory_dict(item: dict[str, Any]) -> dict[str, Any] | None:
    """Validate and normalize a single extracted memory dict."""
    if not isinstance(item, dict):
        return None

    content = item.get("content")
    if not content or not isinstance(content, str) or not content.strip():
        return None

    valid_types = {"fact", "preference", "insight", "biographical"}
    memory_type = item.get("memory_type", "fact")
    if memory_type not in valid_types:
        memory_type = "fact"

    valid_sources = {"explicit", "implicit"}
    source = item.get("source", "implicit")
    if source not in valid_sources:
        source = "implicit"

    confidence = item.get("confidence", 0.5)
    if not isinstance(confidence, (int, float)):
        confidence = 0.5
    confidence = max(0.0, min(1.0, float(confidence)))

    return {
        "content": content.strip(),
        "memory_type": memory_type,
        "source": source,
        "confidence": confidence,
    }


def extract_memories(
    llm_callback: LLMCallback,
    messages: list[dict[str, str]],
    existing_content_hashes: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract memories from conversation messages using the LLM callback.

    Args:
        llm_callback: Function that calls the developer's LLM.
        messages: Conversation messages as list of {"role": ..., "content": ...}.
        existing_content_hashes: Optional set of hashes for pre-filtering
            (not used in extraction itself, reserved for future use).

    Returns:
        List of dicts with keys: content, memory_type, source, confidence.
    """
    conversation_text = _format_conversation(messages)
    user_prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(conversation=conversation_text)

    request = LLMRequest(
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        expected_format="json",
    )
    response = llm_callback(request)

    # Try to parse the response
    parsed = _parse_json_array(response.content)

    # If parsing failed, retry once with repair prompt
    if parsed is None:
        repair_user_prompt = REPAIR_PROMPT.format(broken_json=response.content)
        repair_request = LLMRequest(
            system_prompt="You are a JSON repair assistant. Return only valid JSON.",
            user_prompt=repair_user_prompt,
            expected_format="json",
        )
        repair_response = llm_callback(repair_request)
        parsed = _parse_json_array(repair_response.content)

    if parsed is None:
        return []

    # Validate each extracted memory
    results: list[dict[str, Any]] = []
    for item in parsed:
        validated = _validate_memory_dict(item)
        if validated is not None:
            results.append(validated)

    return results
