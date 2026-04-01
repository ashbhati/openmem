"""Embedding provider for the MCP server.

Uses stdlib urllib to call OpenAI-compatible embedding APIs — no extra deps.

Configuration via environment variables:
    OPENMEM_EMBEDDING_PROVIDER  - "openai" (default) or "none"
    OPENMEM_EMBEDDING_MODEL     - model name (default: "text-embedding-3-small")
    OPENMEM_EMBEDDING_API_KEY   - API key (falls back to OPENAI_API_KEY)
    OPENMEM_EMBEDDING_BASE_URL  - base URL (default: "https://api.openai.com/v1")
    OPENMEM_EMBEDDING_DIMENSIONS - optional dimension override
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Optional

logger = logging.getLogger("openmem.mcp")


def _get_env(name: str, fallback_name: Optional[str] = None, default: str = "") -> str:
    val = os.environ.get(name, "")
    if not val and fallback_name:
        val = os.environ.get(fallback_name, "")
    return val or default


def get_embedding_callback():
    """Build an embedding callback from environment configuration.

    Returns None if provider is "none" or no API key is available.
    """
    provider = _get_env("OPENMEM_EMBEDDING_PROVIDER", default="openai").lower()

    if provider == "none":
        logger.info("Embedding provider set to 'none' — semantic search disabled")
        return None

    api_key = _get_env("OPENMEM_EMBEDDING_API_KEY", fallback_name="OPENAI_API_KEY")
    if not api_key:
        logger.warning(
            "No embedding API key found (set OPENMEM_EMBEDDING_API_KEY or OPENAI_API_KEY). "
            "Semantic search will be disabled."
        )
        return None

    model = _get_env("OPENMEM_EMBEDDING_MODEL", default="text-embedding-3-small")
    base_url = _get_env("OPENMEM_EMBEDDING_BASE_URL", default="https://api.openai.com/v1")
    base_url = base_url.rstrip("/")

    dimensions_str = _get_env("OPENMEM_EMBEDDING_DIMENSIONS")
    dimensions = int(dimensions_str) if dimensions_str else None

    def embedding_callback(text: str) -> list[float]:
        """Call OpenAI-compatible embedding API."""
        url = f"{base_url}/embeddings"
        body: dict = {"input": text, "model": model}
        if dimensions is not None:
            body["dimensions"] = dimensions

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        return result["data"][0]["embedding"]

    logger.info("Embedding provider: %s (model=%s)", provider, model)
    return embedding_callback
