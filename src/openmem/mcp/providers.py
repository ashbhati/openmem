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
from pathlib import Path
from typing import Optional

logger = logging.getLogger("openmem.mcp")

_config_loaded = False


def load_config_env() -> None:
    """Load ~/.openmem/config.env as fallback for unset environment variables.

    Environment variables always take precedence. Values from config.env
    are only set if the corresponding variable is not already present.
    Safe to call multiple times (idempotent).
    """
    global _config_loaded
    if _config_loaded:
        return
    _config_loaded = True

    config_path = Path.home() / ".openmem" / "config.env"
    if not config_path.exists():
        return

    try:
        for line in config_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Only set if not already in environment
            if key not in os.environ:
                os.environ[key] = value
        logger.info("Loaded config from %s", config_path)
    except OSError as e:
        logger.warning("Could not read config file %s: %s", config_path, e)


def _get_env(name: str, fallback_name: Optional[str] = None, default: str = "") -> str:
    val = os.environ.get(name, "")
    if not val and fallback_name:
        val = os.environ.get(fallback_name, "")
    return val or default


def get_embedding_callback():
    """Build an embedding callback from environment configuration.

    Returns None if provider is "none" or no API key is available.
    Automatically loads ~/.openmem/config.env as fallback.
    """
    load_config_env()
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
