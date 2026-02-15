"""Shared utility helpers used across the OpenMem codebase."""

from datetime import datetime, timezone
import hashlib


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
