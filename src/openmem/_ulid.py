"""Minimal ULID generation — no external dependencies.

ULIDs are 26-character, lexicographically sortable, globally unique identifiers.
Format: 10 chars timestamp (48-bit ms) + 16 chars randomness (80-bit).
"""

import os
import time

_ENCODING = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford's Base32


def _encode_base32(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        chars.append(_ENCODING[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def generate_ulid() -> str:
    """Generate a new ULID string (26 characters, sortable by time)."""
    timestamp_ms = int(time.time() * 1000)
    randomness = int.from_bytes(os.urandom(10), "big")

    ts_part = _encode_base32(timestamp_ms, 10)
    rand_part = _encode_base32(randomness, 16)

    return ts_part + rand_part
