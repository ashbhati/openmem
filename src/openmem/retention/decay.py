"""ACT-R power-law decay — memory strength fades without reinforcement.

Based on Anderson's ACT-R theory: activation = ln(sum(t_i^{-d}))
where t_i is the time since the i-th access and d is the decay parameter.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING

from ..types import DecayResult, MemoryLifespan
from .._utils import utc_now as _now

if TYPE_CHECKING:
    from ..config import OpenMemConfig
    from ..storage.sqlite_store import SQLiteStore


def compute_activation(
    access_timestamps: list[datetime],
    decay_param: float,
    now: datetime | None = None,
) -> float:
    """Compute ACT-R base-level activation from access history.

    activation = ln(sum(t_i^{-d})) where t_i = seconds since i-th access.

    Args:
        access_timestamps: List of datetimes when memory was accessed.
        decay_param: The decay exponent d (higher = faster decay).
        now: Current time. Defaults to utcnow.

    Returns:
        Raw activation value (can be negative for rarely accessed memories).
    """
    if not access_timestamps:
        return 0.0

    if now is None:
        now = _now()

    total = 0.0
    for ts in access_timestamps:
        t_i = (now - ts).total_seconds()
        # Clamp to minimum 1 second to avoid division by zero / huge values
        t_i = max(t_i, 1.0)
        total += t_i ** (-decay_param)

    if total <= 0:
        return 0.0

    return math.log(total)


def normalize_strength(activation: float) -> float:
    """Map raw activation to 0.0-1.0 via sigmoid.

    strength = 1 / (1 + exp(-activation))
    """
    # Clamp input to avoid overflow in exp
    activation = max(min(activation, 500.0), -500.0)
    strength = 1.0 / (1.0 + math.exp(-activation))
    return max(0.0, min(1.0, strength))


def run_decay(store: SQLiteStore, config: OpenMemConfig) -> DecayResult:
    """Run decay on all active memories.

    For each active memory:
    1. Compute activation from access_timestamps using the decay_param for its lifespan.
    2. Normalize activation to strength via sigmoid.
    3. If strength < config.strength_threshold, soft-delete the memory.
    4. Batch-update all strengths in one transaction.

    Returns:
        DecayResult with counts of evaluated, decayed, and soft-deleted memories.
    """
    now = _now()
    memories = store.get_active_memories()

    if not memories:
        store.set_meta("last_decay_run", now.isoformat())
        return DecayResult(evaluated=0, decayed=0, soft_deleted=0)

    updates: list[tuple[str, float, bool]] = []
    decayed = 0
    soft_deleted = 0

    for memory in memories:
        # Check hard TTL first
        if memory.ttl and memory.ttl <= now:
            updates.append((memory.id, 0.0, False))
            soft_deleted += 1
            continue

        decay_param = config.decay_params.get(
            memory.lifespan,
            config.decay_params[MemoryLifespan.LONG_TERM],
        )

        activation = compute_activation(memory.access_timestamps, decay_param, now)
        new_strength = normalize_strength(activation)

        if new_strength < memory.strength:
            decayed += 1

        if new_strength < config.strength_threshold:
            updates.append((memory.id, new_strength, False))
            soft_deleted += 1
        else:
            updates.append((memory.id, new_strength, True))

    store.batch_update_strength(updates)
    store.set_meta("last_decay_run", now.isoformat())

    return DecayResult(
        evaluated=len(memories),
        decayed=decayed,
        soft_deleted=soft_deleted,
    )
