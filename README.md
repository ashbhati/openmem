# OpenMem

**The SQLite of AI memory** -- embedded, zero-dependency, lifecycle-aware memory for AI agents.

<!--
[![PyPI version](https://badge.fury.io/py/openmem.svg)](https://pypi.org/project/openmem/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://img.shields.io/badge/tests-96%20passing-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-97%25-brightgreen.svg)]()
-->

## Why OpenMem?

AI agents are stateless by default. Between sessions, they forget everything. Developers cobble together vector databases, conversation logs, and custom storage to give agents continuity -- but every existing solution requires infrastructure: servers, Docker containers, cloud services, or heavy dependency trees.

OpenMem takes a different approach. It is an **embedded** memory engine: a single `pip install`, a single `.db` file, zero running services. Behind that simplicity, it implements the complete **memory lifecycle** -- memories decay over time following cognitive science models, consolidate into higher-level understanding, and self-correct when contradictions arise.

```
pip install openmem    # numpy is the only dependency
```

Five lines to working memory:

```python
from openmem import OpenMem, LLMRequest, LLMResponse

mem = OpenMem(
    llm_callback=my_llm,           # you provide the LLM
    embedding_callback=my_embed,    # you provide the embeddings
)
```

## Features

- **Embedded** -- single `.db` file, zero infrastructure, no servers, no Docker
- **BYO LLM** -- zero LLM SDK dependencies; you provide callback functions for your model of choice
- **Memory lifecycle** -- ACT-R power-law decay, two-phase consolidation, conflict resolution
- **Ghost-memory-free** -- SQLite is the single source of truth; vector index is a derived cache
- **Hybrid search** -- semantic (cosine similarity) + keyword (FTS5), configurable ranking weights
- **GDPR-ready** -- atomic deletion with `delete()` and `delete_all()`, JSON/CSV export
- **Transparent** -- every memory is human-readable, inspectable, and auditable

## Quick Start

```python
import json
from openmem import OpenMem, OpenMemConfig, LLMRequest, LLMResponse

# --- Provide your own LLM and embedding callbacks ---

def my_llm(request: LLMRequest) -> LLMResponse:
    # Call any LLM: OpenAI, Anthropic, Ollama, etc.
    response = your_model.generate(
        system=request.system_prompt,
        user=request.user_prompt,
    )
    return LLMResponse(content=response.text)

def my_embed(text: str) -> list[float]:
    return your_model.embed(text)

# --- Initialize ---

mem = OpenMem(
    llm_callback=my_llm,
    embedding_callback=my_embed,
    storage_path="./my_agent.db",  # default: ~/.openmem/memory.db
)

# --- Capture memories from a conversation ---

memories = mem.capture(
    user_id="user_123",
    messages=[
        {"role": "user", "content": "I just moved to Austin. I'm a morning person."},
        {"role": "assistant", "content": "Welcome to Austin!"},
    ],
)
# Returns: [Memory(content="Lives in Austin", ...), Memory(content="Prefers mornings", ...)]

# --- Recall relevant memories ---

results = mem.recall(user_id="user_123", query="schedule a meeting")
for m in results:
    print(f"{m.content} [strength={m.strength:.2f}]")

# --- Build context for your LLM prompt ---

context = mem.build_context(user_id="user_123", query="help me plan my day")
# Returns: "Known about this user:\n- Lives in Austin [biographical, confidence: 0.9]\n- ..."

mem.close()
```

For a complete runnable example with mock callbacks (no API keys needed), see [`examples/quickstart.py`](examples/quickstart.py).

## Core API

### Capture -- Extract memories from conversations

```python
# Extract memories from a single conversation
memories = mem.capture(
    user_id="user_123",
    messages=[{"role": "user", "content": "..."}, ...],
    namespace="support_bot",       # optional: isolate memories by agent
    lifespan="long_term",          # optional: short_term | working | long_term
    metadata={"session_id": "s1"}, # optional: attached to all extracted memories
)

# Batch capture from multiple conversations
all_memories = mem.capture_batch(
    user_id="user_123",
    conversations=[conversation_1, conversation_2],
)
```

Internally, `capture()` sends the conversation to your LLM callback with an extraction prompt, parses the structured response, deduplicates against existing memories (exact hash + near-duplicate embedding check), detects conflicts, generates embeddings, and stores atomically in SQLite.

### Recall -- Retrieve relevant memories

```python
results = mem.recall(
    user_id="user_123",
    query="programming preferences",
    top_k=5,                             # max results (default: 10)
    min_confidence=0.5,                  # filter low-confidence memories
    min_strength=0.2,                    # filter nearly-decayed memories
    memory_types=["preference", "fact"], # optional type filter
    namespace="support_bot",             # optional namespace filter
)
```

`recall()` performs hybrid search (semantic + keyword), returns the top results, and **reinforces** returned memories -- each recall adds an access timestamp that strengthens the memory under the decay model.

### Search -- Read-only query (no side effects)

```python
results = mem.search(
    user_id="user_123",
    query="coffee preferences",
    top_k=10,
)
```

Identical to `recall()` but does **not** reinforce memories. Use this for debugging, inspection, or analytics where you don't want to influence decay.

### Lifecycle -- Decay, consolidate, resolve conflicts

```python
# Run decay on all active memories (call periodically, e.g. daily cron)
result = mem.decay()
# DecayResult(evaluated=5000, decayed=320, soft_deleted=45)

# Permanently remove soft-deleted memories
mem.purge()

# Two-phase consolidation: propose, review, apply
proposals = mem.consolidate_propose(user_id="user_123")
# [ConsolidationProposal(source_memory_ids=[...], proposed_content="...", reasoning="...")]

applied = mem.consolidate_apply(proposals)  # apply approved proposals

# Or one-step (propose + auto-apply)
consolidated = mem.consolidate(user_id="user_123")

# Find potentially contradictory memories
conflicts = mem.find_conflicts(user_id="user_123")
# [ConflictPair(memory_id_a="...", memory_id_b="...", similarity_score=0.92)]

# Health check
stats = mem.stats()
# MemoryStats(total_memories=1200, active_memories=1100, avg_strength=0.62,
#   recommendations=["Consider running mem.decay() -- last run 26h ago"])
```

### User Control -- CRUD, export, stats

```python
# Add a memory directly (without LLM extraction)
m = mem.add(user_id="user_123", content="Prefers dark mode")

# Get, list, update, delete
memory = mem.get(memory_id="01HXYZ...")
page = mem.list(user_id="user_123", limit=50, offset=0)
mem.update(memory_id="01HXYZ...", content="Prefers light mode")  # re-embeds automatically
mem.delete(memory_id="01HXYZ...")  # atomic: SQLite + cache

# Delete ALL memories for a user (GDPR right to erasure)
mem.delete_all(user_id="user_123")

# Export
data = mem.export(user_id="user_123", format="json")  # or "csv"
```

## Configuration

All configuration is optional with sensible defaults:

```python
from openmem import OpenMemConfig, MemoryLifespan, ConflictStrategy

config = OpenMemConfig(
    storage_path="~/.openmem/memory.db",    # where to store the database
    strength_threshold=0.1,                 # soft-delete memories below this strength
    max_memories_per_recall=10,             # default top_k for recall/search
    semantic_weight=0.7,                    # vs keyword weight (0.3) in hybrid search
    default_lifespan=MemoryLifespan.LONG_TERM,
    default_namespace="default",
    conflict_strategy=ConflictStrategy.KEEP_BOTH,
    dedup_similarity_threshold=0.95,        # cosine sim above this = near-duplicate
    conflict_similarity_threshold=0.85,     # cosine sim above this triggers conflict check
    vector_cache_max_users=50,              # LRU cache size for in-memory vector indices
    sqlite_busy_timeout_ms=5000,            # for multi-process access
)

mem = OpenMem(
    llm_callback=my_llm,
    embedding_callback=my_embed,
    config=config,
)
```

| Option | Default | Description |
|--------|---------|-------------|
| `storage_path` | `~/.openmem/memory.db` | SQLite database location |
| `strength_threshold` | `0.1` | Soft-delete memories with strength below this |
| `max_memories_per_recall` | `10` | Default `top_k` for `recall()` and `search()` |
| `semantic_weight` | `0.7` | Weight for semantic search (keyword weight = 1 - this) |
| `default_lifespan` | `long_term` | Default lifespan for new memories |
| `default_namespace` | `default` | Default namespace for isolation |
| `conflict_strategy` | `keep_both` | How to handle contradictions: `keep_both`, `supersede`, `keep_newer`, `keep_higher_confidence` |
| `dedup_similarity_threshold` | `0.95` | Cosine similarity above this is treated as a near-duplicate |
| `conflict_similarity_threshold` | `0.85` | Cosine similarity above this triggers conflict check |
| `vector_cache_max_users` | `50` | Max users in the LRU in-memory vector cache |
| `sqlite_busy_timeout_ms` | `5000` | SQLite busy timeout for multi-process access |
| `decay_params` | `{short_term: 0.8, working: 0.5, long_term: 0.3}` | ACT-R decay parameter `d` per lifespan category |

## How It Works

```
Your Application
       |
       v
+---------------------------------------------+
|              OpenMem Client                  |
|  +---------+  +----------+  +---------+     |
|  | Capture |  | Retention|  |  Recall |     |
|  | Engine  |  |  Engine  |  |  Engine |     |
|  +----+----+  +----+-----+  +----+----+     |
|       |             |             |          |
|  +----v-------------v-------------v------+   |
|  |           Storage Layer               |   |
|  |  +----------------------------------+ |   |
|  |  | SQLite (single source of truth)  | |   |
|  |  | - Structured fields              | |   |
|  |  | - FTS5 full-text index           | |   |
|  |  | - Embeddings as BLOBs            | |   |
|  |  +----------------+-----------------+ |   |
|  |                   | rebuilt on demand  |   |
|  |  +----------------v-----------------+ |   |
|  |  | Vector Cache (numpy, in-memory)  | |   |
|  |  | - Per-user, lazily loaded        | |   |
|  |  | - LRU eviction by user           | |   |
|  |  +----------------------------------+ |   |
|  +---------------------------------------+   |
|       |              |                       |
|  +----v--------------v----+                  |
|  | LLM Adapter Layer      |                  |
|  | (your callbacks)       |                  |
|  +------------------------+                  |
+---------------------------------------------+
```

**SQLite is the single source of truth.** All data -- structured fields, FTS5 index, and embedding BLOBs -- lives in one SQLite database file. The in-memory numpy vector cache is a derived index rebuilt from SQLite on demand. This means:

- **Crash safety**: If the process crashes after a SQLite commit but before the cache is updated, it rebuilds on restart. No data loss.
- **Atomic deletion**: Deleting from SQLite is sufficient. No orphaned embeddings influencing recall after "deletion."
- **Inspectable**: You can open the `.db` file with any SQLite tool to query memories directly.

**BYO LLM via callbacks.** OpenMem never imports an LLM SDK. You provide two callbacks:

```python
# LLM callback: OpenMem sends structured requests, you call your model
def my_llm(request: LLMRequest) -> LLMResponse:
    # request.system_prompt  -- extraction/consolidation instructions
    # request.user_prompt    -- the conversation or data to process
    # request.expected_format -- "json" or "text"
    return LLMResponse(content=result_text)

# Embedding callback: text in, vector out
def my_embed(text: str) -> list[float]:
    return embedding_vector
```

For quick prototyping, use the simplified constructor:

```python
mem = OpenMem.from_simple_callback(
    llm_fn=lambda prompt: my_model(prompt),     # (str) -> str
    embed_fn=lambda text: my_model.embed(text),  # (str) -> list[float]
)
```

## Memory Lifecycle

### Decay (ACT-R Power-Law)

Memories decay following the ACT-R cognitive architecture from psychology research. Unlike exponential decay (which treats all memories equally after the same idle period), power-law decay preserves well-rehearsed memories:

```
activation(m) = ln(sum(t_i^(-d)))
```

Where `t_i` is the time since the i-th access and `d` is the decay parameter.

- A memory accessed once yesterday and a memory accessed 100 times over the past year will have very different strengths after the same idle period
- Each `recall()` call adds an access timestamp, strengthening the memory
- Use `search()` instead of `recall()` to query without affecting decay
- Memories soft-delete when strength drops below `strength_threshold` (default: 0.1)
- Call `mem.purge()` to permanently remove soft-deleted memories

Decay parameters per lifespan:

| Lifespan | Decay `d` | Behavior |
|----------|-----------|----------|
| `short_term` | 0.8 | Fast forgetting -- session-level context |
| `working` | 0.5 | Medium -- days to weeks |
| `long_term` | 0.3 | Slow, persistent -- months to indefinite |

Decay runs **explicitly** via `mem.decay()`. There is no implicit background decay. Call it on your schedule (daily cron, startup hook, etc.).

### Consolidation (Two-Phase)

Over time, related memories accumulate. Consolidation merges them into higher-level understanding:

1. **Propose**: `mem.consolidate_propose(user_id)` identifies clusters of similar memories and uses your LLM to draft merged versions. Returns `ConsolidationProposal` objects for review.
2. **Apply**: `mem.consolidate_apply(proposals)` creates the merged memory and marks source memories as superseded (not deleted -- audit trail preserved).

Source memories retain `superseded_by` and `valid_until` fields for full provenance tracking.

For automated pipelines, `mem.consolidate(user_id)` runs both phases in one call.

### Conflict Resolution

When `capture()` detects a new memory that may contradict an existing one (cosine similarity above `conflict_similarity_threshold`), it applies the configured strategy:

| Strategy | Behavior |
|----------|----------|
| `keep_both` (default) | Store both, flag for later review via `find_conflicts()` |
| `supersede` | New memory replaces old, with `valid_until` audit trail |
| `keep_newer` | Keep the most recent memory |
| `keep_higher_confidence` | Keep whichever has higher confidence |

The default `keep_both` is intentionally conservative -- "I prefer mornings" and "I've been staying up late" may not be a true contradiction. Use `find_conflicts()` to review flagged pairs.

## Comparison

| Dimension | OpenMem | Mem0 |
|-----------|---------|------|
| **Deployment** | Embedded, single `.db` file | Requires vector DB (Qdrant default) |
| **Dependencies** | `numpy` only | Ships with `openai`, provider SDKs |
| **Infrastructure** | None -- `pip install` and go | Vector DB server required |
| **Memory lifecycle** | Decay + consolidation + conflict as integrated system | Basic memory management |
| **Atomic deletion** | SQLite single source of truth, vector cache derived | No guaranteed cross-store atomicity |
| **LLM vendor lock-in** | None -- BYO callbacks | Multi-provider, but SDKs bundled |
| **Ecosystem breadth** | SQLite + numpy | 16+ LLM providers, 24+ vector stores |
| **Graph memory** | Flat model (simpler, less expressive) | Flat model |
| **Managed cloud** | No (embedded only) | Yes (hosted platform) |
| **Scale** | Designed for per-user workloads (10K memories/user) | Designed for larger-scale multi-tenant |

OpenMem is a good fit when you want zero infrastructure, vendor independence, and lifecycle management. If you need a managed cloud service, 24+ vector store backends, or graph-based memory, other tools may be better suited.

## Examples

See the [`examples/`](examples/) directory:

- [`quickstart.py`](examples/quickstart.py) -- Complete working example with mock callbacks (no API keys needed). Demonstrates capture, recall, decay, stats, and export.
- [`chatbot_with_memory.py`](examples/chatbot_with_memory.py) -- Full chatbot example with memory lifecycle.

## Requirements

- Python 3.10+
- `numpy >= 1.24.0`
- Everything else is Python stdlib (`sqlite3`, `json`, `datetime`, `hashlib`)

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

```bash
# Clone and install in development mode
git clone https://github.com/your-org/openmem.git
cd openmem
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run tests with coverage
python -m pytest tests/ --cov=openmem --cov-report=term-missing
```

## License

Apache 2.0
