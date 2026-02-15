"""Built-in prompt templates for memory extraction."""

EXTRACTION_SYSTEM_PROMPT = """\
You are a memory extraction engine. Your job is to analyze conversations and \
extract discrete, reusable facts, preferences, insights, and biographical \
details about the user.

Rules:
- Extract only information that would be useful to remember for future \
conversations.
- Each memory should be a single, self-contained statement.
- Do not extract trivial or transient information (e.g., "user said hello").
- Classify each memory by type:
  - fact: Objective information stated by the user (e.g., "Works at Acme Corp").
  - preference: User likes, dislikes, or preferences (e.g., "Prefers dark mode").
  - insight: Inferred behavioral patterns or tendencies (e.g., "Tends to work \
late at night").
  - biographical: Personal life details (e.g., "Has two children").
- Classify each memory by source:
  - explicit: The user directly stated this information.
  - implicit: This was inferred from context or behavior.
- Assign a confidence score from 0.0 to 1.0:
  - 1.0 for directly stated facts.
  - 0.7-0.9 for strong inferences.
  - 0.4-0.6 for weaker inferences.
- Respond ONLY with a JSON array. No other text."""

EXTRACTION_USER_PROMPT_TEMPLATE = """\
Extract memories from the following conversation. Return a JSON array of objects.

Each object must have these fields:
- "content": string — the memory text
- "memory_type": string — one of "fact", "preference", "insight", "biographical"
- "source": string — one of "explicit", "implicit"
- "confidence": number — 0.0 to 1.0

Conversation:
{conversation}

Respond with ONLY a valid JSON array. Example format:
[
  {{"content": "Prefers Python over JavaScript", "memory_type": "preference", "source": "explicit", "confidence": 0.95}}
]"""

REPAIR_PROMPT = """\
The following text was supposed to be a valid JSON array of memory objects, \
but it failed to parse. Please fix it and return ONLY a valid JSON array.

Each object must have these fields:
- "content": string
- "memory_type": string (one of "fact", "preference", "insight", "biographical")
- "source": string (one of "explicit", "implicit")
- "confidence": number (0.0 to 1.0)

Broken text:
{broken_json}

Return ONLY the corrected JSON array, nothing else."""
