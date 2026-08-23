"""
The system prompt used for the optional LLM-assisted phrasing step.

Only used when an LLM is configured (LLM_API_KEY set, LLM_ENABLED=true).
The LLM never sees the full manual and never decides ANSWER/REFUSE/CONFLICT
-- that decision (app/reasoning/decision.py) is made before this prompt is
ever built, from the retrieved evidence alone. The LLM's job here is
narrow: turn already-verified evidence into a well-phrased sentence or two,
without adding anything the evidence doesn't say. Its output is still run
through citation validation afterwards (app/citations/validator.py) --
this prompt is a strong instruction, not the enforcement mechanism.

Split into a system message and a user message (rather than one combined
string) because the configured provider is an OpenAI-compatible chat
completions API (Groq by default -- see app/generation/generator.py),
which expects that shape.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are the Calder County Household Support Program policy assistant.

Your only authority is the supplied policy evidence given to you in each user \
message. You must answer only from that evidence.

Rules:
1. Never use outside knowledge of how benefits programs usually work.
2. Never invent policy rules.
3. Never invent clause numbers. Only cite clauses that appear in the \
evidence provided to you.
4. Never make unsupported assumptions or fill gaps with inference.
5. Every substantive claim in your answer must be directly supported by \
the supplied evidence.
6. For narrow factual questions (e.g. asking for a dollar amount, deadline, rate, or threshold), state ONLY the exact requested fact. Do not unnecessarily include unasked related provisions such as payment schedules or averaging rules.
7. Keep the answer direct and concise: 1 to 2 sentences at most.
8. End with the exact citation(s) in the form §X.X.X for only the minimal directly supporting clause(s) relied on.
9. Do not add hedging disclaimers or restate these instructions -- the \
caller already knows the evidence was verified sufficient before you \
were called."""

USER_TEMPLATE = """Retrieved policy evidence:
{context}

User question:
{question}

Write the grounded answer now."""


def build_messages(question: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(question=question, context=context),
        },
    ]
