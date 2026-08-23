"""
Central configuration, loaded from environment variables (and a local .env
file if python-dotenv is installed and a .env file is present).

No component reaches into os.environ directly outside this module -- that
keeps every environment-dependent knob in one place, which matters for the
"day two requirements change" scenario: a new threshold or a swapped model
should be a one-line change here, not a hunt through the codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is a convenience, not a requirement. If it isn't
    # installed, we simply rely on variables already present in the
    # environment.
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # --- Paths -----------------------------------------------------------
    policy_manual_path: Path = field(
        default_factory=lambda: PROJECT_ROOT
        / os.environ.get("POLICY_MANUAL_PATH", "data/policy-manual.md")
    )
    index_path: Path = field(
        default_factory=lambda: PROJECT_ROOT
        / os.environ.get("INDEX_PATH", "data/index")
    )

    # --- Retrieval ---------------------------------------------------------
    # Number of candidates each retriever (vector, BM25) contributes before
    # fusion. Kept generous relative to the corpus size (~230 clauses) so
    # RRF has real signal to fuse.
    retrieval_top_k: int = field(
        default_factory=lambda: _env_int("RETRIEVAL_TOP_K", 12)
    )
    # Number of fused-and-reranked candidates carried forward into evidence
    # verification.
    rerank_top_k: int = field(default_factory=lambda: _env_int("RERANK_TOP_K", 6))
    # RRF constant (standard default is 60; see DECISIONS.md).
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))

    # --- Evidence sufficiency ------------------------------------------
    # Minimum reranker relevance score (0-1, after normalisation) a clause
    # must clear to even be considered as candidate evidence. This is a
    # *relevance* gate, not a sufficiency decision -- see DECISIONS.md for
    # why those are kept separate.
    min_relevance_score: float = field(
        default_factory=lambda: _env_float("MIN_RELEVANCE_SCORE", 0.12)
    )
    # Minimum term-support score (0-1) a clause must clear for its content
    # to count as *establishing* an answer, not merely being on-topic.
    min_support_score: float = field(
        default_factory=lambda: _env_float("MIN_SUPPORT_SCORE", 0.20)
    )

    # --- LLM (optional) ---------------------------------------------------
    # Groq's OpenAI-compatible chat completions endpoint. Any other
    # OpenAI-compatible provider can be used by overriding LLM_API_BASE /
    # LLM_MODEL -- the generator (app/generation/generator.py) speaks the
    # OpenAI chat-completions request/response shape, not a Groq-specific
    # one.
    llm_api_key: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY", ""))
    llm_api_base: str = field(
        default_factory=lambda: os.environ.get(
            "LLM_API_BASE", "https://api.groq.com/openai/v1/chat/completions"
        )
    )
    llm_model: str = field(
        default_factory=lambda: os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
    )
    llm_enabled: bool = field(
        default_factory=lambda: _env_bool("LLM_ENABLED", True)
    )

    # --- Misc --------------------------------------------------------------
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))


SETTINGS = Settings()
