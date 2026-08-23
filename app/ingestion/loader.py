"""
Loads the raw policy manual from disk.

Deliberately does nothing clever -- reading the file is not where the risk
in this system lives, so this stays a thin, obviously-correct wrapper.
"""

from __future__ import annotations

from pathlib import Path


class ManualNotFoundError(FileNotFoundError):
    pass


def load_manual_text(path: Path) -> str:
    """Read the policy manual and return its raw text.

    Raises ManualNotFoundError with a clear message if the file is missing,
    rather than letting a bare FileNotFoundError with a confusing traceback
    surface to a CLI user.
    """
    if not path.exists():
        raise ManualNotFoundError(
            f"Policy manual not found at {path}. "
            "Check POLICY_MANUAL_PATH in your .env, or that data/policy-manual.md "
            "exists in the project."
        )
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ManualNotFoundError(f"Policy manual at {path} is empty.")
    return text
