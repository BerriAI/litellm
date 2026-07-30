"""
Utility helpers for reading and parsing environment variables.
"""

import os


def get_env_int(env_var: str, default: int) -> int:
    """Parse an environment variable as an integer, falling back to default on invalid values.

    Handles empty strings, whitespace, and non-numeric values gracefully
    so that misconfiguration doesn't crash the process at import time.
    """
    raw = os.getenv(env_var)
    if raw is None:
        return default
    raw = raw.strip()
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


def get_env_int_or_none(env_var: str) -> int | None:
    """Parse an environment variable as an integer, returning None when it is unset or unusable.

    Use this instead of `get_env_int` when callers must distinguish "explicitly configured"
    from "left at the default", for example when an override should take precedence over a
    value resolved from somewhere else.
    """
    raw = os.getenv(env_var)
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except (ValueError, TypeError):
        return None
