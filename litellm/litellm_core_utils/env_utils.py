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
