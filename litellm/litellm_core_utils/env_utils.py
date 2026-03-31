"""
Utility helpers for reading and parsing environment variables.
"""

import os


def get_env_bool(env_var: str, default: bool = False) -> bool:
    """Parse an environment variable as a boolean.

    Truthy (case-insensitive, after strip): ``1``, ``true``, ``yes``, ``on``.
    If the variable is unset, returns ``default``. Any other value—including
    ``0``, ``false``, and ``no``—is False (do not wrap ``os.getenv`` strings in ``bool()``).
    """
    raw = os.getenv(env_var)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


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
