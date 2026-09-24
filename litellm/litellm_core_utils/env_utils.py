"""
Utility helpers for reading and parsing environment variables.
"""

import logging
import os
from typing import Final


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


def get_env_int_in_range(env_var: str, default: int, minimum: int, maximum: int) -> int:
    """Parse an environment variable as an integer constrained to ``[minimum, maximum]``.

    Values outside the range fall back to the default and warn, so a misconfigured knob can
    neither crash the caller nor silently change the meaning of what it computes.
    """
    value: Final = get_env_int(env_var, default)
    if minimum <= value <= maximum:
        return value
    logging.getLogger("LiteLLM").warning(
        "%s=%s is outside the supported range [%s, %s]. Falling back to %s.",
        env_var,
        value,
        minimum,
        maximum,
        default,
    )
    return default


def get_env_int_or_none(env_var: str) -> int | None:
    """Parse an environment variable as an integer, returning None when it is unset or unusable.

    Use this instead of `get_env_int` when callers must distinguish "explicitly configured"
    from "left at the default", for example when an override should take precedence over a
    value resolved from somewhere else.
    """
    raw: Final = os.getenv(env_var)
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except (ValueError, TypeError):
        return None
