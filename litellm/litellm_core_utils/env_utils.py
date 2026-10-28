"""
Utility helpers for reading and parsing environment variables.
"""

import logging
import os
from typing import Final, TypeVar

_NumberT: Final = TypeVar("_NumberT", int, float)

# Marked before logging: the LiteLLM loggers' own filters read env knobs through these helpers, so
# warning first would re-enter the helper from inside the log call and recurse until RecursionError.
_reported_unparsable_values: Final[set[tuple[str, str]]] = set()  # mutable-ok: process-wide warn-once registry


def _warn_unparsable_once(env_var: str, raw: str, number_type: type[int] | type[float]) -> None:
    key: Final = (env_var, raw)
    if key in _reported_unparsable_values:
        return
    _reported_unparsable_values.add(key)
    logging.getLogger("LiteLLM").warning(
        "%s=%r is not a valid %s. Treating it as unset.", env_var, raw, number_type.__name__
    )


def _get_env_number_or_none(env_var: str, number_type: type[_NumberT]) -> _NumberT | None:
    raw: Final = os.getenv(env_var)
    if raw is None or not raw.strip():
        return None
    try:
        return number_type(raw.strip())
    except ValueError:
        _warn_unparsable_once(env_var, raw, number_type)
        return None


def get_env_int_or_none(env_var: str) -> int | None:
    """Parse an environment variable as an integer, returning None when it is unset, blank or unparsable.

    Use this instead of `get_env_int` when callers must distinguish "explicitly configured"
    from "left at the default", for example when an override should take precedence over a
    value resolved from somewhere else.
    """
    return _get_env_number_or_none(env_var, int)


def get_env_float_or_none(env_var: str) -> float | None:
    """Float counterpart of `get_env_int_or_none`."""
    return _get_env_number_or_none(env_var, float)


def get_env_int(env_var: str, default: int) -> int:
    """Parse an environment variable as an integer, falling back to default when it is unset, blank or unparsable.

    An unparsable value is warned about once, so misconfiguration neither crashes the process at
    import time nor goes unnoticed.
    """
    value: Final = get_env_int_or_none(env_var)
    return default if value is None else value


def get_env_float(env_var: str, default: float) -> float:
    """Float counterpart of `get_env_int`."""
    value: Final = get_env_float_or_none(env_var)
    return default if value is None else value


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
