"""
Shared validation for the agentic loop ceiling.

``max_agentic_loops`` can be set in two places, and the two disagreed about
what a bad value means. The feature-level
``litellm_settings.websearch_interception_params.max_agentic_loops`` was
checked at config load, while a per-deployment
``model_list[].litellm_params.max_agentic_loops`` was passed straight through
to ``int(... or 3)``. That let a per-deployment ``0`` read as the default 3,
turning the tightest ceiling into the loosest one, and let a per-deployment
``"three"`` boot the proxy and then fail every request to that model.

Both settings now go through :func:`validated_max_agentic_loops`, which names
the field it rejected so the error says which line of the config to fix.

Anything that spells a whole number is still accepted, because the old
``int(... or 3)`` accepted those and a ceiling is routinely parameterized as
``max_agentic_loops: os.environ/MAX_AGENTIC_LOOPS``, which resolves to a
string. Rejecting ``"5"`` would stop such a proxy from booting on upgrade.
"""

from typing import Final

DEFAULT_MAX_AGENTIC_LOOPS: Final = 3


def _as_whole_number(value: object) -> int | None:
    """
    Return ``value`` as an int when it spells a whole number, else ``None``.

    ``bool`` is excluded explicitly because it is an ``int`` subclass, so
    ``max_agentic_loops: true`` would otherwise be read as a ceiling of 1.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def validated_max_agentic_loops(max_agentic_loops: object, field: str) -> int | None:
    """
    Return ``max_agentic_loops`` as an int, or raise naming ``field``.
    """
    if max_agentic_loops is None:
        return None
    ceiling: Final = _as_whole_number(max_agentic_loops)
    if ceiling is None:
        raise TypeError(f"{field} must be an integer, got {max_agentic_loops!r}")
    if ceiling < 1:
        raise ValueError(f"{field} must be at least 1, got {ceiling}")
    return ceiling
