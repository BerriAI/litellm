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
"""

from typing import Final

DEFAULT_MAX_AGENTIC_LOOPS: Final = 3


def validated_max_agentic_loops(max_agentic_loops: object, field: str) -> int | None:
    """
    Return ``max_agentic_loops`` as an int, or raise naming ``field``.

    ``bool`` is excluded explicitly because it is an ``int`` subclass, so
    ``max_agentic_loops: true`` would otherwise be read as a ceiling of 1.
    """
    if max_agentic_loops is None:
        return None
    if isinstance(max_agentic_loops, bool) or not isinstance(max_agentic_loops, int):
        raise TypeError(f"{field} must be an integer, got {max_agentic_loops!r}")
    if max_agentic_loops < 1:
        raise ValueError(f"{field} must be at least 1, got {max_agentic_loops}")
    return max_agentic_loops
