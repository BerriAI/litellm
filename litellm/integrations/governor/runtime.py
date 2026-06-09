"""SDK-free entrypoints the proxy core may call before settings load.

Importing this module must not pull in redis or prisma: it imports only the pure
``model.config`` gate. The engine (which imports plumbing) is reached lazily and
only ever set by the adapter once the gate is on, so the gate-off path never
constructs a cache tier.
"""

from typing import TYPE_CHECKING, Optional

from litellm.integrations.governor.model.config import is_governor_v2_enabled

if TYPE_CHECKING:
    from litellm.integrations.governor.engine.governor import Engine

_engine: "Optional[Engine]" = None


def is_enabled() -> bool:
    return is_governor_v2_enabled()


def set_engine(engine: "Engine") -> None:
    global _engine
    _engine = engine


def get_engine() -> "Optional[Engine]":
    """Return the live engine, or None when the gate is off or the adapter has
    not built one yet. Never constructs an engine as a side effect."""
    if not is_governor_v2_enabled():
        return None
    return _engine
