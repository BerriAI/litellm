"""
Internal request context for LiteLLM.

Provides a ContextVar-based mechanism for internal signals that must not
be settable from user input. Context variables are scoped to the current
asyncio task and cannot be injected via HTTP request bodies.
"""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Final

# When True, suppresses async logging and billing for internal sub-calls
# (e.g., emulated file-search steps that make nested LLM calls).
is_internal_call: Final[ContextVar[bool]] = ContextVar("is_internal_call", default=False)

# One request prices its totals, its per-token-type lines and the rates it reports on
# separate code paths. Each reads the clock for off-peak pricing, so without a pinned
# moment they can land on either side of a window boundary and disagree with each other.
_billing_time: Final[ContextVar[datetime | None]] = ContextVar("billing_time", default=None)


@contextmanager
def pinned_billing_time(moment: datetime) -> Generator[None]:
    """Price every rate lookup inside this block at ``moment`` rather than at each one's own clock read."""
    token: Final = _billing_time.set(moment)
    try:
        yield
    finally:
        _billing_time.reset(token)


def current_billing_time() -> datetime:
    """The pinned billing moment, or now in UTC outside a pinned block."""
    pinned: Final = _billing_time.get()
    return pinned if pinned is not None else datetime.now(timezone.utc)
