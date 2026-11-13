"""The real native session pin store without Redis, optionally on a clock the test moves, as ``InMemoryCache``'s
``clock`` argument does on main."""

from __future__ import annotations

from collections.abc import Callable

from litellm.redis.stores.router_session_pins import RouterSessionPinStore
from litellm.rust_bridge._native import NativeRouterSessionPins


def memory_pins(clock: Callable[[], float] | None = None) -> RouterSessionPinStore:
    return RouterSessionPinStore(NativeRouterSessionPins(None, clock))
