"""Admission wrapper for the native Rust Responses WebSocket route."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol

from litellm.rust_bridge.loader import get_native_bridge


class RustResponsesWebSocketRoute(Protocol):
    def __call__(self) -> None: ...


class _Unset:
    pass


_UNSET: Final[_Unset] = _Unset()


@dataclass(slots=True)
class _RustResponsesWebSocketState:
    route: RustResponsesWebSocketRoute | None = None


_STATE: Final[_RustResponsesWebSocketState] = _RustResponsesWebSocketState()


def set_rust_responses_websocket(*, route: RustResponsesWebSocketRoute | None | _Unset = _UNSET) -> None:
    if not isinstance(route, _Unset):
        _STATE.route = route


def load_rust_responses_websocket() -> RustResponsesWebSocketRoute | None:
    if _STATE.route is not None:
        return _STATE.route
    native_bridge: Final = get_native_bridge()
    if native_bridge is None:
        return None
    route: Final[RustResponsesWebSocketRoute | None] = getattr(native_bridge, "responses_websocket", None)
    return route


def admit() -> bool:
    route: Final = load_rust_responses_websocket()
    if route is None:
        return False
    native_bridge: Final = get_native_bridge()
    declined_type: Final = getattr(native_bridge, "RustBridgeDeclined", ()) if native_bridge is not None else ()
    try:
        route()
    except declined_type:
        return False
    raise RuntimeError("Rust Responses WebSocket returned without taking session ownership")
