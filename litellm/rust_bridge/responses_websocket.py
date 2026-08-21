"""Thin Python wrapper for the native Rust Responses WebSocket bridge."""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Final, Protocol

import httpx
from websockets.exceptions import ConnectionClosedOK

from litellm.rust_bridge.loader import get_native_bridge
from litellm.rust_bridge.timeouts import timeout_to_seconds


class RustResponsesWebSocketSocket(Protocol):
    """Open socket handle handed back by the native bridge."""

    def send_text(self, text: str) -> Awaitable[None]: ...

    def recv_text(self) -> Awaitable[str | None]: ...

    def close(self) -> Awaitable[None]: ...


class RustResponsesWebSocketConnection(Protocol):
    @classmethod
    def connect(
        cls,
        url: str,
        headers: dict[str, str],
        timeout_seconds: float | None,
    ) -> Awaitable[RustResponsesWebSocketSocket]:
        raise NotImplementedError


class _Unset:
    pass


_UNSET: Final[_Unset] = _Unset()


@dataclass(slots=True)
class _RustResponsesWebSocketState:
    connection: type[RustResponsesWebSocketConnection] | None = None


_STATE: Final[_RustResponsesWebSocketState] = _RustResponsesWebSocketState()


def set_rust_responses_websocket(
    *,
    connection: type[RustResponsesWebSocketConnection] | None | _Unset = _UNSET,
) -> None:
    if not isinstance(connection, _Unset):
        _STATE.connection = connection


def load_rust_responses_websocket() -> type[RustResponsesWebSocketConnection] | None:
    if _STATE.connection is not None:
        return _STATE.connection
    native_bridge: Final = get_native_bridge()
    if native_bridge is None:
        return None
    try:
        return native_bridge.ResponsesWebSocketConnection
    except AttributeError:
        return None


class _ConnectionAdapter:
    def __init__(self, connection: RustResponsesWebSocketSocket):
        self._connection = connection

    async def send(self, text: str) -> None:
        await self._connection.send_text(text)

    async def recv(self) -> str:
        message: Final = await self._connection.recv_text()
        if message is None:
            raise ConnectionClosedOK(None, None)
        return message

    async def close(self) -> None:
        await self._connection.close()


async def connect(
    *,
    url: str,
    headers: dict[str, str],
    timeout: float | httpx.Timeout | None,
) -> _ConnectionAdapter | None:
    connection_type: Final = load_rust_responses_websocket()
    if connection_type is None:
        return None
    try:
        connection: Final = await connection_type.connect(
            url=url,
            headers=headers,
            timeout_seconds=timeout_to_seconds(timeout),
        )
    except Exception:  # noqa: BLE001  # bridge failures must fall back to Python
        return None
    return _ConnectionAdapter(connection)
