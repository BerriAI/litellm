"""Native Responses WebSocket bindings."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Final, Protocol, cast  # noqa: TID251  # native class is validated at load time

import httpx
from websockets.exceptions import ConnectionClosedOK

from litellm.rust_bridge.bindings import BINDING_UNSET, BindingUnset
from litellm.rust_bridge.responses import ROUTE
from litellm.rust_bridge.timeouts import timeout_to_seconds


class RustResponsesWebSocket(Protocol):
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
    ) -> Awaitable[RustResponsesWebSocket]: ...


def _as_connection(value: object) -> RustResponsesWebSocketConnection | None:
    if not callable(getattr(value, "connect", None)):
        return None
    return cast(RustResponsesWebSocketConnection, value)  # cast-ok: native connection factory validated above


_CONNECTION: Final = ROUTE.bind("ResponsesWebSocketConnection", validate=_as_connection)


def set_rust_responses_websocket(
    *,
    connection: RustResponsesWebSocketConnection | None | BindingUnset = BINDING_UNSET,
) -> None:
    _CONNECTION.configure(connection)


def load_rust_responses_websocket() -> RustResponsesWebSocketConnection | None:
    return ROUTE.select(_CONNECTION)


class _ConnectionAdapter:
    def __init__(self, connection: RustResponsesWebSocket):
        self._connection: Final[RustResponsesWebSocket] = connection

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
