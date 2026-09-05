"""Thin Python wrapper for the native Rust Responses WebSocket bridge."""

from __future__ import annotations

from typing import Final

import httpx
from websockets.exceptions import ConnectionClosedOK

from litellm.rust_bridge.bindings import UNCHANGED, Unchanged
from litellm.rust_bridge.configuration import rust_enabled
from litellm.rust_bridge.protocols import (
    RustResponsesWebSocket,
    RustResponsesWebSocketConnection,
)
from litellm.rust_bridge.runtime import (
    BridgeErrorContext,
    EndpointBinding,
    async_none,
    identity,
)
from litellm.rust_bridge.timeouts import timeout_to_seconds

_RESPONSES_WEBSOCKET: Final[EndpointBinding[RustResponsesWebSocketConnection]] = EndpointBinding.native(
    route="responses_websocket",
    select=lambda native: native.ResponsesWebSocketConnection,
    enabled=rust_enabled,
)


def set_rust_responses_websocket(
    *,
    connection: RustResponsesWebSocketConnection | None | Unchanged = UNCHANGED,
) -> None:
    if not isinstance(connection, Unchanged):
        if connection is None:
            _RESPONSES_WEBSOCKET.reset()
        else:
            _RESPONSES_WEBSOCKET.override(connection)


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
    try:
        connection: Final = await _RESPONSES_WEBSOCKET.ainvoke(
            prepare=lambda: timeout_to_seconds(timeout),
            call=lambda connection_type, timeout_seconds: connection_type.connect(
                url=url,
                headers=headers,
                timeout_seconds=timeout_seconds,
            ),
            fallback=async_none,
            adapt=identity,
            error_context=BridgeErrorContext(provider="openai", model="responses websocket"),
        )
    except Exception:  # noqa: BLE001  # preserve the existing WebSocket connection fallback
        return None
    return None if connection is None else _ConnectionAdapter(connection)
