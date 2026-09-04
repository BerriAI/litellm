"""Thin Python wrapper for the native Rust Responses WebSocket bridge."""

from __future__ import annotations

from typing import Final, Protocol

import httpx
from websockets.exceptions import ConnectionClosedOK

from litellm.rust_bridge.bindings import UNCHANGED, Unchanged
from litellm.rust_bridge.configuration import rust_enabled
from litellm.rust_bridge.runtime import (
    BridgeErrorContext,
    RustBridge,
    async_none,
)
from litellm.rust_bridge.timeouts import timeout_to_seconds


class RustResponsesWebSocket(Protocol):
    async def send_text(self, text: str) -> None: ...

    async def recv_text(self) -> str | None: ...

    async def close(self) -> None: ...


class RustResponsesWebSocketConnection(Protocol):
    @classmethod
    async def connect(
        cls,
        url: str,
        headers: dict[str, str],
        timeout_seconds: float | None,
    ) -> RustResponsesWebSocket: ...


_RESPONSES_WEBSOCKET: Final[RustBridge[RustResponsesWebSocketConnection]] = RustBridge.native(
    route="responses websocket",
    attribute="ResponsesWebSocketConnection",
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
    request_override: bool | None = None,
) -> _ConnectionAdapter | None:
    return await _RESPONSES_WEBSOCKET.ainvoke(
        call=lambda connection_type: connection_type.connect(
            url=url,
            headers=headers,
            timeout_seconds=timeout_to_seconds(timeout),
        ),
        fallback=async_none,
        adapt=_ConnectionAdapter,
        context=BridgeErrorContext(provider="openai", model="responses websocket"),
        request_override=request_override,
    )
