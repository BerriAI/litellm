"""Thin Python wrapper for the native Rust Responses WebSocket bridge."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Final

import httpx
from websockets.exceptions import ConnectionClosedOK

from litellm.rust_bridge.bindings import UNCHANGED, NativeBinding, Unchanged
from litellm.rust_bridge.configuration import rust_enabled
from litellm.rust_bridge.protocols import (
    RustResponsesWebSocket,
    RustResponsesWebSocketConnection,
)
from litellm.rust_bridge.runtime import DispatchResult, aattempt, adapt_result
from litellm.rust_bridge.timeouts import timeout_to_seconds

_RESPONSES_WEBSOCKET: Final[NativeBinding[RustResponsesWebSocketConnection]] = NativeBinding(
    lambda native: native.ResponsesWebSocketConnection,
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


class ConnectionAdapter:
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
) -> DispatchResult[ConnectionAdapter]:
    return await aattempt(
        load=_RESPONSES_WEBSOCKET.load,
        enabled=rust_enabled(),
        eligible=True,
        prepare=lambda: timeout_to_seconds(timeout),
        call=lambda connection_type, timeout_seconds: connection_type.connect(
            url=url,
            headers=headers,
            timeout_seconds=timeout_seconds,
        ),
        adapt=ConnectionAdapter,
    )


@asynccontextmanager
async def _connection_context(connection: ConnectionAdapter) -> AsyncGenerator[ConnectionAdapter, None]:
    try:
        yield connection
    finally:
        await connection.close()


async def managed_connect(
    *,
    url: str,
    headers: dict[str, str],
    timeout: float | httpx.Timeout | None,
) -> DispatchResult[AbstractAsyncContextManager[ConnectionAdapter]]:
    result: Final = await connect(url=url, headers=headers, timeout=timeout)
    return adapt_result(result, _connection_context)
