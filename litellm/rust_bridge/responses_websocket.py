"""Thin Python wrapper for the native Rust Responses WebSocket bridge."""

from __future__ import annotations

from typing import Final, Protocol, cast  # noqa: TID251  # runtime typing constructs

import httpx
from websockets.exceptions import ConnectionClosedOK

from .bindings import UNCHANGED, Unchanged
from .callbacks import SessionCallbackHandle
from .configuration import rust_enabled
from .runtime import (
    AsyncEndpointDispatch,
    BridgeErrorContext,
    async_none,
)  # cast-ok: generic classmethod cannot preserve the route Protocol parameter
from .timeouts import timeout_to_seconds


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
        callback_adapter: SessionCallbackHandle | None,
    ) -> RustResponsesWebSocket: ...


_RESPONSES_WEBSOCKET: Final = cast(  # cast-ok: generic classmethod loses the route Protocol parameter
    AsyncEndpointDispatch[RustResponsesWebSocketConnection],
    AsyncEndpointDispatch.native(
        route="responses_websocket",
        asynchronous="ResponsesWebSocketConnection",
        enabled=rust_enabled,
    ),
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
    eligible: bool = True,
    callback_adapter: SessionCallbackHandle | None = None,
) -> _ConnectionAdapter | None:
    return await _RESPONSES_WEBSOCKET.ainvoke(
        call=lambda connection_type: connection_type.connect(
            url=url,
            headers=headers,
            timeout_seconds=timeout_to_seconds(timeout),
            callback_adapter=callback_adapter,
        ),
        fallback=async_none,
        adapt=_ConnectionAdapter,
        context=BridgeErrorContext(provider="openai", model="responses websocket"),
        request_override=request_override,
        eligible=eligible,
    )
