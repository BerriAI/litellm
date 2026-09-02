"""Thin Python wrapper for the native Rust Responses WebSocket bridge."""

from __future__ import annotations

from typing import Final, Protocol

import httpx
from websockets.exceptions import ConnectionClosedOK

from litellm.rust_bridge.bindings import UNSET, NativeBinding, Unset
from litellm.rust_bridge.runtime import (
    BridgeErrorContext,
    FallbackMode,
    acall,
    ainvoke,
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


_CONNECTION: Final = NativeBinding[type[RustResponsesWebSocketConnection]](
    "ResponsesWebSocketConnection"
)


def set_rust_responses_websocket(
    *,
    connection: type[RustResponsesWebSocketConnection] | None | Unset = UNSET,
) -> None:
    _CONNECTION.update(connection)


def load_rust_responses_websocket() -> type[RustResponsesWebSocketConnection] | None:
    return _CONNECTION.load()


class _ConnectionAdapter:
    def __init__(self, connection: RustResponsesWebSocket):
        self._connection: Final = connection

    async def send(self, text: str) -> None:
        await acall(
            lambda: self._connection.send_text(text),
            BridgeErrorContext(route="responses websocket", provider="openai", model=""),
        )

    async def recv(self) -> str:
        message: Final = await acall(
            self._connection.recv_text,
            BridgeErrorContext(route="responses websocket", provider="openai", model=""),
        )
        if message is None:
            raise ConnectionClosedOK(None, None)
        return message

    async def close(self) -> None:
        await acall(
            self._connection.close,
            BridgeErrorContext(route="responses websocket", provider="openai", model=""),
        )


async def connect(
    *,
    url: str,
    headers: dict[str, str],
    timeout: float | httpx.Timeout | None,
) -> _ConnectionAdapter | None:
    connection_type: Final = load_rust_responses_websocket()
    native_call: Final = (
        None
        if connection_type is None
        else lambda: connection_type.connect(
            url=url,
            headers=headers,
            timeout_seconds=timeout_to_seconds(timeout),
        )
    )
    return await ainvoke(
        native_call=native_call,
        fallback=async_none,
        adapt=_ConnectionAdapter,
        mode=FallbackMode.PYTHON,
        context=BridgeErrorContext(route="responses websocket", provider="openai", model=""),
    )
