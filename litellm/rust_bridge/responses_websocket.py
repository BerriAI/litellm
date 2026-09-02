"""Thin Python wrapper for the native Rust Responses WebSocket bridge."""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Protocol

import httpx
from pydantic import TypeAdapter
from websockets.exceptions import ConnectionClosedOK

from litellm.rust_bridge import streaming
from litellm.rust_bridge.bindings import UNSET, NativeBinding, Unset
from litellm.rust_bridge.runtime import (
    BridgeErrorContext,
    CoreEngine,
    ExecutionResult,
    FallbackMode,
    acall,
    ainvoke,
    async_none,
)
from litellm.rust_bridge.timeouts import timeout_to_seconds

_EVENT_ADAPTER: Final = TypeAdapter(Mapping[str, object])


class RustResponsesWebSocket(Protocol):
    async def send_event(self, event: Mapping[str, object]) -> None: ...

    async def recv_event(self) -> Mapping[str, object] | None: ...

    async def close(self) -> None: ...


class RustResponsesWebSocketConnection(Protocol):
    @classmethod
    async def connect(
        cls,
        provider: str,
        credentials: Mapping[str, str] | None,
        api_base: str | None,
        extra_headers: Mapping[str, str] | None,
        timeout_seconds: float | None,
    ) -> RustResponsesWebSocket: ...


_CONNECTION: Final = NativeBinding[type[RustResponsesWebSocketConnection]]("ResponsesWebSocketSession")


def set_rust_responses_websocket(
    *,
    connection: type[RustResponsesWebSocketConnection] | None | Unset = UNSET,
) -> None:
    _CONNECTION.update(connection)


def load_rust_responses_websocket() -> type[RustResponsesWebSocketConnection] | None:
    return _CONNECTION.load()


class _ConnectionAdapter:
    def __init__(self, connection: RustResponsesWebSocket, context: BridgeErrorContext):
        self._connection: Final = connection
        self._context: Final = context
        self.core_engine: Final = CoreEngine.RUST

    async def send(self, text: str) -> None:
        event: Final = _EVENT_ADAPTER.validate_json(text)
        await acall(
            lambda: self._connection.send_event(event),
            self._context,
        )

    async def recv(self) -> str:
        event: Final = await acall(
            self._connection.recv_event,
            self._context,
        )
        if event is None:
            raise ConnectionClosedOK(None, None)
        return json.dumps(dict(event), separators=(",", ":"))  # mutable-ok: JSON needs a concrete dict

    async def close(self) -> None:
        await acall(
            self._connection.close,
            self._context,
        )


async def connect(
    *,
    provider: str,
    api_key: str | None,
    api_base: str | None,
    headers: Mapping[str, str],
    timeout: float | httpx.Timeout | None,
) -> ExecutionResult[_ConnectionAdapter | None]:
    context: Final = BridgeErrorContext(route="responses websocket", provider=provider, model="")
    if not streaming.supports_streaming("responses", provider, "websocket"):
        return ExecutionResult(value=None, source=CoreEngine.PYTHON)
    connection_type: Final = load_rust_responses_websocket()
    credentials: Final = None if api_key is None else MappingProxyType({"api_key": api_key})
    native_call: Final = (
        None
        if connection_type is None
        else lambda: connection_type.connect(
            provider,
            credentials,
            api_base,
            headers,
            timeout_to_seconds(timeout),
        )
    )
    return await ainvoke(
        native_call=native_call,
        fallback=async_none,
        adapt=lambda connection: _ConnectionAdapter(connection, context),
        mode=FallbackMode.PYTHON,
        context=context,
    )
