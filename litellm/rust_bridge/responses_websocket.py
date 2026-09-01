from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Final, Protocol

import httpx
from pydantic import TypeAdapter
from websockets.exceptions import ConnectionClosedOK

from litellm.rust_bridge.runtime import (
    UNSET,
    BridgeErrorContext,
    CoreEngine,
    ExecutionResult,
    FallbackMode,
    NativeBinding,
    Unset,
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
    def __init__(self, connection: RustResponsesWebSocket):
        self._connection: Final = connection
        self.core_engine: Final = CoreEngine.RUST

    async def send(self, text: str) -> None:
        event: Final = _EVENT_ADAPTER.validate_json(text)
        await acall(
            lambda: self._connection.send_event(event),
            BridgeErrorContext(route="responses websocket", provider="openai", model=""),
        )

    async def recv(self) -> str:
        event: Final = await acall(
            self._connection.recv_event,
            BridgeErrorContext(route="responses websocket", provider="openai", model=""),
        )
        if event is None:
            raise ConnectionClosedOK(None, None)
        return json.dumps(dict(event), separators=(",", ":"))  # mutable-ok: JSON requires a concrete dict

    async def close(self) -> None:
        await acall(
            self._connection.close,
            BridgeErrorContext(route="responses websocket", provider="openai", model=""),
        )


async def connect(
    *,
    provider: str,
    api_key: str | None,
    api_base: str | None,
    headers: Mapping[str, str],
    timeout: float | httpx.Timeout | None,
) -> ExecutionResult[_ConnectionAdapter | None]:
    connection_type: Final = load_rust_responses_websocket()
    credentials: Final = (
        None
        if api_key is None
        else {"api_key": api_key}  # mutable-ok: native bridge serialization requires a plain dict
    )
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
        adapt=_ConnectionAdapter,
        mode=FallbackMode.PYTHON,
        context=BridgeErrorContext(route="responses websocket", provider=provider, model=""),
    )
