from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol

import httpx
from pydantic import TypeAdapter
from websockets.exceptions import ConnectionClosedOK

from litellm.rust_bridge.loader import get_native_bridge
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


class _Unset:
    pass


_UNSET: Final = _Unset()


@dataclass(slots=True)
class _RustResponsesWebSocketState:
    connection: type[RustResponsesWebSocketConnection] | None = None


_STATE: Final = _RustResponsesWebSocketState()


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
    connection_type: Final[type[RustResponsesWebSocketConnection] | None] = getattr(
        native_bridge, "ResponsesWebSocketSession", None
    )
    return connection_type


class _ConnectionAdapter:
    def __init__(self, connection: RustResponsesWebSocket):
        self._connection: Final = connection

    async def send(self, text: str) -> None:
        event: Final = _EVENT_ADAPTER.validate_json(text)
        await self._connection.send_event(event)

    async def recv(self) -> str:
        event: Final = await self._connection.recv_event()
        if event is None:
            raise ConnectionClosedOK(None, None)
        return json.dumps(dict(event), separators=(",", ":"))  # mutable-ok: JSON requires a concrete dict

    async def close(self) -> None:
        await self._connection.close()


async def connect(
    *,
    provider: str,
    api_key: str | None,
    api_base: str | None,
    headers: Mapping[str, str],
    timeout: float | httpx.Timeout | None,
) -> _ConnectionAdapter | None:
    connection_type: Final = load_rust_responses_websocket()
    if connection_type is None:
        return None
    credentials: Final = None if api_key is None else MappingProxyType({"api_key": api_key})
    try:
        connection: Final = await connection_type.connect(
            provider,
            credentials,
            api_base,
            headers,
            timeout_to_seconds(timeout),
        )
    except Exception as error:
        native: Final = get_native_bridge()
        declined: Final = None if native is None else getattr(native, "RustBridgeDeclined", None)
        if isinstance(declined, type) and isinstance(error, declined):
            return None
        raise
    return _ConnectionAdapter(connection)
