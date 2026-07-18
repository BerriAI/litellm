"""Thin Python wrapper for the native Rust Responses WebSocket bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Protocol

import httpx

from litellm.rust_bridge.loader import get_native_bridge
from litellm.rust_bridge.timeouts import timeout_to_seconds


class RustResponsesWebSocketConnection(Protocol):
    @classmethod
    def connect(
        cls,
        url: str,
        headers: dict[str, str],
        timeout_seconds: float | None,
    ) -> Any:
        raise NotImplementedError


class _Unset:
    pass


_UNSET: Final[_Unset] = _Unset()


@dataclass(slots=True)
class _RustResponsesWebSocketState:
    connection: Any = None


_STATE: Final[_RustResponsesWebSocketState] = _RustResponsesWebSocketState()


def set_rust_responses_websocket(
    *,
    connection: Any = _UNSET,
) -> None:
    if not isinstance(connection, _Unset):
        _STATE.connection = connection


def load_rust_responses_websocket() -> Any:
    if _STATE.connection is not None:
        return _STATE.connection
    native_bridge = get_native_bridge()
    if native_bridge is None:
        return None
    return getattr(native_bridge, "ResponsesWebSocketConnection", None)


class _ConnectionAdapter:
    def __init__(self, connection: Any):
        self._connection = connection

    async def send(self, text: str) -> None:
        await self._connection.send_text(text)

    async def recv(self) -> str | None:
        return await self._connection.recv_text()

    async def close(self) -> None:
        await self._connection.close()


async def connect(
    *,
    url: str,
    headers: dict[str, str],
    timeout: float | httpx.Timeout | None,
) -> _ConnectionAdapter | None:
    connection_type = load_rust_responses_websocket()
    if connection_type is None:
        return None
    try:
        connection = await connection_type.connect(
            url=url,
            headers=headers,
            timeout_seconds=timeout_to_seconds(timeout),
        )
    except Exception:  # noqa: BLE001  # bridge failures must fall back to Python
        return None
    return _ConnectionAdapter(connection)
