"""Proxy-owned sideband connections for WebRTC voice sessions.

WebRTC audio and events go from the caller to the provider without passing through the proxy,
so the proxy opens its own sideband WebSocket to the same session and meters it through the
WebSocket session route, with a client that never speaks.
"""

import asyncio
from collections.abc import Mapping
from typing import Final

from starlette.datastructures import URL, Headers
from starlette.websockets import WebSocketDisconnect

_SIDEBAND_TASKS: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: holds each task until it finishes


class SidebandClientWebSocket:
    """Stands in for the caller's socket on a session the proxy only listens to."""

    def __init__(self, path: str) -> None:
        self.scope: Final[Mapping[str, object]] = {"type": "websocket", "path": path, "headers": ()}
        self.headers: Final = Headers()
        self.url: Final = URL(path)
        self._closed: Final = asyncio.Event()

    async def receive_text(self) -> str:
        await self._closed.wait()
        raise WebSocketDisconnect(code=1000)

    async def send_text(self, data: str) -> None:
        return None

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self._closed.set()


def realtime_call_id(location: str | None) -> str | None:
    """Read the call id from the `Location` header of a created Realtime WebRTC call."""
    if not location:
        return None
    call_id: Final = location.rstrip("/").rsplit("/", 1)[-1]
    return call_id if call_id.startswith("rtc_") else None


def keep_sideband_task(task: asyncio.Task[None]) -> None:
    _SIDEBAND_TASKS.add(task)
    task.add_done_callback(_SIDEBAND_TASKS.discard)
