"""Proxy-owned sideband connections for WebRTC voice sessions.

WebRTC audio and events go from the caller to the provider without passing through the proxy,
so the proxy opens its own sideband WebSocket to the same session and meters it through the
WebSocket session route, with a client that never speaks.
"""

import asyncio
import json
from collections.abc import Coroutine, Mapping
from typing import Final

from starlette.datastructures import URL, Headers
from starlette.websockets import WebSocketDisconnect

from litellm.litellm_core_utils.core_helpers import as_str_mapping

SIDEBAND_ATTACH_TIMEOUT_SECONDS: Final = 10.0
_SIDEBAND_WIND_DOWN_SECONDS: Final = 5.0

_SIDEBAND_TASKS: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: holds each task until it finishes


def _is_session_event(data: str) -> bool:
    """Whether the frame is an event of the session, and not the route's notice that it could not join."""
    try:
        event: Final = as_str_mapping(json.loads(data))
    except ValueError:
        return False
    return event is not None and event.get("type") != "error"


class SidebandClientWebSocket:
    """Stands in for the caller's socket on a session the proxy only listens to."""

    def __init__(self, path: str) -> None:
        self.scope: Final[Mapping[str, object]] = {"type": "websocket", "path": path, "headers": ()}
        self.headers: Final = Headers()
        self.url: Final = URL(path)
        self._closed: Final = asyncio.Event()
        self._settled: Final = asyncio.Event()
        self._joined: Final[list[bool]] = []  # mutable-ok: written once, when the first frame or the close arrives

    def settle(self, joined: bool) -> None:
        if not self._settled.is_set():
            self._joined.append(joined)
            self._settled.set()

    async def joined(self, timeout: float = SIDEBAND_ATTACH_TIMEOUT_SECONDS) -> bool:
        """Whether the relay to the provider started, so the session's usage will be seen."""
        try:
            await asyncio.wait_for(self._settled.wait(), timeout)
        except asyncio.TimeoutError:
            return False
        return self._joined == [True]

    async def receive_text(self) -> str:
        # The relay reads its client only once the provider accepted the connection.
        self.settle(not self._closed.is_set())
        await self._closed.wait()
        raise WebSocketDisconnect(code=1000)

    async def send_text(self, data: str) -> None:
        self.settle(_is_session_event(data))

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self._closed.set()
        self.settle(False)


def realtime_call_id(location: str | None) -> str | None:
    """Read the call id from the `Location` header of a created Realtime WebRTC call."""
    if not location:
        return None
    call_id: Final = location.rstrip("/").rsplit("/", 1)[-1]
    return call_id if call_id.startswith("rtc_") else None


async def join_sideband(
    client: SidebandClientWebSocket,
    session: Coroutine[object, object, None],
    timeout: float = SIDEBAND_ATTACH_TIMEOUT_SECONDS,
    wind_down: float = _SIDEBAND_WIND_DOWN_SECONDS,
) -> bool:
    """Run the metering session in the background and say whether it joined the call.

    A call whose sideband did not join would run unmetered, so its creator refuses it.
    A refused session gets `wind_down` seconds to release its budget hold and parallel slot before it is stopped.
    """
    task: Final = asyncio.create_task(session)
    _SIDEBAND_TASKS.add(task)
    task.add_done_callback(_SIDEBAND_TASKS.discard)
    task.add_done_callback(lambda _finished: client.settle(False))
    if await client.joined(timeout):
        return True
    await client.close()
    finished, _ = await asyncio.wait({task}, timeout=wind_down)
    if not finished:
        _ = task.cancel()
    return False
