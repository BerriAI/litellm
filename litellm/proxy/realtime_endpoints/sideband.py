"""Proxy-owned sideband connections for WebRTC voice sessions.

WebRTC audio and events go from the caller to the provider without passing through the proxy,
so the proxy opens its own sideband WebSocket to the same session and meters it through the
WebSocket session route, with a client that never speaks.
"""

import asyncio
import json
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from typing import Final, TypeAlias

from starlette.datastructures import URL, Headers
from starlette.websockets import WebSocketDisconnect

from litellm.litellm_core_utils.core_helpers import as_str_mapping

SIDEBAND_ATTACH_TIMEOUT_SECONDS: Final = 10.0
_SIDEBAND_WIND_DOWN_SECONDS: Final = 5.0
_SIDEBAND_REJOIN_DELAY_SECONDS: Final = 1.0
_SIDEBAND_MAX_REJOINS: Final = 10

_SIDEBAND_TASKS: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: holds each task until it finishes
_SIDEBAND_SUPERVISORS: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: holds each task until it finishes


def _event_type(data: str) -> object:
    try:
        event: Final = as_str_mapping(json.loads(data))
    except ValueError:
        return None
    return None if event is None else event.get("type")


def _is_session_event(data: str) -> bool:
    """Whether the frame is an event of the session, and not the route's notice that it could not join."""
    return _event_type(data) not in (None, "error")


class SidebandClientWebSocket:
    """Stands in for the caller's socket on a session the proxy only listens to."""

    def __init__(self, path: str) -> None:
        self.scope: Final[Mapping[str, object]] = {"type": "websocket", "path": path, "headers": ()}
        self.headers: Final = Headers()
        self.url: Final = URL(path)
        self._closed: Final = asyncio.Event()
        self._settled: Final = asyncio.Event()
        self._joined: Final[list[bool]] = []  # mutable-ok: written once, when the first frame or the close arrives
        self._ended: Final = asyncio.Event()

        self._billed: Final[list[float]] = []  # mutable-ok: written once, when the session reports what it billed

    @property
    def refused(self) -> bool:
        """Whether the session ended or was turned away before its relay to the provider ran."""
        return self._joined == [False]

    @property
    def billed_seconds(self) -> float:
        """The seconds of the call this session billed, for a provider that bills by the second."""
        return sum(self._billed)

    def bill_seconds(self, seconds: float) -> None:
        self._billed.append(seconds)

    @property
    def session_ended(self) -> bool:
        """Whether the provider said the session is over, so there is nothing left to rejoin."""
        return self._ended.is_set()

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
        if _event_type(data) == "session.closed":
            self._ended.set()
        self.settle(_is_session_event(data))

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self._closed.set()
        self.settle(False)


SidebandSession: TypeAlias = Callable[[SidebandClientWebSocket, float], Coroutine[object, object, None]]


def realtime_call_id(location: str | None) -> str | None:
    """Read the call id from the `Location` header of a created Realtime WebRTC call."""
    if not location:
        return None
    call_id: Final = location.rstrip("/").rsplit("/", 1)[-1]
    return call_id if call_id.startswith("rtc_") else None


@dataclass(frozen=True, slots=True)
class _JoinedSession:
    client: SidebandClientWebSocket
    task: asyncio.Task[None]


async def _joined_session(
    path: str,
    start: SidebandSession,
    metered_seconds: float,
    timeout: float,
    wind_down: float,
) -> _JoinedSession | None:
    """Run one metering session in the background and hand it back once it joined the call.

    A refused session gets `wind_down` seconds to release its budget hold and parallel slot before it is stopped.
    One that is still on its way to the provider is stopped at once, so it cannot join and bill a call given up on.
    """
    client: Final = SidebandClientWebSocket(path)
    task: Final = asyncio.create_task(start(client, metered_seconds))
    _SIDEBAND_TASKS.add(task)
    task.add_done_callback(_SIDEBAND_TASKS.discard)
    task.add_done_callback(lambda _finished: client.settle(False))
    if await client.joined(timeout):
        return _JoinedSession(client=client, task=task)
    refused: Final = client.refused
    await client.close()
    finished, _ = await asyncio.wait({task}, timeout=wind_down if refused else 0)
    if not finished:
        _ = task.cancel()
    return None


async def _stay_joined(
    path: str,
    start: SidebandSession,
    joined: _JoinedSession,
    metered_seconds: float,
    rejoins_left: int,
    rejoin_delay: float,
) -> None:
    """Rejoin the call each time its metering session drops, until the provider says the call is over.

    The call runs between the caller and the provider, so a dropped sideband does not end it. A provider
    that refuses the rejoin no longer has the call.

    A session is told the seconds the sessions before it billed, since a provider that bills by the second
    reports a running total for the whole call.
    """
    _ = await asyncio.wait({joined.task})
    if joined.client.session_ended or rejoins_left <= 0:
        return
    now_metered_seconds: Final = metered_seconds + joined.client.billed_seconds
    await asyncio.sleep(rejoin_delay)
    rejoined: Final = await _joined_session(
        path, start, now_metered_seconds, SIDEBAND_ATTACH_TIMEOUT_SECONDS, _SIDEBAND_WIND_DOWN_SECONDS
    )
    if rejoined is not None:
        await _stay_joined(path, start, rejoined, now_metered_seconds, rejoins_left - 1, rejoin_delay)


async def join_sideband(
    path: str,
    start: SidebandSession,
    timeout: float = SIDEBAND_ATTACH_TIMEOUT_SECONDS,
    wind_down: float = _SIDEBAND_WIND_DOWN_SECONDS,
    rejoin_delay: float = _SIDEBAND_REJOIN_DELAY_SECONDS,
) -> bool:
    """Meter a call from a session `start` opens on a sideband client, and say whether it joined the call.

    `start` also gets the seconds of the call earlier sessions billed, for a provider whose usage is a
    running total. A session reports its own on the client with `bill_seconds`.

    A call whose sideband did not join would run unmetered, so its creator refuses it. Once joined, the
    sideband is reopened when it drops while the call is still running.
    """
    joined: Final = await _joined_session(path, start, 0.0, timeout, wind_down)
    if joined is None:
        return False
    supervisor: Final = asyncio.create_task(_stay_joined(path, start, joined, 0.0, _SIDEBAND_MAX_REJOINS, rejoin_delay))
    _SIDEBAND_SUPERVISORS.add(supervisor)
    supervisor.add_done_callback(_SIDEBAND_SUPERVISORS.discard)
    return True
