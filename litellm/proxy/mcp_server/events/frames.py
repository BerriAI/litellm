from collections import deque
from dataclasses import dataclass
from typing import Final, Literal, Protocol, TypeAlias

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.shared.exceptions import MCPError

from litellm.types.mcp_server.events import (
    JsonObject,
)

StreamMethod: TypeAlias = Literal[
    "notifications/events/active",
    "notifications/events/event",
    "notifications/events/heartbeat",
    "notifications/events/error",
    "notifications/events/terminated",
]
_FRAME_BUFFER: Final = 256
_HELD_LIMIT: Final = 32


@dataclass(frozen=True, slots=True)
class StreamFrame:
    method: StreamMethod
    params: JsonObject


_DROPPABLE: Final = frozenset({"notifications/events/event", "notifications/events/heartbeat"})


@dataclass(frozen=True, slots=True)
class Overflow:
    cursor: str | None


class Frames(Protocol):
    async def receive(self) -> StreamFrame | Overflow: ...


_FELL_BEHIND: Final = StreamFrame(
    "notifications/events/terminated",
    {"error": {"code": -32603, "message": "The event stream fell too far behind and was ended; resubscribe"}},
)


class StreamRoute:
    """A bounded buffer between the upstream stream and the relay. Under backpressure, event and heartbeat
    frames are dropped and reported as one overflow notice; error and terminated frames are held and delivered,
    in order, after it. A reader that falls so far behind that the held frames reach their limit is terminated."""

    def __init__(self) -> None:
        streams: Final = anyio.create_memory_object_stream[StreamFrame | Overflow](_FRAME_BUFFER)
        self.send: MemoryObjectSendStream[StreamFrame | Overflow] = streams[0]
        self.receive: MemoryObjectReceiveStream[StreamFrame | Overflow] = streams[1]
        self.failure: MCPError | None = None
        self._dropped_cursor: str | None = None
        self._dropping = False
        self._held: deque[StreamFrame] = deque()  # mutable-ok: control frames waiting for buffer space
        self._overrun = False

    def deliver(self, frame: StreamFrame) -> None:
        if self._overrun:
            return
        if not self._flush():
            self._defer(frame)
            return
        try:
            self.send.send_nowait(frame)
        except anyio.WouldBlock:
            self._defer(frame)
        except anyio.ClosedResourceError:
            return

    def _defer(self, frame: StreamFrame) -> None:
        if frame.method in _DROPPABLE:
            self._dropping = True
            cursor: Final = frame.params.get("cursor")
            if isinstance(cursor, str):
                self._dropped_cursor = cursor
        elif len(self._held) < _HELD_LIMIT:
            self._held.append(frame)
        else:
            self._overrun = True
            self._held.append(_FELL_BEHIND)

    def _flush(self) -> bool:
        """Send the overflow notice, then held control frames, while there is room; True once none remain."""
        try:
            if self._dropping:
                self.send.send_nowait(Overflow(cursor=self._dropped_cursor))
                self._dropping = False
            while self._held:
                self.send.send_nowait(self._held[0])
                self._held.popleft()
        except (anyio.WouldBlock, anyio.ClosedResourceError):
            return False
        return True

    def finish(self, failure: MCPError | None) -> None:
        self.failure = failure
        self._flush()
        self.send.close()

    def reader(self) -> "RouteReader":
        return RouteReader(self)

    async def next_frame(self) -> StreamFrame | Overflow:
        """The next frame. Reading makes room, so a pending notice is released without waiting for the upstream;
        after the last buffered frame come any notices the full buffer could not take."""
        self._flush()
        try:
            return await self.receive.receive()
        except anyio.EndOfStream:
            if self._dropping:
                self._dropping = False
                return Overflow(cursor=self._dropped_cursor)
            if self._held:
                return self._held.popleft()
            raise


@dataclass(frozen=True, slots=True)
class RouteReader:
    route: StreamRoute

    async def receive(self) -> StreamFrame | Overflow:
        return await self.route.next_frame()

    def __aiter__(self) -> "RouteReader":
        return self

    async def __anext__(self) -> StreamFrame | Overflow:
        try:
            return await self.route.next_frame()
        except anyio.EndOfStream:
            raise StopAsyncIteration from None
