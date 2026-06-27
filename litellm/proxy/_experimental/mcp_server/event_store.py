"""In-memory EventStore enabling Streamable HTTP resumability for the MCP gateway.

Wired into the stateful ``StreamableHTTPSessionManager`` so the SDK transport
tags SSE events with ids and replays missed events when a client reconnects
with ``Last-Event-ID`` (MCP Streamable HTTP transport, resumability section).

Storage is per-worker, which matches the constraint that the live session
object itself only exists on the worker that created it.
"""

from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional
from uuid import uuid4

from mcp.server.streamable_http import (
    EventCallback,
    EventId,
    EventMessage,
    EventStore,
    StreamId,
)
from mcp.types import JSONRPCMessage

from litellm.constants import (
    MCP_EVENT_STORE_MAX_EVENTS_PER_STREAM,
    MCP_EVENT_STORE_MAX_STREAMS,
)


@dataclass
class _StoredEvent:
    event_id: EventId
    message: Optional[JSONRPCMessage]


class InMemoryMCPEventStore(EventStore):
    """Bounded in-memory event store.

    Streams are evicted least-recently-used once ``max_streams`` is reached,
    and each stream keeps at most ``max_events_per_stream`` events, so a
    long-lived session cannot grow worker memory without bound.
    """

    def __init__(
        self,
        max_streams: int = MCP_EVENT_STORE_MAX_STREAMS,
        max_events_per_stream: int = MCP_EVENT_STORE_MAX_EVENTS_PER_STREAM,
    ) -> None:
        self._max_streams = max_streams
        self._max_events_per_stream = max_events_per_stream
        self._streams: "OrderedDict[StreamId, Deque[_StoredEvent]]" = OrderedDict()
        self._event_to_stream: Dict[EventId, StreamId] = {}

    async def store_event(
        self, stream_id: StreamId, message: Optional[JSONRPCMessage]
    ) -> EventId:
        event_id: EventId = uuid4().hex

        stream = self._streams.get(stream_id)
        if stream is None:
            while len(self._streams) >= self._max_streams:
                _, evicted = self._streams.popitem(last=False)
                for event in evicted:
                    self._event_to_stream.pop(event.event_id, None)
            stream = deque()
            self._streams[stream_id] = stream
        else:
            self._streams.move_to_end(stream_id)

        while len(stream) >= self._max_events_per_stream:
            oldest = stream.popleft()
            self._event_to_stream.pop(oldest.event_id, None)

        stream.append(_StoredEvent(event_id=event_id, message=message))
        self._event_to_stream[event_id] = stream_id
        return event_id

    async def replay_events_after(
        self,
        last_event_id: EventId,
        send_callback: EventCallback,
    ) -> Optional[StreamId]:
        stream_id = self._event_to_stream.get(last_event_id)
        if stream_id is None:
            return None

        found_last = False
        for event in list(self._streams.get(stream_id, ())):
            if found_last:
                if event.message is not None:
                    await send_callback(
                        EventMessage(message=event.message, event_id=event.event_id)
                    )
            elif event.event_id == last_event_id:
                found_last = True

        return stream_id
