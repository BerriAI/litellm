import itertools
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession
from mcp.client._transport import ReadStream, WriteStream
from mcp.client.session import ElicitationFnT, LoggingFnT, MessageHandlerFnT, SamplingFnT
from mcp.shared.dispatcher import CallOptions, as_request_id
from mcp.shared.exceptions import MCPError
from mcp.shared.inbound import MCP_NAME_HEADER, encode_header_value, mcp_param_headers, x_mcp_header_map
from mcp.shared.message import SessionMessage
from mcp.types import RequestId
from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.mcp_client.client import MCPClient
from litellm.types.mcp_server.events import (
    EVENTS_EXTENSION,
    SUBSCRIPTION_ID_META_KEY,
    EventType,
    JsonObject,
    ListEventsResult,
)

StreamMethod: TypeAlias = Literal[
    "notifications/events/active",
    "notifications/events/event",
    "notifications/events/heartbeat",
    "notifications/events/error",
    "notifications/events/terminated",
]
_STREAM_METHODS: Final[Mapping[str, StreamMethod]] = MappingProxyType(
    {
        "notifications/events/active": "notifications/events/active",
        "notifications/events/event": "notifications/events/event",
        "notifications/events/heartbeat": "notifications/events/heartbeat",
        "notifications/events/error": "notifications/events/error",
        "notifications/events/terminated": "notifications/events/terminated",
    }
)
_FRAME_BUFFER: Final = 256
_JSON_OBJECT: Final = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue])
_stream_ids: Final = itertools.count(1)


@dataclass(frozen=True, slots=True)
class StreamFrame:
    method: StreamMethod
    params: JsonObject


@dataclass(frozen=True, slots=True)
class Overflow:
    cursor: str | None


class StreamRoute:
    def __init__(self) -> None:
        streams: Final = anyio.create_memory_object_stream[StreamFrame | Overflow](_FRAME_BUFFER)
        self.send: MemoryObjectSendStream[StreamFrame | Overflow] = streams[0]
        self.receive: MemoryObjectReceiveStream[StreamFrame | Overflow] = streams[1]
        self.failure: MCPError | None = None
        self._dropped_cursor: str | None = None
        self._dropping = False

    def deliver(self, frame: StreamFrame) -> None:
        if self._dropping and self._flush_overflow():
            self._dropping = False
        try:
            self.send.send_nowait(frame)
        except anyio.WouldBlock:
            self._dropping = True
            cursor: Final = frame.params.get("cursor")
            if isinstance(cursor, str):
                self._dropped_cursor = cursor
        except anyio.ClosedResourceError:
            return

    def _flush_overflow(self) -> bool:
        try:
            self.send.send_nowait(Overflow(cursor=self._dropped_cursor))
        except anyio.WouldBlock:
            return False
        return True

    def finish(self, failure: MCPError | None) -> None:
        self.failure = failure
        if self._dropping:
            self._flush_overflow()
        self.send.close()


class EventsClientSession(ClientSession):
    def __init__(
        self,
        read_stream: ReadStream[SessionMessage | Exception],
        write_stream: WriteStream[SessionMessage],
        read_timeout_seconds: float | None = None,
        message_handler: MessageHandlerFnT | None = None,
        sampling_callback: SamplingFnT | None = None,
        elicitation_callback: ElicitationFnT | None = None,
        logging_callback: LoggingFnT | None = None,
    ) -> None:
        super().__init__(
            read_stream,
            write_stream,
            read_timeout_seconds=read_timeout_seconds,
            message_handler=message_handler,
            sampling_callback=sampling_callback,
            elicitation_callback=elicitation_callback,
            logging_callback=logging_callback,
            extensions={EVENTS_EXTENSION: {}},
        )
        self._event_routes: Final[dict[RequestId, StreamRoute]] = {}  # mutable-ok: live stream demux table

    def _intercept_notification(self, method: str, params: Mapping[str, object] | None) -> bool:
        stream_method: Final = _STREAM_METHODS.get(method)
        meta: Final = params.get("_meta") if params is not None else None
        subscription: Final = as_request_id(meta.get(SUBSCRIPTION_ID_META_KEY)) if isinstance(meta, Mapping) else None
        route: Final = self._event_routes.get(subscription) if subscription is not None else None
        if stream_method is None or route is None or params is None:
            return super()._intercept_notification(method, params)
        try:
            route.deliver(StreamFrame(method=stream_method, params=_JSON_OBJECT.validate_python(params)))
        except ValidationError:
            return True
        return True

    async def events_request(
        self,
        method: str,
        params: JsonObject,
        headers: Mapping[str, str],
        *,
        request_id: RequestId | None = None,
        timeout: float | None = None,
    ) -> Mapping[str, object]:
        data: Final[dict[str, object]] = {
            "method": method,
            "params": dict(params),
        }  # mutable-ok: the SDK stamp writes the envelope into it
        opts: Final[CallOptions] = {"request_id": request_id} if request_id is not None else {}
        self._stamp(data, opts)  # pyright: ignore[reportPrivateUsage]  # same per-request envelope the SDK listen driver uses
        opts.setdefault("headers", {}).update(headers)
        if timeout is not None:
            opts["timeout"] = timeout
        return await self._dispatcher.send_raw_request(method, _JSON_OBJECT.validate_python(data["params"]), opts)  # pyright: ignore[reportPrivateUsage]  # raw send so a stream carries no read timeout, as in the SDK listen driver

    @asynccontextmanager
    async def stream_route(self, request_id: RequestId) -> AsyncIterator[StreamRoute]:
        route: Final = StreamRoute()
        self._event_routes[request_id] = route
        try:
            yield route
        finally:
            self._event_routes.pop(request_id, None)
            route.send.close()
            route.receive.close()


def routing_headers(upstream_name: str, arguments: JsonObject, descriptor: EventType) -> Mapping[str, str]:
    return {
        MCP_NAME_HEADER: encode_header_value(upstream_name),
        **mcp_param_headers(x_mcp_header_map(dict(descriptor.input_schema)), arguments),
    }


async def _run(
    client: MCPClient, operation: Callable[[EventsClientSession], Awaitable[Mapping[str, object]]]
) -> Mapping[str, object]:
    async def typed(session: ClientSession) -> Mapping[str, object]:
        if not isinstance(session, EventsClientSession):
            raise MCPError(code=-32603, message="Events session unavailable")
        return await operation(session)

    client.route_session_type(EventsClientSession)
    return await client.run_with_session(typed, quiet_on_error=True)


async def call(
    client: MCPClient, method: str, params: JsonObject, headers: Mapping[str, str], timeout: float
) -> Mapping[str, object]:
    return await _run(client, lambda session: session.events_request(method, params, headers, timeout=timeout))


async def _pages(session: EventsClientSession, max_pages: int, timeout: float) -> Mapping[str, object]:
    cursor: str | None = None  # rebind-ok: advances through upstream pages
    collected: tuple[EventType, ...] = ()  # rebind-ok: accumulates pages within one session
    for _ in range(max_pages):
        raw = await session.events_request(
            "events/list", {"cursor": cursor} if cursor is not None else {}, {}, timeout=timeout
        )
        page = ListEventsResult.model_validate(raw)
        collected = collected + page.events
        cursor = page.next_cursor
        if not cursor:
            return {"events": [item.model_dump(by_alias=True, mode="json") for item in collected]}
    raise MCPError(code=-32603, message="Upstream events/list exceeds the pagination limit")


async def list_all(client: MCPClient, max_pages: int, timeout: float) -> tuple[EventType, ...]:
    return ListEventsResult.model_validate(
        await _run(client, lambda session: _pages(session, max_pages, timeout))
    ).events


async def _drive(
    session: EventsClientSession,
    params: JsonObject,
    headers: Mapping[str, str],
    consume: Callable[[MemoryObjectReceiveStream[StreamFrame | Overflow]], Awaitable[None]],
) -> Mapping[str, object]:
    request_id: Final = f"events-stream-{next(_stream_ids)}"
    async with session.stream_route(request_id) as route:

        async def request() -> None:
            try:
                await session.events_request("events/stream", params, headers, request_id=request_id)
            except MCPError as error:
                route.finish(error)
                return
            route.finish(None)

        async with anyio.create_task_group() as group:
            group.start_soon(request)
            await consume(route.receive)
            group.cancel_scope.cancel()
        if route.failure is not None:
            raise route.failure
    return {}


async def stream(
    client: MCPClient,
    params: JsonObject,
    headers: Mapping[str, str],
    consume: Callable[[MemoryObjectReceiveStream[StreamFrame | Overflow]], Awaitable[None]],
) -> None:
    await _run(client, lambda session: _drive(session, params, headers, consume))
