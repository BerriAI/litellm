from collections.abc import Awaitable, Callable, Mapping
from typing import Final

import anyio
from mcp import ClientSession
from mcp.client._transport import ReadStream, WriteStream
from mcp.client.session import ElicitationFnT, LoggingFnT, MessageHandlerFnT, SamplingFnT
from mcp.shared.dispatcher import CallOptions
from mcp.shared.exceptions import MCPError
from mcp.shared.inbound import MCP_NAME_HEADER, encode_header_value, mcp_param_headers, x_mcp_header_map
from mcp.shared.message import SessionMessage
from mcp.types import RequestId
from pydantic import JsonValue, TypeAdapter

from litellm.mcp_client.client import MCPClient
from litellm.types.mcp_server.events import (
    EVENTS_EXTENSION,
    EventType,
    JsonObject,
    ListEventsResult,
)

_JSON_OBJECT: Final = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue])


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

    async def events_request(
        self,
        method: str,
        params: JsonObject,
        headers: Mapping[str, str],
        *,
        request_id: RequestId | None = None,
        timeout: float | None = None,
    ) -> Mapping[str, object]:
        capabilities: Final = self.server_capabilities
        if capabilities is None or EVENTS_EXTENSION not in (capabilities.extensions or {}):
            raise MCPError(code=-32601, message="Upstream does not advertise Events")
        data: Final[dict[str, object]] = {"method": method, "params": dict(params)}  # mutable-ok: SDK stamp writes it
        opts: Final[CallOptions] = {"request_id": request_id} if request_id is not None else {}
        self._stamp(data, opts)  # pyright: ignore[reportPrivateUsage]  # same per-request envelope the SDK listen driver uses
        opts.setdefault("headers", {}).update(headers)
        if timeout is not None:
            opts["timeout"] = timeout
        return await self._dispatcher.send_raw_request(method, _JSON_OBJECT.validate_python(data["params"]), opts)  # pyright: ignore[reportPrivateUsage]  # raw send so a stream carries no read timeout, as in the SDK listen driver


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

    return await client.run_with_session(typed, quiet_on_error=True, session_type=EventsClientSession)


async def call(
    client: MCPClient, method: str, params: JsonObject, headers: Mapping[str, str], timeout: float
) -> Mapping[str, object]:
    return await _run(client, lambda session: session.events_request(method, params, headers, timeout=timeout))


async def _pages(
    session: EventsClientSession,
    max_pages: int,
    timeout: float,
) -> Mapping[str, object]:
    cursor: str | None = None  # rebind-ok: upstream supplies the next token after each awaited page
    events: tuple[EventType, ...] = ()  # rebind-ok: accumulate bounded pages without recursive stack growth
    for _ in range(max_pages):
        raw: Mapping[str, object] = await session.events_request(
            "events/list", {"cursor": cursor} if cursor is not None else {}, {}, timeout=timeout
        )
        page: ListEventsResult = ListEventsResult.model_validate(raw)
        events = events + page.events
        if page.next_cursor is None:
            return {"events": [item.model_dump(by_alias=True, mode="json") for item in events]}
        cursor = page.next_cursor
    raise MCPError(code=-32603, message="Upstream events/list exceeds the pagination limit")


async def list_all(client: MCPClient, max_pages: int, timeout: float) -> tuple[EventType, ...]:
    with anyio.fail_after(timeout):
        return ListEventsResult.model_validate(
            await _run(client, lambda session: _pages(session, max_pages, timeout))
        ).events
