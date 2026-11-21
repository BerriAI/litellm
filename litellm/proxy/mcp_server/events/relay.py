import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Final

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream
from mcp.shared.inbound import MCP_NAME_HEADER, decode_header_value, validate_mcp_param_headers
from pydantic import ValidationError

from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.events.access import event_grant, grant_allows, principal_of
from litellm.proxy.mcp_server.events.catalog import PreparedEvent, allowed_servers, offered_modes, prepare_event
from litellm.proxy.mcp_server.events.config import EventsSettings, server_allows_event
from litellm.proxy.mcp_server.events.cursors import CursorBinding, arguments_digest, open_cursor, seal_cursor
from litellm.proxy.mcp_server.events.errors import (
    EventsFailure,
    exhausted,
    forbidden,
    header_mismatch,
    unsupported,
)
from litellm.proxy.mcp_server.events.guardrails import Blocked, screen_event
from litellm.proxy.mcp_server.events.observability import EventActivity, record
from litellm.proxy.mcp_server.events.upstream import Overflow, StreamFrame, call, routing_headers, stream
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.types.mcp_server.events import (
    SUBSCRIPTION_ID_META_KEY,
    ActiveNotification,
    DeliveryMode,
    EventOccurrence,
    EventsErrorCode,
    HeartbeatNotification,
    JsonObject,
    PollEventsParams,
    PollEventsResult,
    StreamEventsParams,
)

_DEFAULT_HEARTBEAT_MS: Final = 30_000
_MIN_HEARTBEAT_MS: Final = 5_000
_MAX_HEARTBEAT_MS: Final = 60_000
_RECHECK_SECONDS: Final = 15.0
_BUDGET_WINDOW_SECONDS: Final = 60.0
_BUDGET_CALLERS: Final = 10_000

SendNotification = Callable[[str, JsonObject], Awaitable[None]]
Authorize = Callable[[], Awaitable[OperationContext]]


class EventBudget:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._windows: OrderedDict[str, tuple[float, int]] = OrderedDict()

    def take(self, caller: str, limit: int) -> bool:
        now: Final = self._clock()
        window: Final = self._windows.get(caller)
        start, used = window if window is not None and now - window[0] < _BUDGET_WINDOW_SECONDS else (now, 0)
        if used >= limit:
            return False
        self._windows[caller] = (start, used + 1)
        self._windows.move_to_end(caller)
        while len(self._windows) > _BUDGET_CALLERS:
            self._windows.popitem(last=False)
        return True


class StreamSlots:
    def __init__(self) -> None:
        self._per_caller: dict[str, int] = {}  # mutable-ok: live stream counts per caller on this replica
        self._total = 0

    @asynccontextmanager
    async def hold(self, caller: str, settings: EventsSettings) -> AsyncIterator[None]:
        if self._total >= settings.max_streams:
            raise exhausted("streams", settings.max_streams).to_mcp()
        if self._per_caller.get(caller, 0) >= settings.max_streams_per_caller:
            raise exhausted("streams", settings.max_streams_per_caller).to_mcp()
        self._total += 1
        self._per_caller[caller] = self._per_caller.get(caller, 0) + 1
        try:
            yield
        finally:
            self._total -= 1
            remaining: Final = self._per_caller.get(caller, 1) - 1
            if remaining:
                self._per_caller[caller] = remaining
            else:
                self._per_caller.pop(caller, None)


event_budget: Final = EventBudget()
stream_slots: Final = StreamSlots()


def check_routing_headers(
    headers: Mapping[str, str] | None, name: str, arguments: JsonObject, prepared: PreparedEvent
) -> EventsFailure | None:
    if headers is None:
        return None
    if decode_header_value(headers.get(MCP_NAME_HEADER)) != name:
        return header_mismatch("Mcp-Name")
    rejection: Final = validate_mcp_param_headers(dict(prepared.descriptor.input_schema), arguments, headers)
    return header_mismatch(rejection.message) if rejection is not None else None


def caller_key(context: OperationContext) -> str:
    return principal_of(context.user_api_key_auth) or "anonymous"


async def admit(
    context: OperationContext,
    name: str,
    arguments: JsonObject,
    mode: DeliveryMode,
    headers: Mapping[str, str] | None,
    settings: EventsSettings,
) -> Result[PreparedEvent, EventsFailure]:
    from litellm.proxy.mcp_server import operations

    prepared: Final = await prepare_event(context, name)
    if isinstance(prepared, Error):
        return prepared
    mismatch: Final = check_routing_headers(headers, name, arguments, prepared.ok)
    if mismatch is not None:
        return Error(mismatch)
    if mode not in offered_modes(prepared.ok.descriptor, settings, context):
        return Error(unsupported("deliveryMode", mode))
    rejection: Final = await operations._mcp_server_rate_limit_rejection(  # pyright: ignore[reportPrivateUsage]  # same per-server admission as tools/call
        prepared.ok.resolved.target.server, context.user_api_key_auth
    )
    if rejection is not None:
        return Error(exhausted("rate", prepared.ok.resolved.target.server.rpm or 0))
    return prepared


def binding_for(context: OperationContext, prepared: PreparedEvent, arguments: JsonObject) -> CursorBinding:
    return CursorBinding(
        server_id=prepared.resolved.target.server.server_id,
        event=prepared.resolved.upstream_name,
        args_hash=arguments_digest(arguments),
        caller=caller_key(context),
    )


async def still_authorized(context: OperationContext, prepared: PreparedEvent) -> bool:
    server: Final = prepared.resolved.target.server
    if not any(item.server_id == server.server_id for item in await allowed_servers(context)):
        return False
    grant: Final = await event_grant(server.server_id, context.user_api_key_auth)
    return server_allows_event(prepared.resolved.target.config, prepared.resolved.upstream_name) and grant_allows(
        grant, prepared.resolved.upstream_name
    )


@dataclass(frozen=True, slots=True)
class Admitted:
    events: tuple[EventOccurrence, ...]
    dropped: int


async def _screen_all(
    occurrences: tuple[EventOccurrence, ...],
    context: OperationContext,
    prepared: PreparedEvent,
    settings: EventsSettings,
    subscription_id: str | None,
) -> Admitted:
    async def one(occurrence: EventOccurrence) -> EventOccurrence | None:
        if not event_budget.take(caller_key(context), settings.max_events_per_minute):
            return None
        outcome = await screen_event(
            occurrence,
            server_name=prepared.resolved.target.server.name,
            gateway_name=prepared.resolved.gateway_name,
            subscription_id=subscription_id,
            caller=context.user_api_key_auth,
        )
        if isinstance(outcome, Blocked):
            return None
        return occurrence.model_copy(
            update={"name": prepared.resolved.gateway_name, "data": outcome.data, "cursor": None, "meta": None}
        )

    screened: Final = tuple([await one(occurrence) for occurrence in occurrences])
    kept: Final = tuple(item for item in screened if item is not None)
    return Admitted(events=kept, dropped=len(screened) - len(kept))


async def poll(
    context: OperationContext,
    params: PollEventsParams,
    headers: Mapping[str, str] | None,
    settings: EventsSettings,
) -> Result[PollEventsResult, EventsFailure]:
    prepared: Final = await admit(context, params.name, params.arguments, "poll", headers, settings)
    if isinstance(prepared, Error):
        return prepared
    binding: Final = binding_for(context, prepared.ok, params.arguments)
    opened: Final = open_cursor(params.cursor, binding, int(time.time()))
    if isinstance(opened, Error):
        return opened
    upstream_params: Final = PollEventsParams(
        name=prepared.ok.resolved.upstream_name,
        arguments=params.arguments,
        cursor=opened.ok.upstream,
        max_age_ms=params.max_age_ms,
        max_events=params.max_events,
    )
    try:
        raw: Final = await call(
            prepared.ok.upstream.client,
            "events/poll",
            upstream_params,
            routing_headers(prepared.ok.resolved.upstream_name, params.arguments, prepared.ok.descriptor),
        )
        upstream_result: Final = PollEventsResult.model_validate(raw)
    except ValidationError:
        return Error(EventsFailure(EventsErrorCode.INTERNAL_ERROR, "Upstream returned an invalid events/poll result"))
    admitted: Final = await _screen_all(upstream_result.events, context, prepared.ok, settings, None)
    sealed: Final = seal_cursor(binding, upstream_result.cursor, int(time.time()))
    if isinstance(sealed, Error):
        return sealed
    if admitted.events or admitted.dropped:
        await record(
            EventActivity(
                call_type="mcp_event_poll",
                mode="poll",
                server_id=prepared.ok.resolved.target.server.server_id,
                event_name=prepared.ok.resolved.gateway_name,
                outcome="throttled" if admitted.dropped else "delivered",
                caller=context.user_api_key_auth,
                event_ids=tuple(item.event_id for item in admitted.events),
            )
        )
    return Ok(
        PollEventsResult(
            events=admitted.events,
            cursor=sealed.ok,
            truncated=upstream_result.truncated or opened.ok.expired or admitted.dropped > 0,
            has_more=upstream_result.has_more,
            next_poll_ms=max(
                upstream_result.next_poll_ms or 0, prepared.ok.resolved.target.config.min_poll_interval_ms
            ),
        )
    )


def granted_heartbeat(requested: int | None) -> int:
    return min(max(requested or _DEFAULT_HEARTBEAT_MS, _MIN_HEARTBEAT_MS), _MAX_HEARTBEAT_MS)


class _Idle:
    pass


_IDLE: Final = _Idle()


class _PushRelay:
    def __init__(
        self,
        send: SendNotification,
        subscription_id: str | int,
        context: OperationContext,
        prepared: PreparedEvent,
        binding: CursorBinding,
        settings: EventsSettings,
        authorize: Authorize,
        heartbeat_ms: int,
        expired: bool,
    ) -> None:
        self._send = send
        self._meta: Final[JsonObject] = {SUBSCRIPTION_ID_META_KEY: subscription_id}
        self._context = context
        self._prepared = prepared
        self._binding = binding
        self._settings = settings
        self._authorize = authorize
        self._heartbeat_ms = heartbeat_ms
        self._expired_on_open = expired
        self._cursor: str | None = None
        self._ended = False
        self._next_check = time.monotonic() + _RECHECK_SECONDS

    async def _emit(self, method: str, params: JsonObject) -> None:
        await self._send(method, {**params, "_meta": self._meta})

    def _seal(self, upstream: object) -> str | None:
        if not isinstance(upstream, str):
            return None
        sealed: Final = seal_cursor(self._binding, upstream, int(time.time()))
        return sealed.ok if isinstance(sealed, Ok) else None

    async def _terminate(self, failure: EventsFailure) -> None:
        self._ended = True
        await self._emit(
            "notifications/events/terminated",
            {"error": {"code": int(failure.code), "message": failure.message, "data": dict(failure.data or {})}},
        )
        await record(
            EventActivity(
                call_type="mcp_event_stream",
                mode="push",
                server_id=self._prepared.resolved.target.server.server_id,
                event_name=self._prepared.resolved.gateway_name,
                outcome="terminated",
                caller=self._context.user_api_key_auth,
                subscription_id=str(self._meta[SUBSCRIPTION_ID_META_KEY]),
            )
        )

    async def _authorized(self) -> bool:
        try:
            fresh: Final = await self._authorize()
        except Exception:  # noqa: BLE001  # any failure to re-authorize ends the subscription
            return False
        return caller_key(fresh) == self._binding.caller and await still_authorized(fresh, self._prepared)

    async def _truncate(self, upstream_cursor: object) -> None:
        self._cursor = self._seal(upstream_cursor) or self._cursor
        await self._emit(
            "notifications/events/active",
            ActiveNotification(cursor=self._cursor, truncated=True, heartbeat_ms=self._heartbeat_ms).model_dump(
                by_alias=True, mode="json"
            ),
        )

    async def _relay(self, frame: StreamFrame) -> None:
        params: Final = frame.params
        match frame.method:
            case "notifications/events/active":
                self._cursor = self._seal(params.get("cursor"))
                truncated: Final = params.get("truncated") is True or self._expired_on_open
                self._expired_on_open = False
                await self._emit(
                    frame.method,
                    ActiveNotification(
                        cursor=self._cursor, truncated=truncated, heartbeat_ms=self._heartbeat_ms
                    ).model_dump(by_alias=True, mode="json"),
                )
            case "notifications/events/event":
                await self._relay_event(params)
            case "notifications/events/heartbeat":
                self._cursor = self._seal(params.get("cursor")) or self._cursor
                await self._emit(frame.method, HeartbeatNotification(cursor=self._cursor).model_dump(by_alias=True))
            case "notifications/events/error":
                await self._emit(frame.method, {"error": params.get("error") or {}})
            case "notifications/events/terminated":
                self._ended = True
                await self._emit(frame.method, {"error": params.get("error") or {}})

    async def _relay_event(self, params: JsonObject) -> None:
        if not await self._authorized():
            await self._terminate(forbidden("Access revoked"))
            return
        try:
            occurrence: Final = EventOccurrence.model_validate(params)
        except ValidationError:
            return
        admitted: Final = await _screen_all(
            (occurrence,), self._context, self._prepared, self._settings, str(self._meta[SUBSCRIPTION_ID_META_KEY])
        )
        if not admitted.events:
            await self._truncate(occurrence.cursor)
            return
        self._cursor = self._seal(occurrence.cursor) or self._cursor
        await self._emit(
            "notifications/events/event",
            admitted.events[0].model_copy(update={"cursor": self._cursor}).model_dump(by_alias=True, mode="json"),
        )
        await record(
            EventActivity(
                call_type="mcp_event_stream",
                mode="push",
                server_id=self._prepared.resolved.target.server.server_id,
                event_name=self._prepared.resolved.gateway_name,
                outcome="delivered",
                caller=self._context.user_api_key_auth,
                subscription_id=str(self._meta[SUBSCRIPTION_ID_META_KEY]),
                event_ids=(occurrence.event_id,),
            )
        )

    async def consume(self, frames: MemoryObjectReceiveStream[StreamFrame | Overflow]) -> None:
        while not self._ended:
            item = await self._receive(frames)
            if item is None:
                return
            if isinstance(item, Overflow):
                await self._truncate(item.cursor)
            elif isinstance(item, StreamFrame):
                await self._relay(item)
            else:
                await self._emit(
                    "notifications/events/heartbeat",
                    HeartbeatNotification(cursor=self._cursor).model_dump(by_alias=True),
                )
            if not self._ended and time.monotonic() >= self._next_check:
                self._next_check = time.monotonic() + _RECHECK_SECONDS
                if not await self._authorized():
                    await self._terminate(forbidden("Access revoked"))

    async def _receive(
        self, frames: MemoryObjectReceiveStream[StreamFrame | Overflow]
    ) -> StreamFrame | Overflow | _Idle | None:
        with anyio.move_on_after(self._heartbeat_ms / 1000):
            try:
                return await frames.receive()
            except anyio.EndOfStream:
                return None
        return _IDLE


async def push(
    send: SendNotification,
    subscription_id: str | int,
    context: OperationContext,
    params: StreamEventsParams,
    headers: Mapping[str, str] | None,
    settings: EventsSettings,
    authorize: Authorize,
) -> None:
    prepared: Final = await admit(context, params.name, params.arguments, "push", headers, settings)
    if isinstance(prepared, Error):
        raise prepared.error.to_mcp()
    binding: Final = binding_for(context, prepared.ok, params.arguments)
    opened: Final = open_cursor(params.cursor, binding, int(time.time()))
    if isinstance(opened, Error):
        raise opened.error.to_mcp()
    heartbeat_ms: Final = granted_heartbeat(params.heartbeat_ms)
    relay: Final = _PushRelay(
        send, subscription_id, context, prepared.ok, binding, settings, authorize, heartbeat_ms, opened.ok.expired
    )
    upstream_params: Final = StreamEventsParams(
        name=prepared.ok.resolved.upstream_name,
        arguments=params.arguments,
        cursor=opened.ok.upstream,
        max_age_ms=params.max_age_ms,
        heartbeat_ms=heartbeat_ms,
    ).model_dump(by_alias=True, mode="json", exclude_none=True, exclude={"meta"})
    async with stream_slots.hold(caller_key(context), settings):
        await stream(
            prepared.ok.upstream.client,
            upstream_params,
            routing_headers(prepared.ok.resolved.upstream_name, params.arguments, prepared.ok.descriptor),
            relay.consume,
        )
