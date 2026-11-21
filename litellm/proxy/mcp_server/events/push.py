from collections.abc import Awaitable, Callable, Mapping
from typing import Final, TypeAlias

import anyio
from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.events.admission import admit, binding_for, caller_key, screen_all, still_authorized
from litellm.proxy.mcp_server.events.catalog import PreparedEvent
from litellm.proxy.mcp_server.events.config import EventsSettings, events_settings
from litellm.proxy.mcp_server.events.cursors import CursorBinding, ensure_sealable, open_cursor, seal_cursor
from litellm.proxy.mcp_server.events.errors import EventsFailure, forbidden, unsupported
from litellm.proxy.mcp_server.events.observability import EventActivity, EventOutcome
from litellm.proxy.mcp_server.events.runtime import EventsRuntime, Frames
from litellm.proxy.mcp_server.events.upstream import Overflow, StreamFrame, routing_headers
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok
from litellm.types.mcp_server.events import (
    SUBSCRIPTION_ID_META_KEY,
    ActiveNotification,
    EventOccurrence,
    EventsError,
    HeartbeatNotification,
    JsonObject,
    StreamEventsParams,
)

DEFAULT_HEARTBEAT_MS: Final = 30_000
MIN_HEARTBEAT_MS: Final = 5_000
MAX_HEARTBEAT_MS: Final = 60_000
RECHECK_SECONDS: Final = 15.0
_JSON_OBJECT: Final = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue])

SendNotification: TypeAlias = Callable[[str, JsonObject], Awaitable[None]]
Authorize: TypeAlias = Callable[[], Awaitable[OperationContext]]


class _Idle:
    pass


_IDLE: Final = _Idle()


def granted_heartbeat(requested: int | None) -> int:
    return min(max(requested or DEFAULT_HEARTBEAT_MS, MIN_HEARTBEAT_MS), MAX_HEARTBEAT_MS)


class PushRelay:
    def __init__(
        self,
        send: SendNotification,
        subscription_id: str | int,
        prepared: PreparedEvent,
        binding: CursorBinding,
        settings: EventsSettings,
        authorize: Authorize,
        heartbeat_ms: int,
        expired: bool,
        runtime: EventsRuntime,
        caller: UserAPIKeyAuth | None,
        enabled: Callable[[], bool] = lambda: events_settings().enabled,
    ) -> None:
        self._caller = caller
        self._enabled = enabled
        self._send = send
        self._subscription_id = subscription_id
        self._prepared = prepared
        self._binding = binding
        self._settings = settings
        self._authorize = authorize
        self._heartbeat_ms = heartbeat_ms
        self._truncate_next_active = expired
        self._runtime = runtime
        self._cursor: str | None = None
        self._active = False
        self._ended = False
        self._next_check = runtime.clock() + RECHECK_SECONDS

    async def _emit(self, method: str, params: JsonObject) -> None:
        await self._send(method, {**params, "_meta": {SUBSCRIPTION_ID_META_KEY: self._subscription_id}})

    def _seal(self, upstream: JsonValue | None) -> str | None:
        if not isinstance(upstream, str):
            return None
        sealed: Final = seal_cursor(self._binding, upstream, self._runtime.now())
        return sealed.ok if isinstance(sealed, Ok) else None

    async def _record(self, outcome: EventOutcome, event_ids: tuple[str, ...] = ()) -> None:
        await self._runtime.record(
            EventActivity(
                call_type="mcp_event_stream",
                mode="push",
                server_id=self._prepared.resolved.target.server.server_id,
                event_name=self._prepared.resolved.gateway_name,
                outcome=outcome,
                caller=self._caller,
                subscription_id=str(self._subscription_id),
                event_ids=event_ids,
            )
        )

    async def _terminate(self, failure: EventsFailure) -> None:
        self._ended = True
        error: Final = EventsError(code=int(failure.code), message=failure.message, data=failure.data)
        envelope: Final = _JSON_OBJECT.validate_python({"error": error.model_dump(mode="json", exclude_none=True)})
        await self._emit("notifications/events/terminated", envelope)
        await self._record("terminated")

    async def _authorized(self) -> OperationContext | None:
        try:
            fresh: Final = await self._authorize()
        except Exception:  # noqa: BLE001  # any failure to re-authorize ends the subscription
            return None
        if caller_key(fresh) != self._binding.caller:
            return None
        try:
            allowed: Final = await still_authorized(fresh, self._prepared, self._runtime)
        except Exception:  # noqa: BLE001  # a permission lookup that fails ends the subscription like a denial
            return None
        return fresh if allowed else None

    async def _announce(self, upstream_cursor: JsonValue | None, truncated: bool) -> None:
        self._cursor = self._seal(upstream_cursor) or self._cursor
        notification: Final = ActiveNotification(
            cursor=self._cursor, truncated=truncated, heartbeat_ms=self._heartbeat_ms
        )
        await self._emit("notifications/events/active", notification.model_dump(by_alias=True, mode="json"))

    async def _heartbeat(self, upstream_cursor: JsonValue | None) -> None:
        self._cursor = self._seal(upstream_cursor) or self._cursor
        await self._emit("notifications/events/heartbeat", HeartbeatNotification(cursor=self._cursor).model_dump())

    async def _relay(self, frame: StreamFrame) -> None:
        params: Final = frame.params
        match frame.method:
            case "notifications/events/active":
                truncated: Final = params.get("truncated") is True or self._truncate_next_active
                self._truncate_next_active = False
                self._active = True
                await self._announce(params.get("cursor"), truncated)
            case "notifications/events/event":
                await self._relay_event(params)
            case "notifications/events/heartbeat":
                await self._heartbeat(params.get("cursor"))
            case "notifications/events/error":
                await self._emit(frame.method, {"error": params.get("error") or {}})
            case "notifications/events/terminated":
                self._ended = True
                await self._emit(frame.method, {"error": params.get("error") or {}})
                await self._record("terminated")

    async def _relay_event(self, params: JsonObject) -> None:
        fresh: Final = await self._authorized()
        if fresh is None:
            await self._terminate(forbidden("Access revoked"))
            return
        try:
            occurrence: Final = EventOccurrence.model_validate(params)
        except ValidationError:
            await self._announce(params.get("cursor"), True)
            await self._record("failed")
            return
        admitted: Final = await screen_all(
            (occurrence,), fresh, self._prepared, self._settings, str(self._subscription_id), self._runtime
        )
        if not admitted.events:
            await self._announce(occurrence.cursor, True)
            await self._record("blocked" if admitted.blocked else "throttled")
            return
        self._cursor = self._seal(occurrence.cursor) or self._cursor
        delivered: Final = admitted.events[0].model_copy(update={"cursor": self._cursor})
        await self._emit(
            "notifications/events/event", _JSON_OBJECT.validate_python(delivered.model_dump(by_alias=True, mode="json"))
        )
        await self._record("delivered", (occurrence.event_id,))

    async def _receive(self, frames: Frames) -> StreamFrame | Overflow | _Idle | None:
        with anyio.move_on_after(self._heartbeat_ms / 1000):
            try:
                return await frames.receive()
            except anyio.EndOfStream:
                return None
        return _IDLE

    async def consume(self, frames: Frames) -> None:
        while not self._ended and await self._step(frames):
            pass

    async def _step(self, frames: Frames) -> bool:
        """Handle one frame or idle tick; False once the upstream stream has ended."""
        item: Final = await self._receive(frames)
        if item is None:
            return False
        if isinstance(item, Overflow):
            await self._announce(item.cursor, True)
        elif isinstance(item, StreamFrame):
            await self._relay(item)
        elif self._active:
            await self._heartbeat(None)
        if not self._ended and self._runtime.clock() >= self._next_check:
            self._next_check = self._runtime.clock() + RECHECK_SECONDS
            if not self._enabled():
                await self._terminate(unsupported("push"))
            elif await self._authorized() is None:
                await self._terminate(forbidden("Access revoked"))
        return True


async def push(
    send: SendNotification,
    subscription_id: str | int,
    context: OperationContext,
    params: StreamEventsParams,
    headers: Mapping[str, str] | None,
    settings: EventsSettings,
    authorize: Authorize,
    runtime: EventsRuntime,
) -> None:
    prepared: Final = await admit(context, params.name, params.arguments, "push", headers, settings, runtime)
    if isinstance(prepared, Error):
        raise prepared.error.to_mcp()
    binding: Final = binding_for(context, prepared.ok, params.arguments)
    unsealable: Final = ensure_sealable(binding, runtime.now())
    if unsealable is not None:
        raise unsealable.to_mcp()
    opened: Final = open_cursor(params.cursor, binding, runtime.now())
    if isinstance(opened, Error):
        raise opened.error.to_mcp()
    heartbeat_ms: Final = granted_heartbeat(params.heartbeat_ms)
    relay: Final = PushRelay(
        send,
        subscription_id,
        prepared.ok,
        binding,
        settings,
        authorize,
        heartbeat_ms,
        opened.ok.expired,
        runtime,
        context.user_api_key_auth,
    )
    upstream_params: Final = StreamEventsParams(
        name=prepared.ok.resolved.upstream_name,
        arguments=params.arguments,
        cursor=opened.ok.upstream,
        max_age_ms=params.max_age_ms,
        heartbeat_ms=heartbeat_ms,
    )
    async with runtime.slots.hold(caller_key(context), settings):
        await runtime.stream(
            prepared.ok.upstream,
            _JSON_OBJECT.validate_python(upstream_params.model_dump(by_alias=True, mode="json", exclude_none=True)),
            routing_headers(prepared.ok.resolved.upstream_name, params.arguments, prepared.ok.descriptor),
            relay.consume,
        )
