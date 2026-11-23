from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Final, TypeAlias

import anyio
from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.shared.exceptions import MCPError
from mcp.types import METHOD_NOT_FOUND, EmptyResult
from mcp_types import CLIENT_CAPABILITIES_META_KEY
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError
from starlette.requests import Request as StarletteRequest

from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.events.catalog import event_servers, list_events
from litellm.proxy.mcp_server.events.config import EventsSettings, events_settings
from litellm.proxy.mcp_server.events.errors import invalid_params
from litellm.proxy.mcp_server.events.poll import poll
from litellm.proxy.mcp_server.events.push import push
from litellm.proxy.mcp_server.events.runtime import DEFAULT_RUNTIME, EventsRuntime
from litellm.proxy.mcp_server.events.webhooks.store import default_store
from litellm.proxy.mcp_server.outbound_credentials.result import Error
from litellm.types.mcp_server.events import (
    EVENTS_EXTENSION,
    EVENTS_PROTOCOL_VERSION,
    SUBSCRIPTION_ID_META_KEY,
    JsonObject,
    ListEventsParams,
    ListEventsResult,
    PollEventsParams,
    PollEventsResult,
    StreamEventsParams,
    StreamEventsResult,
    SubscribeEventsParams,
    SubscribeEventsResult,
    UnsubscribeEventsParams,
)

if TYPE_CHECKING:
    from mcp.server.lowlevel.server import Server

    from litellm.proxy.mcp_server.subscriptions import SubscriptionAuthorization

OperationScope: TypeAlias = Callable[
    [ServerRequestContext[object, object]], AbstractAsyncContextManager[OperationContext]
]
AuthorizationScope: TypeAlias = Callable[
    [ServerRequestContext[object, object]], AbstractAsyncContextManager["SubscriptionAuthorization"]
]
Middleware: TypeAlias = Callable[[ServerRequestContext[object, object], CallNext], Awaitable[HandlerResult]]

_META: Final = TypeAdapter[dict[str, object]](dict[str, object])
EVENT_METHODS: Final = frozenset(
    {"events/list", "events/poll", "events/stream", "events/subscribe", "events/unsubscribe"}
)


class _Notification(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: str
    params: JsonObject


def events_served(settings: EventsSettings, protocol_version: str | None, proxy_mode: bool) -> bool:
    return settings.enabled and protocol_version == EVENTS_PROTOCOL_VERSION and not proxy_mode


def gated_events_method(
    ctx: ServerRequestContext[object, object],
    proxy_mode: bool,
    settings: Callable[[], EventsSettings] = events_settings,
) -> bool:
    """An events/* method the gateway answers as unknown: events are off, or this request did not declare them."""
    if ctx.method not in EVENT_METHODS:
        return False
    return not events_served(settings(), ctx.protocol_version, proxy_mode) or not client_declares_events(
        _request_meta(ctx)
    )


def _request_meta(ctx: ServerRequestContext[object, object]) -> Mapping[str, object] | None:
    try:
        return _META.validate_python((ctx.params or {}).get("_meta") or {})
    except ValidationError:
        return None


def _method_not_found(method: str) -> MCPError:
    return MCPError(code=METHOD_NOT_FOUND, message="Method not found", data=method)


def events_gate(proxy_mode: Callable[[], bool], settings: Callable[[], EventsSettings] = events_settings) -> Middleware:
    async def gate(ctx: ServerRequestContext[object, object], call_next: CallNext) -> HandlerResult:
        if gated_events_method(ctx, proxy_mode(), settings):
            raise _method_not_found(ctx.method)
        return await call_next(ctx)

    return gate


class _ClientCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True)

    extensions: Mapping[str, JsonValue] | None = None


def client_declares_events(meta: Mapping[str, object] | None) -> bool:
    try:
        capabilities: Final = _ClientCapabilities.model_validate((meta or {}).get(CLIENT_CAPABILITIES_META_KEY) or {})
    except ValidationError:
        return False
    return EVENTS_EXTENSION in (capabilities.extensions or {})


async def advertised_extensions(
    context: OperationContext,
    meta: Mapping[str, object] | None,
    settings: Callable[[], EventsSettings] = events_settings,
    runtime: EventsRuntime = DEFAULT_RUNTIME,
    *,
    timeout: float | None = None,
) -> Mapping[str, JsonObject]:
    if not events_served(settings(), context.protocol_version, context.mcp_proxy_mode) or not client_declares_events(
        meta
    ):
        return {}
    try:
        if timeout is not None and timeout <= 0:
            return {}
        with anyio.move_on_after(timeout):
            servers: Final = event_servers(await runtime.allowed_servers(context))
            return {EVENTS_EXTENSION: {}} if servers else {}
    except Exception:  # noqa: BLE001  # an optional extension probe must not fail server/discover
        return {}
    return {}


def request_headers(ctx: ServerRequestContext[object, object]) -> Mapping[str, str] | None:
    request: Final = getattr(ctx, "request", None)
    return dict(request.headers) if isinstance(request, StarletteRequest) else None


def register_event_handlers(
    server: "Server[object]",
    operation_scope: OperationScope,
    authorization_scope: AuthorizationScope,
    settings: Callable[[], EventsSettings] = events_settings,
    runtime: EventsRuntime = DEFAULT_RUNTIME,
) -> None:
    def served(ctx: ServerRequestContext[object, object], context: OperationContext) -> EventsSettings:
        current: Final = settings()
        if not events_served(current, ctx.protocol_version, context.mcp_proxy_mode):
            raise _method_not_found(ctx.method)
        return current

    async def list_handler(ctx: ServerRequestContext[object, object], params: ListEventsParams) -> ListEventsResult:
        async with operation_scope(ctx) as context:
            current: Final = served(ctx, context)
            if params.cursor is not None:
                raise invalid_params("Invalid pagination cursor").to_mcp()
            return ListEventsResult(
                events=await list_events(context, current, runtime, store_available=default_store() is not None)
            )

    async def poll_handler(ctx: ServerRequestContext[object, object], params: PollEventsParams) -> PollEventsResult:
        async with operation_scope(ctx) as context:
            result: Final = await poll(context, params, request_headers(ctx), served(ctx, context), runtime)
            if isinstance(result, Error):
                raise result.error.to_mcp()
            return result.ok

    async def stream_handler(
        ctx: ServerRequestContext[object, object], params: StreamEventsParams
    ) -> StreamEventsResult:
        subscription_id: Final = ctx.request_id if ctx.request_id is not None else 0

        async def send(method: str, payload: JsonObject) -> None:
            await ctx.session.send_notification(
                _Notification(method=method, params=payload),  # pyright: ignore[reportArgumentType]  # events notifications are outside the SDK's typed union
                related_request_id=subscription_id,
            )

        async with operation_scope(ctx) as context:
            current: Final = served(ctx, context)
            async with authorization_scope(ctx) as authorization:
                await push(
                    send, subscription_id, context, params, request_headers(ctx), current, authorization, runtime
                )
        return StreamEventsResult.model_validate({"_meta": {SUBSCRIPTION_ID_META_KEY: subscription_id}})

    async def subscribe_handler(
        ctx: ServerRequestContext[object, object], params: SubscribeEventsParams
    ) -> SubscribeEventsResult:
        from litellm.proxy.mcp_server.events.webhooks.subscribe import subscribe

        async with operation_scope(ctx) as context:
            result: Final = await subscribe(context, params, request_headers(ctx), served(ctx, context), runtime)
            if isinstance(result, Error):
                raise result.error.to_mcp()
            return result.ok

    async def unsubscribe_handler(
        ctx: ServerRequestContext[object, object], params: UnsubscribeEventsParams
    ) -> EmptyResult:
        from litellm.proxy.mcp_server.events.webhooks.subscribe import unsubscribe

        async with operation_scope(ctx) as context:
            result: Final = await unsubscribe(context, params, request_headers(ctx), served(ctx, context), runtime)
            if isinstance(result, Error):
                raise result.error.to_mcp()
            return EmptyResult()

    server.add_request_handler("events/list", ListEventsParams, list_handler)
    server.add_request_handler("events/poll", PollEventsParams, poll_handler)
    server.add_request_handler("events/stream", StreamEventsParams, stream_handler)
    server.add_request_handler("events/subscribe", SubscribeEventsParams, subscribe_handler)
    server.add_request_handler("events/unsubscribe", UnsubscribeEventsParams, unsubscribe_handler)
