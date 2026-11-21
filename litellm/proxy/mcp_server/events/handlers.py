from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Final

from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.shared.exceptions import MCPError
from mcp.types import METHOD_NOT_FOUND
from mcp_types import CLIENT_CAPABILITIES_META_KEY
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError
from starlette.requests import Request as StarletteRequest

from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.events.catalog import allowed_servers, event_servers, list_events
from litellm.proxy.mcp_server.events.config import EventsSettings, events_settings
from litellm.proxy.mcp_server.events.errors import invalid_params
from litellm.proxy.mcp_server.events.relay import poll, push
from litellm.proxy.mcp_server.outbound_credentials.result import Error
from litellm.types.mcp_server.events import (
    EVENTS_EXTENSION,
    EVENTS_PROTOCOL_VERSION,
    SUBSCRIPTION_ID_META_KEY,
    JsonObject,
    ListEventsParams,
    ListEventsResult,
    PollEventsParams,
    StreamEventsParams,
    SubscribeEventsParams,
    UnsubscribeEventsParams,
)

if TYPE_CHECKING:
    from mcp.server.lowlevel.server import Server

    from litellm.proxy.mcp_server.subscriptions import SubscriptionAuthorization

OperationScope = Callable[[ServerRequestContext], AbstractAsyncContextManager[OperationContext]]
AuthorizationScope = Callable[[ServerRequestContext], AbstractAsyncContextManager["SubscriptionAuthorization"]]

EVENT_METHODS: Final = frozenset(
    {"events/list", "events/poll", "events/stream", "events/subscribe", "events/unsubscribe"}
)
_EXTENSION_MAP: Final = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue])


class _Notification(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: str
    params: JsonObject


def events_served(settings: EventsSettings, protocol_version: str | None, proxy_mode: bool) -> bool:
    return settings.enabled and protocol_version == EVENTS_PROTOCOL_VERSION and not proxy_mode


def gated_events_method(
    method: str,
    protocol_version: str | None,
    proxy_mode: bool,
    settings: Callable[[], EventsSettings] = events_settings,
) -> bool:
    return method in EVENT_METHODS and not events_served(settings(), protocol_version, proxy_mode)


def events_gate(
    proxy_mode: Callable[[], bool], settings: Callable[[], EventsSettings] = events_settings
) -> Callable[[ServerRequestContext[object, object], CallNext], Awaitable[HandlerResult]]:
    async def gate(ctx: ServerRequestContext[object, object], call_next: CallNext) -> HandlerResult:
        if gated_events_method(ctx.method, ctx.protocol_version, proxy_mode(), settings):
            raise MCPError(code=METHOD_NOT_FOUND, message="Method not found", data=ctx.method)
        return await call_next(ctx)

    return gate


def client_declares_events(meta: Mapping[str, object] | None) -> bool:
    capabilities: Final = (meta or {}).get(CLIENT_CAPABILITIES_META_KEY)
    if not isinstance(capabilities, Mapping):
        return False
    try:
        return EVENTS_EXTENSION in _EXTENSION_MAP.validate_python(capabilities.get("extensions") or {})
    except ValidationError:
        return False


async def advertised_extensions(
    context: OperationContext, meta: Mapping[str, object] | None, settings: EventsSettings | None = None
) -> Mapping[str, JsonObject]:
    resolved: Final = settings or events_settings()
    if not events_served(resolved, context.protocol_version, context.mcp_proxy_mode) or not client_declares_events(
        meta
    ):
        return {}
    try:
        servers: Final = event_servers(await allowed_servers(context))
    except Exception:  # noqa: BLE001  # an optional extension probe must not fail server/discover
        return {}
    return {EVENTS_EXTENSION: {}} if servers else {}


def request_headers(ctx: ServerRequestContext) -> Mapping[str, str] | None:
    request: Final = getattr(ctx, "request", None)
    return dict(request.headers) if isinstance(request, StarletteRequest) else None


def register_event_handlers(
    server: "Server[object]",
    operation_scope: OperationScope,
    authorization_scope: AuthorizationScope,
    settings: Callable[[], EventsSettings] = events_settings,
) -> None:
    def served(ctx: ServerRequestContext, context: OperationContext) -> EventsSettings:
        current: Final = settings()
        if not events_served(current, ctx.protocol_version, context.mcp_proxy_mode):
            raise MCPError(code=METHOD_NOT_FOUND, message="Method not found", data=ctx.method)
        return current

    async def list_handler(ctx: ServerRequestContext, params: ListEventsParams) -> dict[str, object]:
        async with operation_scope(ctx) as context:
            current: Final = served(ctx, context)
            if params.cursor is not None:
                raise invalid_params("Invalid pagination cursor").to_mcp()
            return ListEventsResult(events=await list_events(context, current)).model_dump(
                by_alias=True, mode="json", exclude_none=True
            )

    async def poll_handler(ctx: ServerRequestContext, params: PollEventsParams) -> dict[str, object]:
        async with operation_scope(ctx) as context:
            result: Final = await poll(context, params, request_headers(ctx), served(ctx, context))
            if isinstance(result, Error):
                raise result.error.to_mcp()
            return result.ok.model_dump(by_alias=True, mode="json", exclude_none=True)

    async def stream_handler(ctx: ServerRequestContext, params: StreamEventsParams) -> dict[str, object]:
        subscription_id: Final = ctx.request_id if ctx.request_id is not None else 0

        async def send(method: str, payload: JsonObject) -> None:
            await ctx.session.send_notification(
                _Notification(method=method, params=payload),  # pyright: ignore[reportArgumentType]  # events notifications are outside the SDK's typed union
                related_request_id=subscription_id,
            )

        async with operation_scope(ctx) as context:
            current: Final = served(ctx, context)
            async with authorization_scope(ctx) as authorization:
                await push(send, subscription_id, context, params, request_headers(ctx), current, authorization)
        return {"_meta": {SUBSCRIPTION_ID_META_KEY: subscription_id}}

    async def subscribe_handler(ctx: ServerRequestContext, params: SubscribeEventsParams) -> dict[str, object]:
        from litellm.proxy.mcp_server.events.webhooks.subscribe import subscribe

        async with operation_scope(ctx) as context:
            result: Final = await subscribe(context, params, request_headers(ctx), served(ctx, context))
            if isinstance(result, Error):
                raise result.error.to_mcp()
            return result.ok

    async def unsubscribe_handler(ctx: ServerRequestContext, params: UnsubscribeEventsParams) -> dict[str, object]:
        from litellm.proxy.mcp_server.events.webhooks.subscribe import unsubscribe

        async with operation_scope(ctx) as context:
            result: Final = await unsubscribe(context, params, request_headers(ctx), served(ctx, context))
            if isinstance(result, Error):
                raise result.error.to_mcp()
            return {}

    server.add_request_handler("events/list", ListEventsParams, list_handler)
    server.add_request_handler("events/poll", PollEventsParams, poll_handler)
    server.add_request_handler("events/stream", StreamEventsParams, stream_handler)
    server.add_request_handler("events/subscribe", SubscribeEventsParams, subscribe_handler)
    server.add_request_handler("events/unsubscribe", UnsubscribeEventsParams, unsubscribe_handler)
