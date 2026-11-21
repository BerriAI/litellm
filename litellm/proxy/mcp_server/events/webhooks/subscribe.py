from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Final

import anyio
from mcp.shared.exceptions import MCPError
from pydantic import ValidationError

from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.events.access import holds_webhooks, principal_of, stored_key_hash
from litellm.proxy.mcp_server.events.admission import admit, binding_for, check_routing_headers
from litellm.proxy.mcp_server.events.catalog import PreparedEvent, prepare_event
from litellm.proxy.mcp_server.events.config import EventsSettings
from litellm.proxy.mcp_server.events.cursors import (
    CursorBinding,
    OpenedCursor,
    ensure_sealable,
    open_cursor,
    seal_cursor,
)
from litellm.proxy.mcp_server.events.errors import (
    EventsFailure,
    callback_error,
    exhausted,
    forbidden,
    internal_error,
    invalid_params,
    subscription_not_found,
    unsupported,
)
from litellm.proxy.mcp_server.events.observability import EventActivity
from litellm.proxy.mcp_server.events.runtime import EventsRuntime
from litellm.proxy.mcp_server.events.upstream import routing_headers
from litellm.proxy.mcp_server.events.webhooks.delivery import is_https_callback, post_callback
from litellm.proxy.mcp_server.events.webhooks.policy import (
    callback_url,
    client_grant,
    granted_ttl_ms,
    iso,
    parse_time,
    subscription_id,
)
from litellm.proxy.mcp_server.events.webhooks.signing import new_secret, new_webhook_id, parse_secret
from litellm.proxy.mcp_server.events.webhooks.store import (
    Activation,
    EventStore,
    QueuedDelivery,
    SubscriptionRow,
    default_store,
)
from litellm.proxy.mcp_server.events.webhooks.termination import end_upstream
from litellm.proxy.mcp_server.events.webhooks.verification import Poster, verify_callback
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.types.mcp_server.events import (
    DeliveryStatus,
    EventsErrorCode,
    GapEnvelope,
    JsonObject,
    SubscribeEventsParams,
    SubscribeEventsResult,
    UnsubscribeEventsParams,
)

_KNOWN_CODES: Final = frozenset(member.value for member in EventsErrorCode)
_BEING_CREATED: Final = internal_error("Another request is creating this subscription; retry")


@dataclass(frozen=True, slots=True)
class _Request:
    principal: str
    secret: bytes
    store: EventStore


@dataclass(frozen=True, slots=True)
class _Subscription:
    base: SubscriptionRow
    existing: SubscriptionRow | None
    prepared: PreparedEvent
    params: SubscribeEventsParams
    opened: OpenedCursor
    binding: CursorBinding
    request: _Request
    reserved: bool = False


def _validate(
    context: OperationContext, params: SubscribeEventsParams, settings: EventsSettings, store: EventStore | None
) -> Result[_Request, EventsFailure]:
    if settings.public_base_url is None:
        return Error(unsupported("webhook"))
    principal: Final = principal_of(context.user_api_key_auth)
    if principal is None or not holds_webhooks(context.user_api_key_auth):
        return Error(forbidden("Webhook subscriptions need a virtual key or a gateway OAuth sign-in"))
    if not is_https_callback(params.delivery.url):
        return Error(invalid_params("delivery.url must be an https URL"))
    secret: Final = parse_secret(params.delivery.secret)
    if secret is None:
        return Error(invalid_params("delivery.secret must be whsec_ followed by base64 of 24 to 64 bytes"))
    resolved_store: Final = store or default_store()
    if resolved_store is None:
        return Error(unsupported("webhook"))
    return Ok(_Request(principal=principal, secret=secret, store=resolved_store))


def delivery_status(row: SubscriptionRow) -> DeliveryStatus:
    return DeliveryStatus(
        active=row.status == "active",
        last_delivery_at=iso(row.last_delivery_at),
        last_error=row.last_error,
        failed_since=iso(row.failed_since),
    )


def upstream_failure(error: MCPError) -> EventsFailure:
    if error.error.code == EventsErrorCode.CALLBACK_ENDPOINT_ERROR:
        return internal_error("The upstream server could not verify the gateway webhook endpoint")
    known: Final = error.error.code in _KNOWN_CODES
    return EventsFailure(
        EventsErrorCode(error.error.code) if known else EventsErrorCode.INTERNAL_ERROR, error.error.message
    )


def _new_row(
    sid: str, request: _Request, context: OperationContext, prepared: PreparedEvent, params: SubscribeEventsParams
) -> SubscriptionRow:
    caller: Final = context.user_api_key_auth
    return SubscriptionRow(
        id=sid,
        principal=request.principal,
        key_hash=stored_key_hash(caller),
        user_id=caller.user_id if caller is not None else None,
        team_id=caller.team_id if caller is not None else None,
        server_id=prepared.resolved.target.server.server_id,
        event_name=prepared.resolved.upstream_name,
        arguments=params.arguments,
        client_url=params.delivery.url,
        client_secret=params.delivery.secret,
        upstream_secret=new_secret(),
        verified_at=datetime.now(timezone.utc),
    )


async def _admit_new(
    sid: str, request: _Request, params: SubscribeEventsParams, settings: EventsSettings, now: int, post: Poster
) -> EventsFailure | None:
    if await request.store.count_live(request.principal) >= settings.max_subscriptions_per_key:
        return exhausted("subscriptions", settings.max_subscriptions_per_key)
    if await request.store.verified(request.principal, params.delivery.url):
        return None
    failure: Final = await verify_callback(params.delivery.url, request.secret, sid, now, post)
    return callback_error(failure) if failure is not None else None


def _upstream_params(subscription: _Subscription, settings: EventsSettings) -> JsonObject:
    params: Final = subscription.params
    prepared: Final = subscription.prepared
    return {
        "name": prepared.resolved.upstream_name,
        "arguments": dict(params.arguments),
        "delivery": {
            "mode": "webhook",
            "url": callback_url(settings, subscription.base.id),
            "secret": subscription.base.upstream_secret,
        },
        "cursor": subscription.opened.upstream,
        **({"maxAgeMs": params.max_age_ms} if params.max_age_ms is not None else {}),
        "ttlMs": granted_ttl_ms(params, prepared.resolved.target.server),
    }


async def _subscribe_upstream(
    subscription: _Subscription, settings: EventsSettings, runtime: EventsRuntime
) -> Result[SubscribeEventsResult, EventsFailure]:
    prepared: Final = subscription.prepared
    try:
        raw: Final = await runtime.call(
            prepared.upstream,
            "events/subscribe",
            _upstream_params(subscription, settings),
            routing_headers(prepared.resolved.upstream_name, subscription.params.arguments, prepared.descriptor),
        )
        return Ok(SubscribeEventsResult.model_validate(raw))
    except MCPError as error:
        return Error(upstream_failure(error))
    except ValidationError:
        return Error(internal_error("Upstream returned an invalid events/subscribe result"))


async def _discard_pending(subscription: _Subscription) -> None:
    if subscription.reserved:
        with anyio.CancelScope(shield=True):
            await subscription.request.store.discard_pending(subscription.base.id)


async def _reactivation_gap(row: SubscriptionRow, cursor: str | None, store: EventStore) -> None:
    gap_id: Final = new_webhook_id("gap")
    await store.enqueue(
        QueuedDelivery(
            delivery_id=gap_id,
            subscription_id=row.id,
            body=GapEnvelope(cursor=cursor).model_dump_json(),
            webhook_id=gap_id,
            next_attempt_at=datetime.now(timezone.utc),
            prepared=True,
        )
    )


async def _roll_back(
    subscription: _Subscription, upstream: SubscribeEventsResult, settings: EventsSettings, runtime: EventsRuntime
) -> None:
    """Undo an upstream subscription this request created but could not record."""
    orphan: Final = subscription.base.model_copy(update={"upstream_subscription_id": upstream.id})
    await _unsubscribe_upstream(orphan, subscription.prepared, settings, runtime)
    await _discard_pending(subscription)


async def _activate(
    subscription: _Subscription,
    upstream: SubscribeEventsResult,
    context: OperationContext,
    settings: EventsSettings,
    runtime: EventsRuntime,
) -> Result[SubscribeEventsResult, EventsFailure]:
    base: Final = subscription.base
    existing: Final = subscription.existing
    upstream_refresh: Final = parse_time(upstream.refresh_before)
    refresh_before: Final = client_grant(
        datetime.now(timezone.utc),
        granted_ttl_ms(subscription.params, subscription.prepared.resolved.target.server),
        upstream_refresh,
    )
    sealed: Final = seal_cursor(subscription.binding, upstream.cursor, runtime.now())
    if isinstance(sealed, Error):
        await _roll_back(subscription, upstream, settings, runtime)
        return Error(sealed.error)
    activation: Final = Activation(
        client_secret=base.client_secret,
        upstream_subscription_id=upstream.id,
        upstream_refresh_before=upstream_refresh,
        refresh_before=refresh_before,
        cursor=sealed.ok,
    )
    if not await subscription.request.store.activate(base.id, activation):
        await _roll_back(subscription, upstream, settings, runtime)
        return Error(subscription_not_found())
    refreshed: Final = existing if existing is not None and existing.status != "terminated" else None
    if refreshed is not None and refreshed.status == "suspended":
        await _reactivation_gap(base, sealed.ok, subscription.request.store)
    await runtime.record(
        EventActivity(
            call_type="mcp_event_subscribe",
            mode="webhook",
            server_id=base.server_id,
            event_name=subscription.prepared.resolved.gateway_name,
            outcome="subscribed",
            caller=context.user_api_key_auth,
            subscription_id=base.id,
        )
    )
    result: Final = SubscribeEventsResult(
        id=base.id,
        refresh_before=iso(refresh_before),
        cursor=sealed.ok,
        truncated=upstream.truncated or subscription.opened.expired,
        delivery_status=delivery_status(refreshed) if refreshed is not None else None,
    )
    return Ok(result)


async def _retire_if_moved(
    row: SubscriptionRow | None,
    prepared: PreparedEvent,
    settings: EventsSettings,
    runtime: EventsRuntime,
    store: EventStore,
) -> SubscriptionRow | None:
    """A row for a server the name no longer resolves to is ended there, so the new server starts fresh."""
    if row is None or row.status == "terminated" or row.server_id == prepared.resolved.target.server.server_id:
        return row
    await store.terminate(row.id)
    await store.drop_deliveries(row.id)
    await end_upstream(row, settings, runtime)
    return row.model_copy(update={"status": "terminated"})


async def _prepare(
    context: OperationContext,
    params: SubscribeEventsParams,
    headers: Mapping[str, str] | None,
    settings: EventsSettings,
    runtime: EventsRuntime,
    request: _Request,
    post: Poster,
) -> Result[_Subscription, EventsFailure]:
    prepared: Final = await admit(context, params.name, params.arguments, "webhook", headers, settings, runtime)
    if isinstance(prepared, Error):
        return Error(prepared.error)
    sid: Final = subscription_id(request.principal, params.delivery.url, params.name, params.arguments)
    binding: Final = binding_for(context, prepared.ok, params.arguments)
    opened: Final = open_cursor(params.cursor, binding, runtime.now())
    if isinstance(opened, Error):
        return Error(opened.error)
    unsealable: Final = ensure_sealable(binding, runtime.now())
    if unsealable is not None:
        return Error(unsealable)
    found: Final = await request.store.get(sid)
    existing: Final = await _retire_if_moved(found, prepared.ok, settings, runtime, request.store)
    live: Final = existing if existing is not None and existing.status != "terminated" else None
    if live is None:
        refused: Final = await _admit_new(sid, request, params, settings, runtime.now(), post)
        if refused is not None:
            return Error(refused)
    base: Final = (
        live.model_copy(update={"client_secret": params.delivery.secret})
        if live is not None
        else _new_row(sid, request, context, prepared.ok, params)
    )
    return Ok(_Subscription(base, existing, prepared.ok, params, opened.ok, binding, request))


async def _reserve(subscription: _Subscription, settings: EventsSettings) -> Result[_Subscription, EventsFailure]:
    """Claim the subscription id before any upstream work, so racing subscribes share one row and one secret."""
    if subscription.existing is not None and subscription.existing.status == "pending":
        return Error(_BEING_CREATED)
    if subscription.existing is not None and subscription.existing.status != "terminated":
        return Ok(subscription)
    store: Final = subscription.request.store
    if not await store.reserve(subscription.base):
        winner: Final = await store.get(subscription.base.id)
        if winner is None or winner.status in ("terminated", "pending"):
            return Error(_BEING_CREATED)
        refreshed: Final = winner.model_copy(update={"client_secret": subscription.params.delivery.secret})
        return Ok(replace(subscription, base=refreshed, existing=winner))
    if await store.count_live(subscription.request.principal) > settings.max_subscriptions_per_key:
        await store.discard_pending(subscription.base.id)
        return Error(exhausted("subscriptions", settings.max_subscriptions_per_key))
    return Ok(replace(subscription, reserved=True))


async def subscribe(
    context: OperationContext,
    params: SubscribeEventsParams,
    headers: Mapping[str, str] | None,
    settings: EventsSettings,
    runtime: EventsRuntime,
    store: EventStore | None = None,
    post: Poster = post_callback,
) -> Result[SubscribeEventsResult, EventsFailure]:
    request: Final = _validate(context, params, settings, store)
    if isinstance(request, Error):
        return Error(request.error)
    prepared: Final = await _prepare(context, params, headers, settings, runtime, request.ok, post)
    if isinstance(prepared, Error):
        return Error(prepared.error)
    reserved: Final = await _reserve(prepared.ok, settings)
    if isinstance(reserved, Error):
        return Error(reserved.error)
    subscription: Final = reserved.ok
    try:
        upstream: Final = await _subscribe_upstream(subscription, settings, runtime)
    except BaseException:
        await _discard_pending(subscription)
        raise
    if isinstance(upstream, Error):
        await _discard_pending(subscription)
        return Error(upstream.error)
    return await _activate(subscription, upstream.ok, context, settings, runtime)


async def unsubscribe(
    context: OperationContext,
    params: UnsubscribeEventsParams,
    headers: Mapping[str, str] | None,
    settings: EventsSettings,
    runtime: EventsRuntime,
    store: EventStore | None = None,
) -> Result[None, EventsFailure]:
    principal: Final = principal_of(context.user_api_key_auth)
    if principal is None:
        return Error(forbidden("Webhook subscriptions require an authenticated caller"))
    events_store: Final = store or default_store()
    if events_store is None:
        return Error(unsupported("webhook"))
    sid: Final = subscription_id(principal, params.delivery.url, params.name, params.arguments)
    row: Final = await events_store.get(sid)
    if row is None or row.status == "terminated":
        return Error(subscription_not_found())
    prepared: Final = await _same_server_event(context, params.name, row, runtime)
    if prepared is not None:
        mismatch: Final = check_routing_headers(headers, params.name, params.arguments, prepared)
        if mismatch is not None:
            return Error(mismatch)
    if not await events_store.hold_for_unsubscribe(sid, row.upstream_subscription_id):
        return Error(subscription_not_found())
    await events_store.drop_deliveries(sid)
    if prepared is not None:
        await _unsubscribe_upstream(row, prepared, settings, runtime)
    else:
        await end_upstream(row, settings, runtime)
    await events_store.delete_held(sid, row.upstream_subscription_id)
    await runtime.record(
        EventActivity(
            call_type="mcp_event_subscribe",
            mode="webhook",
            server_id=row.server_id,
            event_name=params.name,
            outcome="unsubscribed",
            caller=context.user_api_key_auth,
            subscription_id=sid,
        )
    )
    return Ok(None)


async def _same_server_event(
    context: OperationContext, name: str, row: SubscriptionRow, runtime: EventsRuntime
) -> PreparedEvent | None:
    """The event on the server this row was made on; None when the name moved or the catalog is unreachable."""
    try:
        prepared: Final = await prepare_event(context, name, runtime)
    except Exception:  # noqa: BLE001  # an upstream outage must not keep a cancelled subscription delivering
        return None
    if isinstance(prepared, Error) or prepared.ok.resolved.target.server.server_id != row.server_id:
        return None
    return prepared.ok


async def _unsubscribe_upstream(
    row: SubscriptionRow, prepared: PreparedEvent, settings: EventsSettings, runtime: EventsRuntime
) -> None:
    if row.upstream_subscription_id is None:
        return
    try:
        await runtime.call(
            prepared.upstream,
            "events/unsubscribe",
            {
                "name": row.event_name,
                "arguments": dict(row.arguments),
                "delivery": {"url": callback_url(settings, row.id)},
            },
            routing_headers(row.event_name, row.arguments, prepared.descriptor),
        )
    except (MCPError, OSError, TimeoutError):
        return
