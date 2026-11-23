import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Final

import anyio
from mcp.shared.exceptions import MCPError

from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.events.access import holds_webhooks, principal_of, stored_key_hash
from litellm.proxy.mcp_server.events.admission import admit, binding_for
from litellm.proxy.mcp_server.events.catalog import PreparedEvent
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
from litellm.proxy.mcp_server.events.webhooks.delivery import CALLBACK_DEADLINE_SECONDS, is_https_callback
from litellm.proxy.mcp_server.events.webhooks.policy import (
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
    SubscriptionRow,
    SubscriptionStatus,
    default_store,
)
from litellm.proxy.mcp_server.events.webhooks.termination import end_upstream
from litellm.proxy.mcp_server.events.webhooks.verification import Poster, verify_callback
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.types.mcp_server.events import (
    DeliveryStatus,
    EventsErrorCode,
    SubscribeEventsParams,
    SubscribeEventsResult,
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


def validate_subscription(
    context: OperationContext, params: SubscribeEventsParams, settings: EventsSettings, store: EventStore | None
) -> Result[_Request, EventsFailure]:
    if settings.public_base_url is None:
        return Error(unsupported("webhook"))
    principal: Final = principal_of(context.user_api_key_auth)
    caller: Final = context.user_api_key_auth
    if caller is not None and (caller.end_user_id or caller.agent_id):
        return Error(forbidden("Webhook subscriptions cannot retain request-scoped end-user or agent permissions"))
    if principal is None or not holds_webhooks(caller):
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


async def _discard_pending(subscription: _Subscription) -> None:
    if subscription.reserved:
        await subscription.request.store.discard_pending(subscription.base.id, subscription.base.generation)


async def _roll_back(
    subscription: _Subscription, upstream: SubscribeEventsResult, settings: EventsSettings, runtime: EventsRuntime
) -> None:
    """Undo an upstream subscription this request created but could not record."""
    # The conditional delete serializes with a possibly committed activation before the ownership read.
    await _discard_pending(subscription)
    current: Final = await subscription.request.store.get(subscription.base.id)
    if current is not None and current.upstream_secret == subscription.base.upstream_secret:
        if current.status in ("active", "suspended"):
            return
        if current.status == "pending" and current.generation != subscription.base.generation:
            return
    orphan: Final = subscription.base.model_copy(update={"upstream_subscription_id": upstream.id})
    await end_upstream(orphan, settings, runtime, prepared=subscription.prepared)


async def activate_subscription(
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
        reactivation_gap_id=new_webhook_id("gap") if existing is not None and existing.status == "suspended" else None,
        expected_generation=base.generation,
        client_secret=base.client_secret,
        upstream_subscription_id=upstream.id,
        upstream_refresh_before=upstream_refresh,
        refresh_before=refresh_before,
        cursor=sealed.ok,
    )
    expected: Final[tuple[SubscriptionStatus, ...]] = (
        ("pending",) if subscription.reserved else (existing.status,) if existing is not None else ("pending",)
    )
    try:
        activated: Final = await subscription.request.store.activate(base.id, activation, expected)
    except Exception:
        await _roll_back(subscription, upstream, settings, runtime)
        raise
    if activated is None:
        await _roll_back(subscription, upstream, settings, runtime)
        return Error(subscription_not_found())
    refreshed: Final = existing if existing is not None and existing.status != "terminated" else None
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
        cursor=activated.cursor,
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
    if await store.end_if_current(row.id, row.upstream_subscription_id, row.generation):
        await end_upstream(row, settings, runtime)
    return row.model_copy(update={"status": "terminated"})


async def prepare_subscription(
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


async def reserve_subscription(
    subscription: _Subscription, settings: EventsSettings
) -> Result[_Subscription, EventsFailure]:
    """Claim the subscription id before any upstream work, so racing subscribes share one row and one secret."""
    if subscription.existing is not None and subscription.existing.status == "pending":
        return Error(_BEING_CREATED)
    if subscription.existing is not None and subscription.existing.status != "terminated":
        return Ok(subscription)
    store: Final = subscription.request.store
    claiming: Final = asyncio.create_task(store.reserve(subscription.base))
    try:
        try:
            if not await asyncio.shield(claiming):
                return Error(_BEING_CREATED)
        finally:
            # Prisma can finish an insert after caller cancellation. Order the delete after its write outcome.
            if not claiming.done():
                with anyio.move_on_after(CALLBACK_DEADLINE_SECONDS, shield=True):
                    await claiming
        if await store.count_live(subscription.request.principal) > settings.max_subscriptions_per_key:
            await store.discard_pending(subscription.base.id, subscription.base.generation)
            return Error(exhausted("subscriptions", settings.max_subscriptions_per_key))
    except BaseException:
        with anyio.move_on_after(CALLBACK_DEADLINE_SECONDS, shield=True):
            await store.discard_pending(subscription.base.id, subscription.base.generation)
        raise
    return Ok(replace(subscription, reserved=True))
