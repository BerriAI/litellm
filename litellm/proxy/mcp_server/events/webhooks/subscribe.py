from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Final

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
)
from litellm.proxy.mcp_server.events.errors import (
    EventsFailure,
    callback_error,
    exhausted,
    forbidden,
    internal_error,
    invalid_params,
    unsupported,
)
from litellm.proxy.mcp_server.events.runtime import EventsRuntime
from litellm.proxy.mcp_server.events.webhooks.delivery import is_https_callback
from litellm.proxy.mcp_server.events.webhooks.policy import (
    subscription_id,
)
from litellm.proxy.mcp_server.events.webhooks.signing import new_secret, parse_secret
from litellm.proxy.mcp_server.events.webhooks.store import (
    EventStore,
    SubscriptionRow,
    default_store,
)
from litellm.proxy.mcp_server.events.webhooks.termination import end_upstream
from litellm.proxy.mcp_server.events.webhooks.verification import Poster, verify_callback
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.types.mcp_server.events import (
    EventsErrorCode,
    SubscribeEventsParams,
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
    if not prepared.ok.resolved.target.server.available_on_public_internet:
        return Error(forbidden("Webhook subscriptions cannot retain client network authority"))
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
