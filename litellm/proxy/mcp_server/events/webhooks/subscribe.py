import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Final

from mcp.shared.exceptions import MCPError
from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.events.access import principal_of, token_hash
from litellm.proxy.mcp_server.events.catalog import PreparedEvent, prepare_event
from litellm.proxy.mcp_server.events.config import EventsSettings
from litellm.proxy.mcp_server.events.cursors import open_cursor, seal_cursor
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
from litellm.proxy.mcp_server.events.observability import EventActivity, EventMetrics, record
from litellm.proxy.mcp_server.events.relay import admit, binding_for
from litellm.proxy.mcp_server.events.upstream import call, routing_headers
from litellm.proxy.mcp_server.events.webhooks.delivery import is_https_callback, post_callback
from litellm.proxy.mcp_server.events.webhooks.signing import new_secret, new_webhook_id, parse_secret, signed_headers
from litellm.proxy.mcp_server.events.webhooks.store import EventStore, QueuedDelivery, SubscriptionRow, default_store
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.events import (
    CallbackFailure,
    DeliveryStatus,
    EventsErrorCode,
    GapEnvelope,
    JsonObject,
    SubscribeEventsParams,
    SubscribeEventsResult,
    UnsubscribeEventsParams,
    VerificationEnvelope,
)
from litellm.types.mcp_server.mcp_server_manager import MCPServer

DEFAULT_TTL_MS: Final = 60 * 60 * 1000
MIN_TTL_MS: Final = 60 * 1000
MAX_TTL_MS: Final = 24 * 60 * 60 * 1000
REQUEST_SCOPED_TTL_MS: Final = 60 * 60 * 1000
_STORED_CREDENTIALS: Final = frozenset(
    {
        MCPAuth.none,
        MCPAuth.api_key,
        MCPAuth.bearer_token,
        MCPAuth.basic,
        MCPAuth.authorization,
        MCPAuth.token,
        MCPAuth.aws_sigv4,
    }
)
_JSON_OBJECT: Final = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue])


def subscription_id(principal: str, url: str, name: str, arguments: JsonObject) -> str:
    material: Final = json.dumps([principal, url, name, arguments], sort_keys=True, separators=(",", ":"))
    return "sub_" + hashlib.sha256(material.encode()).hexdigest()[:32]


def callback_url(settings: EventsSettings, subscription: str) -> str:
    return f"{(settings.public_base_url or '').rstrip('/')}/mcp/events/webhooks/{subscription}"


def credential_outlives_request(server: MCPServer) -> bool:
    if server.extra_headers or server.oauth_passthrough or server.delegate_auth_to_upstream or server.is_byok:
        return False
    if server.auth_type is None or server.auth_type in _STORED_CREDENTIALS:
        return True
    return server.auth_type == MCPAuth.oauth2 and server.has_client_credentials


def granted_ttl_ms(params: SubscribeEventsParams, server: MCPServer) -> int | None:
    requested_no_expiry: Final = "ttl_ms" in params.model_fields_set and params.ttl_ms is None
    if not credential_outlives_request(server):
        return min(params.ttl_ms or REQUEST_SCOPED_TTL_MS, REQUEST_SCOPED_TTL_MS)
    if requested_no_expiry:
        return None
    return min(max(params.ttl_ms or DEFAULT_TTL_MS, MIN_TTL_MS), MAX_TTL_MS)


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if value is not None else None


def _parse_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed: Final = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def client_grant(now: datetime, ttl_ms: int | None, upstream: datetime | None) -> datetime | None:
    local: Final = now + timedelta(milliseconds=ttl_ms) if ttl_ms is not None else None
    candidates: Final = tuple(item for item in (local, upstream) if item is not None)
    return min(candidates) if candidates else None


async def verify_callback(url: str, secret: bytes, subscription: str) -> CallbackFailure | None:
    challenge: Final = secrets.token_urlsafe(24)
    body: Final = VerificationEnvelope(challenge=challenge).model_dump_json().encode()
    headers: Final = signed_headers(secret, new_webhook_id("verification"), int(time.time()), body, subscription)
    response: Final = await post_callback(url, body, headers)
    if isinstance(response, Error):
        return response.error
    if not 200 <= response.ok.status < 300:
        return "challenge_failed"
    try:
        echoed: Final = _JSON_OBJECT.validate_json(response.ok.body).get("challenge")
    except ValidationError:
        return "challenge_failed"
    return None if isinstance(echoed, str) and hmac.compare_digest(echoed, challenge) else "challenge_failed"


def delivery_status(row: SubscriptionRow) -> DeliveryStatus:
    return DeliveryStatus(
        active=row.status == "active",
        last_delivery_at=_iso(row.last_delivery_at),
        last_error=row.last_error,
        failed_since=_iso(row.failed_since),
    )


def _upstream_failure(error: MCPError) -> EventsFailure:
    if error.error.code == EventsErrorCode.CALLBACK_ENDPOINT_ERROR:
        return internal_error("The upstream server could not verify the gateway webhook endpoint")
    known: Final = error.error.code in {member.value for member in EventsErrorCode}
    return EventsFailure(
        EventsErrorCode(error.error.code) if known else EventsErrorCode.INTERNAL_ERROR, error.error.message
    )


def _upstream_params(
    prepared: PreparedEvent,
    params: SubscribeEventsParams,
    url: str,
    secret: str,
    cursor: str | None,
    ttl_ms: int | None,
) -> JsonObject:
    return {
        "name": prepared.resolved.upstream_name,
        "arguments": dict(params.arguments),
        "delivery": {"mode": "webhook", "url": url, "secret": secret},
        "cursor": cursor,
        **({"maxAgeMs": params.max_age_ms} if params.max_age_ms is not None else {}),
        "ttlMs": ttl_ms,
    }


async def subscribe(
    context: OperationContext,
    params: SubscribeEventsParams,
    headers: Mapping[str, str] | None,
    settings: EventsSettings,
    store: EventStore | None = None,
) -> Result[dict[str, object], EventsFailure]:
    if settings.public_base_url is None:
        return Error(unsupported("webhook"))
    principal: Final = principal_of(context.user_api_key_auth)
    if principal is None:
        return Error(forbidden("Webhook subscriptions require an authenticated caller"))
    if not is_https_callback(params.delivery.url):
        return Error(invalid_params("delivery.url must be an https URL"))
    client_secret: Final = parse_secret(params.delivery.secret)
    if client_secret is None:
        return Error(invalid_params("delivery.secret must be whsec_ followed by base64 of 24 to 64 bytes"))
    events_store: Final = store or default_store()
    if events_store is None:
        return Error(unsupported("webhook"))
    prepared: Final = await admit(context, params.name, params.arguments, "webhook", headers, settings)
    if isinstance(prepared, Error):
        return prepared
    sid: Final = subscription_id(principal, params.delivery.url, params.name, params.arguments)
    binding: Final = binding_for(context, prepared.ok, params.arguments)
    opened: Final = open_cursor(params.cursor, binding, int(time.time()))
    if isinstance(opened, Error):
        return opened
    existing: Final = await events_store.get(sid)
    fresh: Final = existing is None or existing.status == "terminated"
    if fresh:
        live: Final = await events_store.count_live(principal)
        if live >= settings.max_subscriptions_per_key:
            return Error(exhausted("subscriptions", settings.max_subscriptions_per_key))
        if not await events_store.verified(principal, params.delivery.url):
            failure = await verify_callback(params.delivery.url, client_secret, sid)
            if failure is not None:
                return Error(callback_error(failure))
    now: Final = datetime.now(timezone.utc)
    caller: Final = context.user_api_key_auth
    server: Final = prepared.ok.resolved.target.server
    base: Final = (
        SubscriptionRow(
            id=sid,
            principal=principal,
            key_hash=token_hash(caller.api_key) if caller is not None and caller.api_key else None,
            user_id=caller.user_id if caller is not None else None,
            team_id=caller.team_id if caller is not None else None,
            server_id=server.server_id,
            event_name=prepared.ok.resolved.upstream_name,
            arguments=params.arguments,
            client_url=params.delivery.url,
            client_secret=params.delivery.secret,
            upstream_secret=new_secret(),
            verified_at=now,
        )
        if fresh or existing is None
        else existing.model_copy(update={"client_secret": params.delivery.secret})
    )
    if fresh:
        await events_store.save(base)
    ttl_ms: Final = granted_ttl_ms(params, server)
    try:
        raw: Final = await call(
            prepared.ok.upstream.client,
            "events/subscribe",
            _upstream_params(
                prepared.ok, params, callback_url(settings, sid), base.upstream_secret, opened.ok.upstream, ttl_ms
            ),
            routing_headers(prepared.ok.resolved.upstream_name, params.arguments, prepared.ok.descriptor),
        )
        upstream: Final = SubscribeEventsResult.model_validate(raw)
    except (MCPError, ValidationError) as error:
        if fresh:
            await events_store.delete(sid)
        return Error(
            _upstream_failure(error)
            if isinstance(error, MCPError)
            else internal_error("Upstream returned an invalid events/subscribe result")
        )
    upstream_refresh: Final = _parse_time(upstream.refresh_before)
    refresh_before: Final = client_grant(now, ttl_ms, upstream_refresh)
    sealed: Final = seal_cursor(binding, upstream.cursor, int(time.time()))
    if isinstance(sealed, Error):
        return sealed
    active: Final = base.model_copy(
        update={
            "status": "active",
            "upstream_subscription_id": upstream.id,
            "upstream_refresh_before": upstream_refresh,
            "refresh_before": refresh_before,
            "cursor": sealed.ok,
            "consecutive_failures": 0,
        }
    )
    await events_store.save(active)
    if existing is not None and existing.status == "suspended":
        await events_store.enqueue(
            QueuedDelivery(
                delivery_id=new_webhook_id("gap"),
                subscription_id=sid,
                body=GapEnvelope(cursor=upstream.cursor).model_dump_json(),
                webhook_id=new_webhook_id("gap"),
                next_attempt_at=now,
            )
        )
    await record(
        EventActivity(
            call_type="mcp_event_subscribe",
            mode="webhook",
            server_id=server.server_id,
            event_name=prepared.ok.resolved.gateway_name,
            outcome="subscribed",
            caller=caller,
            subscription_id=sid,
        )
    )
    EventMetrics.count(server.server_id, prepared.ok.resolved.gateway_name, "subscribed")
    result: Final = SubscribeEventsResult(
        id=sid,
        refresh_before=_iso(refresh_before),
        cursor=sealed.ok,
        truncated=upstream.truncated or opened.ok.expired,
        delivery_status=delivery_status(existing) if existing is not None and not fresh else None,
    ).model_dump(by_alias=True, mode="json", exclude_none=True)
    return Ok({**result, "refreshBefore": _iso(refresh_before)})


async def unsubscribe(
    context: OperationContext,
    params: UnsubscribeEventsParams,
    headers: Mapping[str, str] | None,
    settings: EventsSettings,
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
    prepared: Final = await prepare_event(context, params.name)
    if isinstance(prepared, Ok) and row.upstream_subscription_id is not None:
        try:
            await call(
                prepared.ok.upstream.client,
                "events/unsubscribe",
                {
                    "name": row.event_name,
                    "arguments": dict(row.arguments),
                    "delivery": {"url": callback_url(settings, sid)},
                },
                routing_headers(row.event_name, row.arguments, prepared.ok.descriptor),
            )
        except MCPError:
            pass
    await events_store.delete(sid)
    await record(
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


async def end_upstream(row: SubscriptionRow, settings: EventsSettings) -> None:
    from litellm.proxy.mcp_server.events.catalog import descriptors
    from litellm.proxy.mcp_server.mcp_server_manager import global_mcp_server_manager
    from litellm.proxy.mcp_server.subscriptions import prepare_target

    server: Final = global_mcp_server_manager.get_mcp_server_by_id(row.server_id)
    if server is None or row.upstream_subscription_id is None or not credential_outlives_request(server):
        return
    try:
        target: Final = await prepare_target(OperationContext(_caller=None), server, (server,))
        descriptor: Final = next((item for item in await descriptors(target) if item.name == row.event_name), None)
        if descriptor is None:
            return
        await call(
            target.client,
            "events/unsubscribe",
            {
                "name": row.event_name,
                "arguments": dict(row.arguments),
                "delivery": {"url": callback_url(settings, row.id)},
            },
            routing_headers(row.event_name, row.arguments, descriptor),
        )
    except Exception:  # noqa: BLE001  # the upstream lapses at its own refreshBefore when unsubscribe fails
        return
