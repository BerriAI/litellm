import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.events.access import event_grant, grant_allows
from litellm.proxy.mcp_server.events.catalog import allowed_servers, event_servers, gateway_name
from litellm.proxy.mcp_server.events.config import EventsSettings, events_settings, server_allows_event
from litellm.proxy.mcp_server.events.cursors import CursorBinding, arguments_digest, seal_cursor
from litellm.proxy.mcp_server.events.errors import EventsFailure, forbidden
from litellm.proxy.mcp_server.events.guardrails import Blocked, screen_event
from litellm.proxy.mcp_server.events.observability import EventActivity, EventMetrics, record
from litellm.proxy.mcp_server.events.relay import event_budget
from litellm.proxy.mcp_server.events.webhooks.delivery import failure_for_status, post_callback
from litellm.proxy.mcp_server.events.webhooks.signing import new_webhook_id, parse_secret, signed_headers
from litellm.proxy.mcp_server.events.webhooks.store import EventStore, QueuedDelivery, SubscriptionRow, default_store
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok
from litellm.types.mcp_server.events import EventOccurrence, GapEnvelope, TerminatedEnvelope
from litellm.types.mcp_server.mcp_server_manager import MCPServerEventsConfig

RETRY_DELAYS_SECONDS: Final = (30, 120, 480)
SUSPEND_AFTER_FAILURES: Final = 20
_CLAIM_BATCH: Final = 32
_LEASE_SECONDS: Final = 30
_IDLE_SECONDS: Final = 1.0
_HOUSEKEEPING_SECONDS: Final = 30.0
_JSON_OBJECT: Final = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue])

StoredCaller = Callable[[SubscriptionRow], Awaitable[UserAPIKeyAuth | None]]
Poster = Callable[..., Awaitable[object]]


async def stored_caller(row: SubscriptionRow) -> UserAPIKeyAuth | None:
    from litellm.proxy.auth.auth_checks import get_key_object
    from litellm.proxy.proxy_server import master_key, prisma_client, proxy_logging_obj, user_api_key_cache
    from litellm.proxy.utils import hash_token

    if row.key_hash is None:
        return UserAPIKeyAuth(user_id=row.user_id, team_id=row.team_id) if row.user_id else None
    if isinstance(master_key, str) and row.key_hash == hash_token(master_key):
        from litellm.proxy._types import LitellmUserRoles

        return UserAPIKeyAuth(api_key=row.key_hash, user_role=LitellmUserRoles.PROXY_ADMIN)
    try:
        key: Final = await get_key_object(
            hashed_token=row.key_hash,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
            check_db_only=True,
        )
    except Exception:  # noqa: BLE001  # a key that can no longer be loaded ends its subscriptions
        return None
    expires: Final = key.expires
    expired: Final = isinstance(expires, datetime) and expires.astimezone(timezone.utc) <= datetime.now(timezone.utc)
    return None if key.blocked or expired else key


async def authorize_row(row: SubscriptionRow, load: StoredCaller = stored_caller) -> UserAPIKeyAuth | None:
    caller: Final = await load(row)
    if caller is None:
        return None
    context: Final = OperationContext(_caller=caller)
    server: Final = next((item for item in event_servers(await allowed_servers(context)) if item.server.server_id == row.server_id), None)
    if server is None or not server_allows_event(server.config, row.event_name):
        return None
    return caller if grant_allows(await event_grant(row.server_id, caller), row.event_name) else None


@dataclass(frozen=True, slots=True)
class Outgoing:
    body: bytes
    webhook_id: str
    terminal: bool
    event_id: str | None


class DeliveryWorker:
    def __init__(
        self,
        store: EventStore,
        settings: Callable[[], EventsSettings] = events_settings,
        authorize: Callable[[SubscriptionRow], Awaitable[UserAPIKeyAuth | None]] = authorize_row,
        post: Poster = post_callback,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._store = store
        self._settings = settings
        self._authorize = authorize
        self._post = post
        self._clock = clock

    async def run_forever(self) -> None:
        last_housekeeping = 0.0  # rebind-ok: monotonic time of the previous expiry sweep
        while True:
            try:
                handled = await self.run_once()
                if time.monotonic() - last_housekeeping >= _HOUSEKEEPING_SECONDS:
                    await self.expire()
                    last_housekeeping = time.monotonic()
            except Exception as error:  # noqa: BLE001  # the worker must survive a failed batch and retry it
                verbose_logger.warning("MCP event delivery batch failed: %s", error)
                handled = 0
            if not handled:
                await asyncio.sleep(_IDLE_SECONDS)

    async def run_once(self) -> int:
        now: Final = self._clock()
        claimed: Final = await self._store.claim(now, now + timedelta(seconds=_LEASE_SECONDS), _CLAIM_BATCH)
        for delivery in claimed:
            await self.process(delivery)
        return len(claimed)

    async def expire(self) -> None:
        for row in await self._store.expired(self._clock()):
            await self._store.save(row.model_copy(update={"status": "terminated"}))
            await self._store.drop_deliveries(row.id)

    async def process(self, delivery: QueuedDelivery) -> None:
        row: Final = await self._store.get(delivery.subscription_id)
        if row is None or row.status in ("terminated", "pending"):
            await self._store.finish(delivery.delivery_id)
            return
        caller: Final = await self._authorize(row)
        if caller is None:
            await self._store.finish(delivery.delivery_id)
            await self.terminate(row, forbidden("Access revoked"))
            return
        outgoing: Final = await self._prepare(row, delivery, caller)
        if outgoing is None:
            await self._store.finish(delivery.delivery_id)
            return
        await self._send(row, delivery, outgoing, caller)

    async def _prepare(self, row: SubscriptionRow, delivery: QueuedDelivery, caller: UserAPIKeyAuth) -> Outgoing | None:
        try:
            envelope: Final = _JSON_OBJECT.validate_json(delivery.body)
        except ValidationError:
            return None
        binding: Final = CursorBinding(row.server_id, row.event_name, arguments_digest(row.arguments), row.principal)
        kind: Final = envelope.get("type")
        if kind == "terminated":
            terminated: Final = TerminatedEnvelope.model_validate(envelope)
            return Outgoing(terminated.model_dump_json().encode(), new_webhook_id("terminated"), True, None)
        if kind == "gap":
            return self._gap(row, binding, envelope.get("cursor"))
        try:
            occurrence: Final = EventOccurrence.model_validate(envelope)
        except ValidationError:
            return None
        return await self._event(row, delivery, binding, occurrence, caller)

    def _gap(self, row: SubscriptionRow, binding: CursorBinding, upstream: object) -> Outgoing:
        sealed: Final = seal_cursor(binding, upstream if isinstance(upstream, str) else None, int(time.time()))
        cursor: Final = sealed.ok if isinstance(sealed, Ok) and sealed.ok is not None else row.cursor
        return Outgoing(GapEnvelope(cursor=cursor).model_dump_json().encode(), new_webhook_id("gap"), False, None)

    async def _event(
        self,
        row: SubscriptionRow,
        delivery: QueuedDelivery,
        binding: CursorBinding,
        occurrence: EventOccurrence,
        caller: UserAPIKeyAuth,
    ) -> Outgoing:
        from litellm.proxy.mcp_server.mcp_server_manager import global_mcp_server_manager

        server: Final = global_mcp_server_manager.get_mcp_server_by_id(row.server_id)
        name: Final = gateway_name(server, row.event_name) if server is not None else row.event_name
        if not event_budget.take(row.principal, self._settings().max_events_per_minute):
            EventMetrics.count(row.server_id, name, "throttled")
            return self._gap(row, binding, occurrence.cursor)
        screened: Final = await screen_event(
            occurrence, server_name=server.name if server is not None else row.server_id, gateway_name=name,
            subscription_id=row.id, caller=caller,
        )
        if isinstance(screened, Blocked):
            EventMetrics.count(row.server_id, name, "blocked")
            return self._gap(row, binding, occurrence.cursor)
        sealed: Final = seal_cursor(binding, occurrence.cursor, int(time.time()))
        cursor: Final = sealed.ok if isinstance(sealed, Ok) else None
        legacy: Final = server is not None and isinstance(server.events, MCPServerEventsConfig) and server.events.legacy_webhook_id_equals_event_id
        body: Final = occurrence.model_copy(
            update={"name": name, "data": screened.data, "cursor": cursor, "meta": None}
        ).model_dump_json(by_alias=True, exclude_none=True)
        return Outgoing(body.encode(), occurrence.event_id if legacy else delivery.webhook_id, False, occurrence.event_id)

    async def _send(self, row: SubscriptionRow, delivery: QueuedDelivery, outgoing: Outgoing, caller: UserAPIKeyAuth) -> None:
        secret: Final = parse_secret(row.client_secret)
        if secret is None:
            await self._store.finish(delivery.delivery_id)
            return
        started: Final = time.monotonic()
        response: Final = await self._post(
            row.client_url,
            outgoing.body,
            signed_headers(secret, outgoing.webhook_id, int(time.time()), outgoing.body, row.id),
        )
        status: Final = response.ok.status if isinstance(response, Ok) else None
        failure: Final = response.error if isinstance(response, Error) else failure_for_status(status or 0)
        if failure is None or status in (410, 413):
            await self._delivered(row, delivery, outgoing, caller, time.monotonic() - started, status)
            return
        await self._failed(row, delivery, failure)

    async def _delivered(
        self,
        row: SubscriptionRow,
        delivery: QueuedDelivery,
        outgoing: Outgoing,
        caller: UserAPIKeyAuth,
        elapsed: float,
        status: int | None,
    ) -> None:
        await self._store.finish(delivery.delivery_id)
        accepted: Final = status is not None and 200 <= status < 300
        updated: Final = (
            row.model_copy(update={"last_delivery_at": self._clock(), "consecutive_failures": 0, "failed_since": None, "last_error": None})
            if accepted
            else row
        )
        if outgoing.terminal:
            await self._store.save(updated.model_copy(update={"status": "terminated"}))
            await self._store.drop_deliveries(row.id)
        elif accepted:
            await self._store.save(updated)
        if accepted:
            EventMetrics.observe_latency(elapsed)
        await record(
            EventActivity(
                call_type="mcp_event_delivery",
                mode="webhook",
                server_id=row.server_id,
                event_name=row.event_name,
                outcome="delivered" if accepted else "failed",
                caller=caller,
                subscription_id=row.id,
                event_ids=(outgoing.event_id,) if outgoing.event_id else (),
                attempts=delivery.attempt + 1,
            )
        )

    async def _failed(self, row: SubscriptionRow, delivery: QueuedDelivery, failure: str) -> None:
        attempt: Final = delivery.attempt + 1
        if attempt <= len(RETRY_DELAYS_SECONDS):
            await self._store.reschedule(
                delivery.delivery_id, attempt, self._clock() + timedelta(seconds=RETRY_DELAYS_SECONDS[attempt - 1])
            )
            return
        await self._store.finish(delivery.delivery_id)
        failures: Final = row.consecutive_failures + 1
        await self._store.save(
            row.model_copy(
                update={
                    "consecutive_failures": failures,
                    "last_error": failure,
                    "failed_since": row.failed_since or self._clock(),
                    "status": "suspended" if failures >= SUSPEND_AFTER_FAILURES else row.status,
                }
            )
        )

    async def terminate(self, row: SubscriptionRow, failure: EventsFailure) -> None:
        secret: Final = parse_secret(row.client_secret)
        if secret is not None:
            body: Final = TerminatedEnvelope(
                error={"code": int(failure.code), "message": failure.message, "data": dict(failure.data or {})}
            ).model_dump_json().encode()
            await self._post(row.client_url, body, signed_headers(secret, new_webhook_id("terminated"), int(time.time()), body, row.id))
        await self._store.save(row.model_copy(update={"status": "terminated"}))
        await self._store.drop_deliveries(row.id)
        from litellm.proxy.mcp_server.events.webhooks.subscribe import end_upstream

        await end_upstream(row, self._settings())


_worker_task: asyncio.Task[None] | None = None


def start_delivery_worker() -> None:
    global _worker_task  # noqa: PLW0603  # one worker task per process, started at proxy startup
    if _worker_task is not None and not _worker_task.done():
        return
    store: Final = default_store()
    if store is None or not events_settings().enabled:
        return
    _worker_task = asyncio.get_running_loop().create_task(DeliveryWorker(store).run_forever())
