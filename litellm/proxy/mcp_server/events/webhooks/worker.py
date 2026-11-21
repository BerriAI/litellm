import asyncio
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Final

import anyio
from opentelemetry import trace

from litellm._logging import verbose_logger
from litellm.proxy.mcp_server.events.catalog import gateway_name
from litellm.proxy.mcp_server.events.config import EventsSettings, events_settings
from litellm.proxy.mcp_server.events.errors import forbidden
from litellm.proxy.mcp_server.events.observability import EventActivity, EventMetrics
from litellm.proxy.mcp_server.events.runtime import DEFAULT_RUNTIME, EventsRuntime
from litellm.proxy.mcp_server.events.webhooks.delivery import failure_for_status, post_callback
from litellm.proxy.mcp_server.events.webhooks.outgoing import (
    DeliveryContext,
    Outgoing,
    build_outgoing,
    envelope_type,
    prepared_outgoing,
)
from litellm.proxy.mcp_server.events.webhooks.reauth import (
    Authorized,
    CallerLoader,
    Revoked,
    authorize_row,
    stored_caller,
)
from litellm.proxy.mcp_server.events.webhooks.signing import new_webhook_id, parse_secret, signed_headers
from litellm.proxy.mcp_server.events.webhooks.store import (
    LIVE_STATUSES,
    EventStore,
    QueuedDelivery,
    SubscriptionRow,
    default_store,
)
from litellm.proxy.mcp_server.events.webhooks.termination import registered_server, terminate
from litellm.proxy.mcp_server.events.webhooks.verification import Poster
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok
from litellm.types.mcp_server.events import CallbackFailure, GapEnvelope
from litellm.types.mcp_server.mcp_server_manager import MCPServer

RETRY_DELAYS_SECONDS: Final = (30, 120, 480)
UNAVAILABLE_DELAY_SECONDS: Final = 30
SUSPEND_AFTER_FAILURES: Final = 20
CLAIM_BATCH: Final = 16
CONCURRENCY: Final = 8
LEASE_SECONDS: Final = 60
DELIVERED_RETENTION: Final = timedelta(minutes=10)
_IDLE_SECONDS: Final = 1.0
_HOUSEKEEPING_SECONDS: Final = 30.0
_TRACER: Final = trace.get_tracer("litellm.mcp.events")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DeliveryWorker:
    def __init__(
        self,
        store: EventStore,
        settings: Callable[[], EventsSettings] = events_settings,
        runtime: EventsRuntime = DEFAULT_RUNTIME,
        load_caller: CallerLoader = stored_caller,
        post: Poster = post_callback,
        clock: Callable[[], datetime] = _utc_now,
        lookup: Callable[[str], MCPServer | None] = registered_server,
    ) -> None:
        self._store = store
        self._settings = settings
        self._runtime = runtime
        self._load_caller = load_caller
        self._post = post
        self._clock = clock
        self._lookup = lookup

    async def run_forever(self) -> None:
        next_housekeeping = 0.0  # rebind-ok: monotonic deadline of the next expiry sweep
        while True:
            handled = 0
            try:
                handled = await self.run_once()
                if time.monotonic() >= next_housekeeping:
                    await self.expire()
                    next_housekeeping = time.monotonic() + _HOUSEKEEPING_SECONDS
            except Exception as error:  # noqa: BLE001  # the worker must survive a failed batch and retry it
                verbose_logger.warning("MCP event delivery batch failed: %s", error)
            if not handled:
                await asyncio.sleep(_IDLE_SECONDS)

    async def run_once(self) -> int:
        if not self._settings().enabled:
            return 0
        now: Final = self._clock()
        claimed: Final = await self._store.claim(now, now + timedelta(seconds=LEASE_SECONDS), CLAIM_BATCH)
        limiter: Final = anyio.CapacityLimiter(CONCURRENCY)

        async def bounded(delivery: QueuedDelivery) -> None:
            async with limiter:
                await self.process(delivery)

        async with anyio.create_task_group() as group:
            for delivery in claimed:
                group.start_soon(bounded, delivery)
        return len(claimed)

    async def expire(self) -> None:
        now: Final = self._clock()
        for row in await self._store.expired(now):
            await self._store.terminate(row.id)
            await self._store.drop_deliveries(row.id)
        await self._store.purge_delivered(now - DELIVERED_RETENTION)
        for status in LIVE_STATUSES:
            EventMetrics.set_subscriptions(status, await self._store.count_status(status))

    async def process(self, delivery: QueuedDelivery) -> None:
        row: Final = await self._store.get(delivery.subscription_id)
        paused: Final = row is not None and row.status == "suspended" and not _is_control(delivery)
        if row is None or row.status in ("terminated", "pending") or self._expired(row) or paused:
            await self._store.finish(delivery.delivery_id, self._clock())
            return
        authorization: Final = await authorize_row(row, self._runtime, self._load_caller)
        if isinstance(authorization, Revoked):
            await self._store.finish(delivery.delivery_id, self._clock())
            await terminate(row, forbidden("Access revoked"), self._store, self._settings(), self._runtime, self._post)
            return
        if not isinstance(authorization, Authorized):
            await self._store.reschedule(
                delivery.delivery_id, delivery.attempt, self._clock() + timedelta(seconds=UNAVAILABLE_DELAY_SECONDS)
            )
            return
        server: Final = self._lookup(row.server_id)
        context: Final = DeliveryContext(
            row=row,
            delivery=delivery,
            caller=authorization.caller,
            server=server,
            gateway_name=gateway_name(server, row.event_name) if server is not None else row.event_name,
            max_events_per_minute=self._settings().max_events_per_minute,
        )
        outgoing: Final = await self._outgoing(context)
        if outgoing is None:
            await self._store.finish(delivery.delivery_id, self._clock())
            return
        await self._send(context, outgoing)

    def _expired(self, row: SubscriptionRow) -> bool:
        return row.refresh_before is not None and row.refresh_before <= self._clock()

    async def _outgoing(self, context: DeliveryContext) -> Outgoing | None:
        if context.delivery.prepared:
            return prepared_outgoing(context.delivery)
        built: Final = await build_outgoing(context, self._runtime)
        if built is not None:
            await self._store.prepare(context.delivery.delivery_id, built.body.decode(), built.webhook_id)
        return built

    async def _send(self, context: DeliveryContext, outgoing: Outgoing) -> None:
        row: Final = context.row
        secret: Final = parse_secret(row.client_secret)
        if secret is None:
            await self._store.finish(context.delivery.delivery_id, self._clock())
            return
        started: Final = time.monotonic()
        attributes: Final = {"mcp.event.name": context.gateway_name, "mcp.subscription.id": row.id}
        with _TRACER.start_as_current_span("mcp.event.deliver", attributes=attributes) as span:
            response: Final = await self._post(
                row.client_url,
                outgoing.body,
                signed_headers(secret, outgoing.webhook_id, int(time.time()), outgoing.body, row.id),
            )
            status: Final = response.ok.status if isinstance(response, Ok) else None
            span.set_attribute("http.response.status_code", status or 0)
        failure: Final = response.error if isinstance(response, Error) else failure_for_status(status or 0)
        if failure is None or status in (410, 413):
            await self._settled(context, outgoing, status, time.monotonic() - started)
            return
        await self._failed(row, context.delivery, outgoing, failure)

    async def _settled(self, context: DeliveryContext, outgoing: Outgoing, status: int | None, elapsed: float) -> None:
        row: Final = context.row
        await self._store.finish(context.delivery.delivery_id, self._clock())
        accepted: Final = status is not None and 200 <= status < 300
        if outgoing.terminal:
            await self._store.terminate(row.id)
            await self._store.drop_deliveries(row.id)
        elif accepted:
            await self._store.mark_delivered(row.id, self._clock())
            EventMetrics.observe_latency(elapsed)
        await self._runtime.record(
            EventActivity(
                call_type="mcp_event_delivery",
                mode="webhook",
                server_id=row.server_id,
                event_name=context.gateway_name,
                outcome="delivered" if accepted else "failed",
                caller=context.caller,
                subscription_id=row.id,
                event_ids=(outgoing.event_id,) if outgoing.event_id else (),
                attempts=context.delivery.attempt + 1,
            )
        )

    async def _failed(
        self, row: SubscriptionRow, delivery: QueuedDelivery, outgoing: Outgoing, failure: CallbackFailure
    ) -> None:
        attempt: Final = delivery.attempt + 1
        if attempt <= len(RETRY_DELAYS_SECONDS):
            delay: Final = timedelta(seconds=RETRY_DELAYS_SECONDS[attempt - 1])
            await self._store.reschedule(delivery.delivery_id, attempt, self._clock() + delay)
            return
        if outgoing.event_id is not None:
            await self._report_abandoned(row, outgoing)
        await self._store.finish(delivery.delivery_id, self._clock())
        await self._store.mark_failed(row.id, failure, self._clock(), SUSPEND_AFTER_FAILURES)

    async def _report_abandoned(self, row: SubscriptionRow, outgoing: Outgoing) -> None:
        """An event given up after its retries becomes a gap, so the client knows to resume from its cursor."""
        gap_id: Final = new_webhook_id("gap")
        await self._store.enqueue(
            QueuedDelivery(
                delivery_id=gap_id,
                subscription_id=row.id,
                body=GapEnvelope(cursor=outgoing.cursor or row.cursor).model_dump_json(),
                webhook_id=gap_id,
                next_attempt_at=self._clock(),
                prepared=True,
            )
        )


def _is_control(delivery: QueuedDelivery) -> bool:
    """A gap or terminated notice, which a suspended subscription still receives."""
    return envelope_type(delivery.body) in ("gap", "terminated")


class _WorkerHandle:
    def __init__(self) -> None:
        self.task: asyncio.Task[None] | None = None

    def start(self, settings: Callable[[], EventsSettings] = events_settings) -> None:
        if (self.task is not None and not self.task.done()) or not settings().enabled:
            return
        store: Final = default_store()
        if store is None:
            return
        self.task = asyncio.get_running_loop().create_task(DeliveryWorker(store, settings).run_forever())


_handle: Final = _WorkerHandle()


def start_delivery_worker() -> None:
    _handle.start()
