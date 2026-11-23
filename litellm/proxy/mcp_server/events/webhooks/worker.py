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
from litellm.proxy.mcp_server.events.webhooks.signing import parse_secret, signed_headers
from litellm.proxy.mcp_server.events.webhooks.store import (
    LIVE_STATUSES,
    EventStore,
    QueuedDelivery,
    SubscriptionRow,
    default_store,
)
from litellm.proxy.mcp_server.events.webhooks.termination import notify_termination, registered_server
from litellm.proxy.mcp_server.events.webhooks.verification import Poster
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok
from litellm.types.mcp_server.events import CallbackFailure
from litellm.types.mcp_server.mcp_server_manager import MCPServer

RETRY_DELAYS_SECONDS: Final = (30, 120, 480)
UNAVAILABLE_DELAY_SECONDS: Final = 30
CLAIM_BATCH: Final = 16
CONCURRENCY: Final = 8
LEASE_SECONDS: Final = 300
PROCESS_DEADLINE_SECONDS: Final = 60.0
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

    @classmethod
    async def run_forever(
        cls, settings: Callable[[], EventsSettings], store_for_batch: Callable[[], EventStore | None] = default_store
    ) -> None:
        next_housekeeping = 0.0  # rebind-ok: monotonic deadline of the next expiry sweep
        while True:
            store: Final = store_for_batch()
            if store is None:
                await asyncio.sleep(_IDLE_SECONDS)
            else:
                next_housekeeping = await cls(store, settings)._tick(next_housekeeping)

    async def _tick(self, next_housekeeping: float) -> float:
        """One delivery batch, a sweep when due, and an idle pause when nothing was due; returns the next sweep time."""
        handled: Final = await self._run_guarded()
        due: Final = time.monotonic() >= next_housekeeping
        if due:
            await self._expire_guarded()
        if not handled:
            await asyncio.sleep(_IDLE_SECONDS)
        return time.monotonic() + _HOUSEKEEPING_SECONDS if due else next_housekeeping

    async def _run_guarded(self) -> int:
        try:
            return await self.run_once()
        except Exception as error:  # noqa: BLE001  # the worker must survive a failed batch and retry it
            verbose_logger.warning("MCP event delivery batch failed: %s", error)
            return 0

    async def _expire_guarded(self) -> None:
        try:
            await self.expire()
        except Exception as error:  # noqa: BLE001  # a failed sweep is retried on the next one
            verbose_logger.warning("MCP event subscription sweep failed: %s", error)

    async def run_once(self) -> int:
        if not self._settings().enabled:
            return 0
        now: Final = self._clock()
        claimed: Final = await self._store.claim(now, now + timedelta(seconds=LEASE_SECONDS), CLAIM_BATCH)
        limiter: Final = anyio.CapacityLimiter(CONCURRENCY)

        async def bounded(delivery: QueuedDelivery) -> None:
            async with limiter:
                await self._process_isolated(delivery)

        async with anyio.create_task_group() as group:
            for delivery in claimed:
                group.start_soon(bounded, delivery)
        return len(claimed)

    async def _process_isolated(self, delivery: QueuedDelivery) -> None:
        """One delivery, bounded well inside its lease so no other replica can claim it mid-send; a failure is
        retried after the lease lapses instead of cancelling the rest of the batch."""
        try:
            with anyio.move_on_after(PROCESS_DEADLINE_SECONDS):
                await self.process(delivery)
        except Exception as error:  # noqa: BLE001  # one failing delivery must not cancel the batch
            verbose_logger.warning("MCP event delivery %s failed: %s", delivery.delivery_id, error)

    async def expire(self) -> None:
        now: Final = self._clock()
        for row in await self._store.expired(now):
            await self._store.end_if_expired(row.id, now)
        await self._store.purge_delivered(now - DELIVERED_RETENTION)
        for status in LIVE_STATUSES:
            EventMetrics.set_subscriptions(status, await self._store.count_status(status))

    async def process(self, delivery: QueuedDelivery) -> None:
        row: Final = await self._store.get(delivery.subscription_id)
        if row is None:
            await self._store.finish(delivery, self._clock())
            return
        if not self._deliverable(row, delivery):
            await self._store.settle(delivery, row.generation, self._clock(), "discarded")
            return
        authorization: Final = await authorize_row(row, self._runtime, self._load_caller)
        if isinstance(authorization, Revoked):
            if await self._store.settle(delivery, row.generation, self._clock(), "terminated"):
                await notify_termination(
                    row,
                    forbidden("Access revoked"),
                    self._store,
                    self._settings(),
                    self._runtime,
                    self._post,
                    self._lookup,
                )
            return
        if not isinstance(authorization, Authorized):
            await self._store.reschedule(
                delivery, delivery.attempt, self._clock() + timedelta(seconds=UNAVAILABLE_DELAY_SECONDS)
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
            await self._store.settle(delivery, row.generation, self._clock(), "discarded")
            return
        await self._send(context, outgoing)

    def _deliverable(self, row: SubscriptionRow, delivery: QueuedDelivery) -> bool:
        """Active and unexpired; a suspended subscription still gets its gap and terminated notices."""
        if row.status in ("terminated", "pending") or self._expired(row):
            return False
        return row.status != "suspended" or _is_control(delivery)

    def _expired(self, row: SubscriptionRow) -> bool:
        return row.refresh_before is not None and row.refresh_before <= self._clock()

    async def _outgoing(self, context: DeliveryContext) -> Outgoing | None:
        if context.delivery.prepared:
            return prepared_outgoing(context.delivery)
        built: Final = await build_outgoing(context, self._runtime)
        if built is not None:
            await self._store.prepare(context.delivery, built.body.decode(), built.webhook_id)
        return built

    async def _send(self, context: DeliveryContext, outgoing: Outgoing) -> None:
        row: Final = context.row
        secret: Final = parse_secret(row.client_secret)
        if secret is None:
            await self._store.reschedule(
                context.delivery, context.delivery.attempt, self._clock() + timedelta(seconds=UNAVAILABLE_DELAY_SECONDS)
            )
            return
        started: Final = time.monotonic()
        attributes: Final = {"mcp.event.name": context.gateway_name, "mcp.subscription.id": row.id}
        with _TRACER.start_as_current_span("mcp.event.deliver", attributes=attributes) as span:
            response: Final = await self._post(
                row.client_url,
                outgoing.body,
                signed_headers(secret, outgoing.webhook_id, int(self._clock().timestamp()), outgoing.body, row.id),
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
        accepted: Final = status is not None and 200 <= status < 300
        outcome: Final = "terminated" if outgoing.terminal else "delivered" if accepted else "discarded"
        if not await self._store.settle(context.delivery, row.generation, self._clock(), outcome):
            return
        if accepted and not outgoing.terminal:
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
            await self._store.reschedule(delivery, attempt, self._clock() + delay)
            return
        await self._store.settle(
            delivery,
            row.generation,
            self._clock(),
            "terminated" if outgoing.terminal else failure,
            gap=outgoing.event_id is not None,
        )


def _is_control(delivery: QueuedDelivery) -> bool:
    """A gap or terminated notice, which a suspended subscription still receives."""
    return envelope_type(delivery.body) in ("gap", "terminated")


class _WorkerHandle:
    def __init__(self) -> None:
        self.task: asyncio.Task[None] | None = None

    def start(
        self,
        settings: Callable[[], EventsSettings] = events_settings,
        store_for_batch: Callable[[], EventStore | None] = default_store,
    ) -> None:
        if (self.task is not None and not self.task.done()) or not settings().enabled:
            return
        self.task = asyncio.get_running_loop().create_task(DeliveryWorker.run_forever(settings, store_for_batch))

    async def stop(self) -> None:
        task: Final = self.task
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if self.task is task:
                self.task = None


_handle: Final = _WorkerHandle()


def start_delivery_worker() -> None:
    _handle.start()


async def stop_delivery_worker() -> None:
    await _handle.stop()
