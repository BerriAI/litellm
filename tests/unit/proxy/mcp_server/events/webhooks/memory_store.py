import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

from litellm.proxy.mcp_server.events.webhooks.store import (
    LIVE_STATUSES,
    Activation,
    QueuedDelivery,
    SubscriptionQuery,
    SubscriptionRow,
    pending_row,
)
from litellm.types.mcp_server.events import CallbackFailure


@dataclass
class MemoryStore:
    rows: dict[str, SubscriptionRow] = field(default_factory=dict)  # mutable-ok: in-memory table
    queue: dict[str, QueuedDelivery] = field(default_factory=dict)  # mutable-ok: in-memory queue
    finished: list[str] = field(default_factory=list)  # mutable-ok: records completed deliveries
    delivered: dict[str, datetime] = field(default_factory=dict)  # mutable-ok: delivered tombstones
    leases: dict[str, datetime] = field(default_factory=dict)  # mutable-ok: claimed-until per delivery
    fail_enqueue: bool = False

    async def get(self, subscription_id: str) -> SubscriptionRow | None:
        return self.rows.get(subscription_id)

    async def save(self, row: SubscriptionRow) -> None:
        """Test setup only: seed a row as it would already sit in the table."""
        self.rows[row.id] = row

    async def reserve(self, row: SubscriptionRow) -> bool:
        current = self.rows.get(row.id)
        if current is not None and current.status != "terminated":
            return False
        self.rows[row.id] = pending_row(row)
        return True

    async def discard_pending(self, subscription_id: str) -> None:
        current = self.rows.get(subscription_id)
        if current is not None and current.status == "pending":
            self.rows.pop(subscription_id)

    def _update_live(self, subscription_id: str, **fields: object) -> bool:  # noqa: ANN003  # test double mirrors a guarded UPDATE
        current = self.rows.get(subscription_id)
        if current is None or current.status == "terminated":
            return False
        self.rows[subscription_id] = current.model_copy(update=fields)
        return True

    async def activate(self, subscription_id: str, activation: Activation) -> bool:
        return self._update_live(subscription_id, status="active", consecutive_failures=0, **activation.model_dump())

    async def mark_delivered(self, subscription_id: str, at: datetime) -> None:
        self._update_live(
            subscription_id, last_delivery_at=at, consecutive_failures=0, failed_since=None, last_error=None
        )

    async def mark_failed(
        self, subscription_id: str, failure: CallbackFailure, at: datetime, suspend_after: int
    ) -> None:
        current = self.rows.get(subscription_id)
        if current is None or current.status == "terminated":
            return
        failures = current.consecutive_failures + 1
        self._update_live(
            subscription_id,
            consecutive_failures=failures,
            last_error=failure,
            failed_since=current.failed_since or at,
            status="suspended" if current.status == "active" and failures >= suspend_after else current.status,
        )

    async def end_if_current(self, subscription_id: str, upstream_subscription_id: str | None) -> bool:
        current: Final = self.rows.get(subscription_id)
        if (
            current is None
            or current.status == "terminated"
            or current.upstream_subscription_id != upstream_subscription_id
        ):
            return False
        self.rows[subscription_id] = current.model_copy(update={"status": "terminated"})
        return True

    async def delete_ended(self, subscription_id: str, upstream_subscription_id: str | None) -> None:
        current: Final = self.rows.get(subscription_id)
        if (
            current is not None
            and current.status == "terminated"
            and current.upstream_subscription_id == upstream_subscription_id
        ):
            self.rows.pop(subscription_id)

    async def terminate(self, subscription_id: str) -> None:
        current = self.rows.get(subscription_id)
        if current is not None:
            self.rows[subscription_id] = current.model_copy(update={"status": "terminated"})

    async def delete(self, subscription_id: str) -> None:
        self.rows.pop(subscription_id, None)
        await self.drop_deliveries(subscription_id)

    async def count_live(self, principal: str) -> int:
        return sum(1 for row in self.rows.values() if row.principal == principal and row.status in LIVE_STATUSES)

    async def count_status(self, status: str) -> int:
        return sum(1 for row in self.rows.values() if row.status == status)

    async def verified(self, principal: str, client_url: str) -> bool:
        return any(
            row.principal == principal and row.client_url == client_url and row.verified_at is not None
            for row in self.rows.values()
        )

    async def find(self, query: SubscriptionQuery) -> tuple[SubscriptionRow, ...]:
        return tuple(
            row
            for row in self.rows.values()
            if (query.principal is None or row.principal == query.principal)
            and (query.key_hash is None or row.key_hash == query.key_hash)
            and (query.server_id is None or row.server_id == query.server_id)
            and (query.status is None or row.status == query.status)
            and (not query.live_only or row.status in LIVE_STATUSES)
        )

    async def due_for_upstream_refresh(self, before: datetime) -> tuple[SubscriptionRow, ...]:
        return ()

    async def expired(self, now: datetime) -> tuple[SubscriptionRow, ...]:
        return tuple(
            row
            for row in self.rows.values()
            if row.status in LIVE_STATUSES and row.refresh_before is not None and row.refresh_before <= now
        )

    async def enqueue(self, delivery: QueuedDelivery) -> bool:
        if self.fail_enqueue:
            raise ConnectionError("database unavailable")
        if delivery.delivery_id in self.queue or delivery.delivery_id in self.delivered:
            return False
        self.queue[delivery.delivery_id] = delivery
        return True

    async def claim(self, now: datetime, lease_until: datetime, limit: int) -> tuple[QueuedDelivery, ...]:
        due: Final = [
            item
            for item in self.queue.values()
            if item.next_attempt_at <= now and self.leases.get(item.delivery_id, now) <= now
        ][:limit]
        claimed: Final = tuple(item.model_copy(update={"claim_id": uuid.uuid4().hex}) for item in due)
        for item in claimed:
            self.queue[item.delivery_id] = item
            self.leases[item.delivery_id] = lease_until
        return claimed

    def _holds(self, delivery: QueuedDelivery) -> bool:
        current: Final = self.queue.get(delivery.delivery_id)
        return current is not None and current.claim_id == delivery.claim_id

    async def prepare(self, delivery: QueuedDelivery, body: str, webhook_id: str) -> None:
        if self._holds(delivery):
            self.queue[delivery.delivery_id] = self.queue[delivery.delivery_id].model_copy(
                update={"body": body, "webhook_id": webhook_id, "prepared": True}
            )

    async def finish(self, delivery: QueuedDelivery, at: datetime) -> None:
        if not self._holds(delivery):
            return
        self.queue.pop(delivery.delivery_id, None)
        self.leases.pop(delivery.delivery_id, None)
        self.delivered[delivery.delivery_id] = at
        self.finished.append(delivery.delivery_id)

    async def purge_delivered(self, before: datetime) -> None:
        for key in [key for key, at in self.delivered.items() if at < before]:
            self.delivered.pop(key)

    async def reschedule(self, delivery: QueuedDelivery, attempt: int, next_attempt_at: datetime) -> None:
        if self._holds(delivery):
            self.queue[delivery.delivery_id] = self.queue[delivery.delivery_id].model_copy(
                update={"attempt": attempt, "next_attempt_at": next_attempt_at, "claim_id": None}
            )
            self.leases.pop(delivery.delivery_id, None)

    async def drop_deliveries(self, subscription_id: str) -> None:
        for key in [key for key, item in self.queue.items() if item.subscription_id == subscription_id]:
            self.queue.pop(key)
