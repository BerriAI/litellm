from dataclasses import dataclass, field
from datetime import datetime

from litellm.proxy.mcp_server.events.webhooks.store import (
    LIVE_STATUSES,
    QueuedDelivery,
    SubscriptionQuery,
    SubscriptionRow,
)


@dataclass
class MemoryStore:
    rows: dict[str, SubscriptionRow] = field(default_factory=dict)  # mutable-ok: in-memory table
    queue: dict[str, QueuedDelivery] = field(default_factory=dict)  # mutable-ok: in-memory queue
    finished: list[str] = field(default_factory=list)  # mutable-ok: records completed deliveries
    fail_enqueue: bool = False

    async def get(self, subscription_id: str) -> SubscriptionRow | None:
        return self.rows.get(subscription_id)

    async def save(self, row: SubscriptionRow) -> None:
        self.rows[row.id] = row

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
        if delivery.delivery_id in self.queue:
            return False
        self.queue[delivery.delivery_id] = delivery
        return True

    async def claim(self, now: datetime, lease_until: datetime, limit: int) -> tuple[QueuedDelivery, ...]:
        return tuple(item for item in self.queue.values() if item.next_attempt_at <= now)[:limit]

    async def finish(self, delivery_id: str) -> None:
        self.queue.pop(delivery_id, None)
        self.finished.append(delivery_id)

    async def reschedule(self, delivery_id: str, attempt: int, next_attempt_at: datetime) -> None:
        self.queue[delivery_id] = self.queue[delivery_id].model_copy(
            update={"attempt": attempt, "next_attempt_at": next_attempt_at}
        )

    async def drop_deliveries(self, subscription_id: str) -> None:
        for key in [key for key, item in self.queue.items() if item.subscription_id == subscription_id]:
            self.queue.pop(key)
