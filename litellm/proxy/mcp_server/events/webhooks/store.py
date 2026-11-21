import json
from datetime import datetime
from typing import TYPE_CHECKING, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from litellm.types.mcp_server.events import CallbackFailure, JsonObject

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

SubscriptionStatus = Literal["pending", "active", "suspended", "terminated"]
LIVE_STATUSES: Final[tuple[SubscriptionStatus, ...]] = ("pending", "active", "suspended")


class SubscriptionRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    principal: str
    key_hash: str | None = None
    user_id: str | None = None
    team_id: str | None = None
    server_id: str
    event_name: str
    arguments: JsonObject = Field(default_factory=dict)
    client_url: str
    client_secret: str
    upstream_subscription_id: str | None = None
    upstream_secret: str
    upstream_refresh_before: datetime | None = None
    refresh_before: datetime | None = None
    cursor: str | None = None
    status: SubscriptionStatus = "pending"
    verified_at: datetime | None = None
    last_delivery_at: datetime | None = None
    last_error: CallbackFailure | None = None
    failed_since: datetime | None = None
    consecutive_failures: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None


class QueuedDelivery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    delivery_id: str
    subscription_id: str
    body: str
    webhook_id: str
    attempt: int = 0
    next_attempt_at: datetime


class SubscriptionQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    principal: str | None = None
    key_hash: str | None = None
    server_id: str | None = None
    status: SubscriptionStatus | None = None
    limit: int = Field(default=100, ge=1, le=1000)


class EventStore(Protocol):
    async def get(self, subscription_id: str) -> SubscriptionRow | None: ...

    async def save(self, row: SubscriptionRow) -> None: ...

    async def delete(self, subscription_id: str) -> None: ...

    async def count_live(self, principal: str) -> int: ...

    async def verified(self, principal: str, client_url: str) -> bool: ...

    async def find(self, query: SubscriptionQuery) -> tuple[SubscriptionRow, ...]: ...

    async def due_for_upstream_refresh(self, before: datetime) -> tuple[SubscriptionRow, ...]: ...

    async def expired(self, now: datetime) -> tuple[SubscriptionRow, ...]: ...

    async def enqueue(self, delivery: QueuedDelivery) -> None: ...

    async def claim(self, now: datetime, lease_until: datetime, limit: int) -> tuple[QueuedDelivery, ...]: ...

    async def finish(self, delivery_id: str) -> None: ...

    async def reschedule(self, delivery_id: str, attempt: int, next_attempt_at: datetime) -> None: ...

    async def drop_deliveries(self, subscription_id: str) -> None: ...


_SECRET_FIELDS: Final = ("client_secret", "upstream_secret")


def _encrypt(value: str) -> str:
    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper

    return encrypt_value_helper(value)


def _decrypt(value: str, field: str) -> str:
    from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper

    decrypted: Final = decrypt_value_helper(value, key=field, exception_type="debug")
    return decrypted if decrypted is not None else ""


def _row_from_record(record: BaseModel) -> SubscriptionRow:
    raw: Final = record.model_dump()
    arguments: Final = raw.get("arguments")
    return SubscriptionRow.model_validate(
        {
            **raw,
            "arguments": json.loads(arguments) if isinstance(arguments, str) else arguments or {},
            **{field: _decrypt(str(raw[field]), field) for field in _SECRET_FIELDS},
        }
    )


def _record_from_row(row: SubscriptionRow) -> dict[str, JsonValue | datetime]:
    data: Final = row.model_dump(exclude={"created_at", "updated_at"})
    return {
        **data,
        "arguments": json.dumps(dict(row.arguments), sort_keys=True),
        **{field: _encrypt(getattr(row, field)) for field in _SECRET_FIELDS},
    }


class PrismaEventStore:
    def __init__(self, prisma_client: "PrismaClient") -> None:
        self._db = prisma_client.db

    @property
    def _subscriptions(self):  # noqa: ANN202  # Prisma generates the delegate type at runtime
        return self._db.litellm_mcpeventsubscription

    @property
    def _deliveries(self):  # noqa: ANN202  # Prisma generates the delegate type at runtime
        return self._db.litellm_mcpeventdelivery

    async def get(self, subscription_id: str) -> SubscriptionRow | None:
        record: Final = await self._subscriptions.find_unique(where={"id": subscription_id})
        return _row_from_record(record) if record is not None else None

    async def save(self, row: SubscriptionRow) -> None:
        data: Final = _record_from_row(row)
        update: Final = {key: value for key, value in data.items() if key != "id"}
        await self._subscriptions.upsert(where={"id": row.id}, data={"create": data, "update": update})

    async def delete(self, subscription_id: str) -> None:
        await self._subscriptions.delete_many(where={"id": subscription_id})
        await self.drop_deliveries(subscription_id)

    async def count_live(self, principal: str) -> int:
        return await self._subscriptions.count(where={"principal": principal, "status": {"in": list(LIVE_STATUSES)}})

    async def verified(self, principal: str, client_url: str) -> bool:
        record: Final = await self._subscriptions.find_first(
            where={"principal": principal, "client_url": client_url, "verified_at": {"not": None}}
        )
        return record is not None

    async def find(self, query: SubscriptionQuery) -> tuple[SubscriptionRow, ...]:
        where: Final = {
            key: value
            for key, value in (
                ("principal", query.principal),
                ("key_hash", query.key_hash),
                ("server_id", query.server_id),
                ("status", query.status),
            )
            if value is not None
        }
        records: Final = await self._subscriptions.find_many(where=where, take=query.limit, order={"created_at": "desc"})
        return tuple(_row_from_record(record) for record in records)

    async def due_for_upstream_refresh(self, before: datetime) -> tuple[SubscriptionRow, ...]:
        records: Final = await self._subscriptions.find_many(
            where={"status": {"in": ["active", "suspended"]}, "upstream_refresh_before": {"lte": before}}, take=100
        )
        return tuple(_row_from_record(record) for record in records)

    async def expired(self, now: datetime) -> tuple[SubscriptionRow, ...]:
        records: Final = await self._subscriptions.find_many(
            where={"status": {"in": list(LIVE_STATUSES)}, "refresh_before": {"lte": now}}, take=100
        )
        return tuple(_row_from_record(record) for record in records)

    async def enqueue(self, delivery: QueuedDelivery) -> None:
        await self._deliveries.create(data=delivery.model_dump())

    async def claim(self, now: datetime, lease_until: datetime, limit: int) -> tuple[QueuedDelivery, ...]:
        rows: Final = await self._db.query_raw(
            'UPDATE "LiteLLM_MCPEventDelivery" SET "claimed_until" = $1::timestamp '
            'WHERE "delivery_id" IN ('  # bounded-ok: the subquery LIMITs the claimed batch
            'SELECT "delivery_id" FROM "LiteLLM_MCPEventDelivery" '
            'WHERE "next_attempt_at" <= $2::timestamp AND ("claimed_until" IS NULL OR "claimed_until" < $2::timestamp) '
            'ORDER BY "next_attempt_at" LIMIT $3 FOR UPDATE SKIP LOCKED) '
            'RETURNING "delivery_id", "subscription_id", "body", "webhook_id", "attempt", "next_attempt_at"',
            lease_until.isoformat(),
            now.isoformat(),
            limit,
        )
        return tuple(QueuedDelivery.model_validate(row) for row in rows)

    async def finish(self, delivery_id: str) -> None:
        await self._deliveries.delete_many(where={"delivery_id": delivery_id})

    async def reschedule(self, delivery_id: str, attempt: int, next_attempt_at: datetime) -> None:
        await self._deliveries.update_many(
            where={"delivery_id": delivery_id},
            data={"attempt": attempt, "next_attempt_at": next_attempt_at, "claimed_until": None},
        )

    async def drop_deliveries(self, subscription_id: str) -> None:
        await self._deliveries.delete_many(where={"subscription_id": subscription_id})


def default_store() -> EventStore | None:
    from litellm.proxy.proxy_server import prisma_client

    return PrismaEventStore(prisma_client) if prisma_client is not None else None
