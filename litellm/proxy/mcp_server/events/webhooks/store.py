import json
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from litellm.proxy.mcp_server.events.webhooks.settlement import SettlementOutcome, settle_delivery
from litellm.types.mcp_server.events import CallbackFailure, GapEnvelope, JsonObject

if TYPE_CHECKING:
    from prisma.types import LiteLLM_MCPEventSubscriptionWhereInput

    from litellm.proxy.utils import PrismaClient

SubscriptionStatus: TypeAlias = Literal["pending", "active", "suspended", "terminated"]
LIVE_STATUSES: Final[tuple[SubscriptionStatus, ...]] = ("pending", "active", "suspended")
PENDING_TTL: Final = timedelta(minutes=10)


class SubscriptionRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    generation: str = Field(default_factory=lambda: uuid.uuid4().hex)
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
    prepared: bool = False
    claim_id: str | None = None


class ActivatedCursor(BaseModel):
    cursor: str | None


class Activation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reactivation_gap_id: str | None = None
    client_secret: str
    upstream_subscription_id: str | None
    upstream_refresh_before: datetime | None
    refresh_before: datetime | None
    cursor: str | None


class SubscriptionQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    principal: str | None = None
    key_hash: str | None = None
    server_id: str | None = None
    status: SubscriptionStatus | None = None
    live_only: bool = False
    limit: int = Field(default=100, ge=1, le=1001)
    offset: int = Field(default=0, ge=0)


class EventStore(Protocol):
    async def get(self, subscription_id: str) -> SubscriptionRow | None: ...

    async def reserve(self, row: SubscriptionRow) -> bool:
        """Insert ``row`` as pending unless a live row with its id exists; False when one does."""
        ...

    async def discard_pending(self, subscription_id: str) -> None: ...

    async def activate(
        self,
        subscription_id: str,
        activation: Activation,
        from_statuses: tuple[SubscriptionStatus, ...] = LIVE_STATUSES,
    ) -> ActivatedCursor | None:
        """Mark the row active with fresh secrets and deadlines while its status is still one of ``from_statuses``;
        None when it changed, so a renewal cannot revive a row an unsubscribe is holding."""
        ...

    async def mark_delivered(self, subscription_id: str, at: datetime) -> None: ...

    async def mark_failed(
        self, subscription_id: str, failure: CallbackFailure, at: datetime, suspend_after: int
    ) -> None: ...

    async def terminate(self, subscription_id: str) -> None: ...

    async def end_if_current(
        self, subscription_id: str, upstream_subscription_id: str | None, generation: str | None = None
    ) -> bool:
        """Terminate the row only while it is still this upstream subscription; None when it changed or ended."""
        ...

    async def hold_for_unsubscribe(self, subscription_id: str, upstream_subscription_id: str | None) -> bool:
        """Freeze a live row as pending while it is torn down, so delivery stops and a racing subscribe is told to
        retry instead of reclaiming it; the pending deadline frees the row if teardown never finishes."""
        ...

    async def delete_held(self, subscription_id: str, upstream_subscription_id: str | None) -> None:
        """Delete the row only while it is still held for this upstream subscription's unsubscribe."""
        ...

    async def delete(self, subscription_id: str) -> None: ...

    async def count_live(self, principal: str) -> int: ...

    async def count_status(self, status: SubscriptionStatus) -> int: ...

    async def verified(self, principal: str, client_url: str) -> bool: ...

    async def find(self, query: SubscriptionQuery) -> tuple[SubscriptionRow, ...]: ...

    async def due_for_upstream_refresh(self, before: datetime) -> tuple[SubscriptionRow, ...]: ...

    async def expired(self, now: datetime) -> tuple[SubscriptionRow, ...]: ...

    async def end_if_expired(self, subscription_id: str, now: datetime) -> bool:
        """Terminate the row only while it is still past its deadline, so a renewal that just landed survives."""
        ...

    async def enqueue_current(self, row: SubscriptionRow, delivery: QueuedDelivery) -> bool: ...

    async def enqueue(self, delivery: QueuedDelivery) -> bool: ...

    async def claim(self, now: datetime, lease_until: datetime, limit: int) -> tuple[QueuedDelivery, ...]: ...

    async def prepare(self, delivery: QueuedDelivery, body: str, webhook_id: str) -> None: ...

    async def finish(self, delivery: QueuedDelivery, at: datetime) -> None:
        """Keep the row as a tombstone so a replay of the same upstream message is still a duplicate."""
        ...

    async def reschedule(self, delivery: QueuedDelivery, attempt: int, next_attempt_at: datetime) -> None: ...

    async def drop_deliveries(self, subscription_id: str) -> None: ...

    async def settle(
        self, delivery: QueuedDelivery, generation: str, at: datetime, outcome: SettlementOutcome, *, gap: bool = False
    ) -> bool: ...

    async def purge_delivered(self, before: datetime) -> None: ...


_SECRET_FIELDS: Final = ("client_secret", "upstream_secret")
_RECORD: Final = TypeAdapter[dict[str, object]](dict[str, object])
_NOT_TERMINATED: Final = {"not": "terminated"}


def _encrypt(value: str) -> str:
    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper

    return encrypt_value_helper(value)


def _decrypt(value: str, field: str) -> str:
    from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper

    decrypted: Final = decrypt_value_helper(value, key=field, exception_type="debug")
    return decrypted if decrypted is not None else ""


def _row_from_record(record: BaseModel) -> SubscriptionRow:
    raw: Final = _RECORD.validate_python(record.model_dump())
    arguments: Final = raw.get("arguments")
    return SubscriptionRow.model_validate(
        {
            **raw,
            "arguments": json.loads(arguments) if isinstance(arguments, str) else arguments or {},
            **{field: _decrypt(str(raw[field]), field) for field in _SECRET_FIELDS},
        }
    )


def _record_from_row(row: SubscriptionRow) -> Mapping[str, object]:
    data: Final = _RECORD.validate_python(row.model_dump(exclude={"created_at", "updated_at"}))
    return {
        **data,
        "arguments": json.dumps(dict(row.arguments), sort_keys=True),
        "client_secret": _encrypt(row.client_secret),
        "upstream_secret": _encrypt(row.upstream_secret),
    }


def pending_row(row: SubscriptionRow) -> SubscriptionRow:
    return row.model_copy(update={"status": "pending", "refresh_before": datetime.now(timezone.utc) + PENDING_TTL})


def _claimed(delivery: QueuedDelivery) -> Mapping[str, object]:
    """Fence every write to the worker that holds the claim, so one whose lease expired changes nothing."""
    return {"delivery_id": delivery.delivery_id, "claim_id": delivery.claim_id}


class _RawQueries(Protocol):
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...


class PrismaEventStore:
    def __init__(self, prisma_client: "PrismaClient") -> None:
        from litellm.repositories.table_repositories import MCPEventDeliveryRepository, MCPEventSubscriptionRepository

        # Every read follows a write it must see and the raw statements are UPDATEs, so a read replica is never used.
        self._client: Final = prisma_client
        self._subscriptions: Final = MCPEventSubscriptionRepository(prisma_client, use_writer=True).table
        self._deliveries: Final = MCPEventDeliveryRepository(prisma_client, use_writer=True).table
        self._raw: Final[_RawQueries] = prisma_client.writer_db  # pyright: ignore[reportAttributeAccessIssue]  # PrismaWrapper forwards raw queries via __getattr__

    async def get(self, subscription_id: str) -> SubscriptionRow | None:
        record: Final = await self._subscriptions.find_unique(where={"id": subscription_id})
        return _row_from_record(record) if record is not None else None

    async def reserve(self, row: SubscriptionRow) -> bool:
        from prisma.errors import UniqueViolationError

        data: Final = dict(_record_from_row(pending_row(row)))
        replaced: Final = await self._subscriptions.update_many(
            where={"id": row.id, "status": "terminated"},
            data={key: value for key, value in data.items() if key != "id"},
        )
        if replaced:
            return True
        try:
            await self._subscriptions.create(data=data)
        except UniqueViolationError:
            return False
        return True

    async def discard_pending(self, subscription_id: str) -> None:
        await self._subscriptions.delete_many(where={"id": subscription_id, "status": "pending"})

    async def activate(
        self,
        subscription_id: str,
        activation: Activation,
        from_statuses: tuple[SubscriptionStatus, ...] = LIVE_STATUSES,
    ) -> ActivatedCursor | None:
        eligible: Final = [state for state in LIVE_STATUSES if state in from_statuses]
        async with self._client.tx() as tx:
            # The receiver takes FOR SHARE on this row before it inserts a delivery.
            previous: Final = await tx.query_first(
                'SELECT cursor FROM "LiteLLM_MCPEventSubscription" WHERE id = $1 FOR UPDATE',
                subscription_id,
                model=ActivatedCursor,
            )
            if previous is None:
                return None
            pending: Final = await tx.query_first(
                'SELECT $2::text AS cursor FROM "LiteLLM_MCPEventDelivery" '
                "WHERE subscription_id = $1 AND delivered_at IS NULL LIMIT 1",
                subscription_id,
                previous.cursor,
                model=ActivatedCursor,
            )
            watermark: Final = pending or ActivatedCursor(cursor=activation.cursor)
            updated: Final = await tx.litellm_mcpeventsubscription.update_many(
                where={"id": subscription_id, "status": {"in": eligible}},  # bounded-ok: at most the three live states
                data={
                    "status": "active",
                    "generation": uuid.uuid4().hex,
                    "client_secret": _encrypt(activation.client_secret),
                    "upstream_subscription_id": activation.upstream_subscription_id,
                    "upstream_refresh_before": activation.upstream_refresh_before,
                    "refresh_before": activation.refresh_before,
                    "cursor": watermark.cursor,
                    "consecutive_failures": 0,
                },
            )
            gap: Final = activation.reactivation_gap_id
            if updated and gap is not None:
                await tx.execute_raw(
                    """INSERT INTO "LiteLLM_MCPEventDelivery"
                    (delivery_id, subscription_id, webhook_id, body, prepared)
                    VALUES ($1, $2, $1, $3, true)""",
                    gap,
                    subscription_id,
                    GapEnvelope(cursor=watermark.cursor).model_dump_json(),
                )
        return watermark if updated else None

    async def mark_delivered(self, subscription_id: str, at: datetime) -> None:
        await self._subscriptions.update_many(
            where={"id": subscription_id, "status": _NOT_TERMINATED},
            data={"last_delivery_at": at, "consecutive_failures": 0, "failed_since": None, "last_error": None},
        )

    async def mark_failed(
        self, subscription_id: str, failure: CallbackFailure, at: datetime, suspend_after: int
    ) -> None:
        await self._raw.query_raw(
            'UPDATE "LiteLLM_MCPEventSubscription" SET '
            '"consecutive_failures" = "consecutive_failures" + 1, "last_error" = $2, '
            '"failed_since" = COALESCE("failed_since", $3::timestamp), "updated_at" = now(), '
            '"status" = CASE WHEN "status" = \'active\' AND "consecutive_failures" + 1 >= $4 '
            "THEN 'suspended' ELSE \"status\" END "
            'WHERE "id" = $1 AND "status" <> \'terminated\' RETURNING "id"',
            subscription_id,
            failure,
            at.isoformat(),
            suspend_after,
        )

    async def terminate(self, subscription_id: str) -> None:
        await self._subscriptions.update_many(where={"id": subscription_id}, data={"status": "terminated"})

    async def end_if_current(
        self, subscription_id: str, upstream_subscription_id: str | None, generation: str | None = None
    ) -> bool:
        where: Final[LiteLLM_MCPEventSubscriptionWhereInput] = {
            "id": subscription_id,
            "upstream_subscription_id": upstream_subscription_id,
            "status": {"not": "terminated"},
        }
        if generation is not None:
            where["generation"] = generation
        async with self._client.tx() as tx:
            ended: Final = await tx.litellm_mcpeventsubscription.update_many(
                where=where,
                data={"status": "terminated"},
            )
            if ended:
                await tx.litellm_mcpeventdelivery.delete_many(where={"subscription_id": subscription_id})
        return ended > 0

    async def hold_for_unsubscribe(self, subscription_id: str, upstream_subscription_id: str | None) -> bool:
        held: Final = await self._subscriptions.update_many(
            where={
                "id": subscription_id,
                "upstream_subscription_id": upstream_subscription_id,
                "status": {"in": ["active", "suspended"]},
            },
            data={"status": "pending", "refresh_before": datetime.now(timezone.utc) + PENDING_TTL},
        )
        return held > 0

    async def delete_held(self, subscription_id: str, upstream_subscription_id: str | None) -> None:
        await self._subscriptions.delete_many(
            where={"id": subscription_id, "upstream_subscription_id": upstream_subscription_id, "status": "pending"}
        )

    async def delete(self, subscription_id: str) -> None:
        await self._subscriptions.delete_many(where={"id": subscription_id})
        await self.drop_deliveries(subscription_id)

    async def count_live(self, principal: str) -> int:
        return await self._subscriptions.count(where={"principal": principal, "status": {"in": list(LIVE_STATUSES)}})

    async def count_status(self, status: SubscriptionStatus) -> int:
        return await self._subscriptions.count(where={"status": status})

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
        live: Final = {"status": {"in": list(LIVE_STATUSES)}} if query.live_only and query.status is None else {}
        records: Final = await self._subscriptions.find_many(
            where={**where, **live}, take=query.limit, skip=query.offset, order={"created_at": "desc"}
        )
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

    async def end_if_expired(self, subscription_id: str, now: datetime) -> bool:
        ended: Final = await self._subscriptions.update_many(
            where={"id": subscription_id, "status": {"in": list(LIVE_STATUSES)}, "refresh_before": {"lte": now}},
            data={"status": "terminated"},
        )
        return ended > 0

    async def enqueue_current(self, row: SubscriptionRow, delivery: QueuedDelivery) -> bool:
        """Acknowledge duplicates, but retry if the subscription changed after signature verification."""
        checked: Final = await self._raw.query_raw(
            """WITH current AS (
                SELECT id FROM "LiteLLM_MCPEventSubscription"
                WHERE id = $1 AND generation = $2 AND status = $3 FOR SHARE
            ), inserted AS (
                INSERT INTO "LiteLLM_MCPEventDelivery" (subscription_id, delivery_id, body, webhook_id, next_attempt_at)
                SELECT id, $4, $5, $6, $7::timestamp FROM current ON CONFLICT (delivery_id) DO NOTHING
            ) SELECT id FROM current""",
            delivery.subscription_id,
            row.generation,
            row.status,
            delivery.delivery_id,
            delivery.body,
            delivery.webhook_id,
            delivery.next_attempt_at.isoformat(),
        )
        return bool(checked)

    async def enqueue(self, delivery: QueuedDelivery) -> bool:
        from prisma.errors import UniqueViolationError

        try:
            await self._deliveries.create(data=delivery.model_dump())
        except UniqueViolationError:
            return False
        return True

    async def claim(self, now: datetime, lease_until: datetime, limit: int) -> tuple[QueuedDelivery, ...]:
        rows: Final = await self._raw.query_raw(
            'UPDATE "LiteLLM_MCPEventDelivery" SET "claimed_until" = $1::timestamp, "claim_id" = $4 '
            'WHERE "delivery_id" IN ('  # bounded-ok: the subquery LIMITs the claimed batch
            'SELECT "delivery_id" FROM "LiteLLM_MCPEventDelivery" '
            'WHERE "delivered_at" IS NULL AND "next_attempt_at" <= $2::timestamp '
            'AND ("claimed_until" IS NULL OR "claimed_until" < $2::timestamp) '
            'ORDER BY "next_attempt_at" LIMIT $3 FOR UPDATE SKIP LOCKED) '
            'RETURNING "delivery_id", "subscription_id", "body", "webhook_id", "attempt", "next_attempt_at", "prepared", '
            '"claim_id"',
            lease_until.isoformat(),
            now.isoformat(),
            limit,
            uuid.uuid4().hex,
        )
        return tuple(QueuedDelivery.model_validate(row) for row in rows)

    async def prepare(self, delivery: QueuedDelivery, body: str, webhook_id: str) -> None:
        await self._deliveries.update_many(
            where=_claimed(delivery), data={"body": body, "webhook_id": webhook_id, "prepared": True}
        )

    async def finish(self, delivery: QueuedDelivery, at: datetime) -> None:
        await self._deliveries.update_many(
            where=_claimed(delivery), data={"delivered_at": at, "body": "", "claimed_until": None, "claim_id": None}
        )

    async def reschedule(self, delivery: QueuedDelivery, attempt: int, next_attempt_at: datetime) -> None:
        await self._deliveries.update_many(
            where=_claimed(delivery),
            data={"attempt": attempt, "next_attempt_at": next_attempt_at, "claimed_until": None, "claim_id": None},
        )

    async def drop_deliveries(self, subscription_id: str) -> None:
        await self._deliveries.delete_many(where={"subscription_id": subscription_id})

    async def settle(
        self, delivery: QueuedDelivery, generation: str, at: datetime, outcome: SettlementOutcome, *, gap: bool = False
    ) -> bool:
        return await settle_delivery(self._client, delivery, generation, at, outcome, gap=gap)

    async def purge_delivered(self, before: datetime) -> None:
        await self._deliveries.delete_many(where={"delivered_at": {"lt": before}})


def default_store() -> EventStore | None:
    from litellm.proxy.proxy_server import prisma_client

    return PrismaEventStore(prisma_client) if prisma_client is not None else None
