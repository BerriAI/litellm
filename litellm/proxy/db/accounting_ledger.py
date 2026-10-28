from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, assert_never

from pydantic import AwareDatetime, ConfigDict, JsonValue, TypeAdapter

from litellm.accounting import CostEvent, DeliveryTarget, EventKind, InboundUsageEvent, OutboundUsageEvent
from litellm.proxy.db.db_span import db_span
from litellm.proxy.db.routing_prisma_wrapper import writer_wrapper

if TYPE_CHECKING:
    from prisma import Prisma

    from litellm.proxy.utils import PrismaClient

AccountingEvent: TypeAlias = InboundUsageEvent | OutboundUsageEvent | CostEvent
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_INSTANT: Final[TypeAdapter[datetime]] = TypeAdapter(AwareDatetime)
_STRINGS: Final = TypeAdapter(tuple[dict[str, str], ...])
_TIMEOUT: Final = timedelta(seconds=10)


class AccountingConflictError(ValueError):
    pass


_Transaction: TypeAlias = AbstractAsyncContextManager["Prisma"]


@dataclass(frozen=True, slots=True)
class _AcceptedEvent:
    event_id: str
    kind: EventKind
    source_event_id: str | None
    occurred_at: datetime
    payload: str
    payload_hash: str
    targets: tuple[str, ...]
    schema_version: int
    version: int


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _prepare(
    event: AccountingEvent,
    targets: tuple[DeliveryTarget, ...],
    envelope: str,
    schema_version: int,
    version: int,
) -> _AcceptedEvent:
    if type(schema_version) is not int or type(version) is not int or min(schema_version, version) < 1:
        raise ValueError("Accounting versions must be positive integers")
    validated: Final = type(event).model_validate_json(event.model_dump_json())
    payload: Final = _JSON.validate_json(envelope)
    if payload.get("event") != validated.model_dump(mode="json"):
        raise ValueError("Accounting envelope must contain the matching public event")
    destinations: Final = tuple(sorted(frozenset(DeliveryTarget(target).value for target in targets)))
    serialized: Final = _json(payload)
    identity: Final = _json((schema_version, version, destinations, payload))
    kind: Final = (
        EventKind.COST
        if isinstance(validated, CostEvent)
        else EventKind.INBOUND_USAGE
        if isinstance(validated, InboundUsageEvent)
        else EventKind.OUTBOUND_USAGE
    )
    return _AcceptedEvent(
        event_id=validated.event_id,
        kind=kind,
        source_event_id=validated.usage_event_id if isinstance(validated, CostEvent) else None,
        occurred_at=validated.occurred_at if isinstance(validated, CostEvent) else validated.ended_at,
        payload=serialized,
        payload_hash=hashlib.sha256(identity.encode()).hexdigest(),
        targets=destinations,
        schema_version=schema_version,
        version=version,
    )


_INSERT_EVENT: Final = """
INSERT INTO "LiteLLM_AccountingEvent"
    (event_id, kind, source_event_id, schema_version, payload_hash, occurred_at, payload)
VALUES ($1, $2::"AccountingEventKind", $3, $4::int, $5, $6::timestamp, $7::jsonb)
ON CONFLICT (event_id) DO NOTHING RETURNING event_id
"""
_INSERT_DELIVERIES: Final = """
INSERT INTO "LiteLLM_AccountingDelivery" (event_id, target, version)
SELECT $1, value::"AccountingDeliveryTarget", $3::int FROM jsonb_array_elements_text($2::jsonb)
"""
_LOCK_SOURCE: Final = """
SELECT event_id FROM "LiteLLM_AccountingEvent"
WHERE event_id = $1 AND kind = 'outbound_usage' FOR UPDATE
"""
_EXISTING_COST: Final = """
SELECT event_id FROM "LiteLLM_AccountingEvent" WHERE kind = 'cost' AND source_event_id = $1
"""
_LOCK_EVENT: Final = """
SELECT payload_hash FROM "LiteLLM_AccountingEvent" WHERE event_id = $1 FOR UPDATE
"""
_COMPLETED_FAMILY: Final = """
root.source_event_id IS NULL AND root.kind IN ('inbound_usage', 'outbound_usage')
AND root.recorded_at < $1::timestamp
AND (root.kind = 'inbound_usage' OR EXISTS (
    SELECT 1 FROM "LiteLLM_AccountingEvent" AS cost WHERE cost.source_event_id = root.event_id AND cost.kind = 'cost'
))
AND NOT EXISTS (
    SELECT 1 FROM "LiteLLM_AccountingEvent" AS member
    WHERE (member.event_id = root.event_id OR (member.kind = 'cost' AND member.source_event_id = root.event_id))
    AND (member.recorded_at >= $1::timestamp
        OR EXISTS (SELECT 1 FROM "LiteLLM_AccountingDelivery" AS delivery
            WHERE delivery.event_id = member.event_id AND delivery.state != 'applied'))
)
"""
_RETENTION_CANDIDATES: Final = (
    'SELECT root.event_id FROM "LiteLLM_AccountingEvent" AS root WHERE '
    + _COMPLETED_FAMILY
    + " ORDER BY root.recorded_at, root.event_id LIMIT $2::int FOR UPDATE OF root SKIP LOCKED"
)
_ELIGIBLE_GROUP: Final = (
    'SELECT root.event_id FROM "LiteLLM_AccountingEvent" AS root WHERE root.event_id = $2 AND ' + _COMPLETED_FAMILY
)
_DELETE_DELIVERIES: Final = """
DELETE FROM "LiteLLM_AccountingDelivery" WHERE event_id IN (
    SELECT event_id FROM "LiteLLM_AccountingEvent" WHERE event_id = $1 OR (kind = 'cost' AND source_event_id = $1)
)
"""
_DELETE_EVENTS: Final = """
DELETE FROM "LiteLLM_AccountingEvent" WHERE event_id = $1 OR (kind = 'cost' AND source_event_id = $1)
"""


_Conflict: TypeAlias = Literal["missing_source", "competing_cost", "identity_conflict"]


async def _accept(db: Prisma, record: _AcceptedEvent) -> Literal["inserted", "duplicate"] | _Conflict:
    if record.source_event_id is not None:
        source: Final = _STRINGS.validate_python(await db.query_raw(_LOCK_SOURCE, record.source_event_id))
        if not source:
            return "missing_source"
        costs: Final = _STRINGS.validate_python(tuple(await db.query_raw(_EXISTING_COST, record.source_event_id)))
        if any(cost["event_id"] != record.event_id for cost in costs):
            return "competing_cost"
    inserted: Final = _STRINGS.validate_python(
        await db.query_raw(
            _INSERT_EVENT,
            record.event_id,
            record.kind.value,
            record.source_event_id,
            record.schema_version,
            record.payload_hash,
            record.occurred_at.astimezone(timezone.utc).isoformat(),
            record.payload,
        )
    )
    if not inserted:
        existing: Final = _STRINGS.validate_python(tuple(await db.query_raw(_LOCK_EVENT, record.event_id)))
        if not existing or existing[0]["payload_hash"] != record.payload_hash:
            return "identity_conflict"
        return "duplicate"
    await db.execute_raw(_INSERT_DELIVERIES, record.event_id, _json(record.targets), record.version)
    return "inserted"


async def _delete_completed(db: Prisma, event_id: str, cutoff: str) -> int:
    if not _STRINGS.validate_python(await db.query_raw(_ELIGIBLE_GROUP, cutoff, event_id)):
        return 0
    await db.execute_raw(_DELETE_DELIVERIES, event_id)
    return await db.execute_raw(_DELETE_EVENTS, event_id)


class AccountingLedger:
    def __init__(self, transaction: Callable[[], _Transaction]) -> None:
        self._transaction: Final = transaction

    @classmethod
    def for_client(cls, client: PrismaClient) -> AccountingLedger:
        from prisma.client import TransactionManager

        managers: Final = TypeAdapter(TransactionManager, config=ConfigDict(arbitrary_types_allowed=True))

        @asynccontextmanager
        async def transaction() -> AsyncGenerator[Prisma]:
            primary: Final = writer_wrapper(client.db)
            manager: Final = managers.validate_python(primary.tx(timeout=_TIMEOUT))
            async with db_span("accounting_ledger", "LiteLLM_AccountingEvent"), manager as db:
                await db.execute_raw("SET LOCAL statement_timeout = 5000")
                await db.execute_raw("SET LOCAL lock_timeout = 1000")
                yield db

        return cls(transaction)

    async def insert_event(
        self,
        event: AccountingEvent,
        targets: tuple[DeliveryTarget, ...],
        *,
        envelope: str,
        schema_version: int = 1,
        version: int = 1,
    ) -> Literal["inserted", "duplicate"]:
        """Accept T5/T6's serialized envelope, containing its public event under `event`.

        The immutable acceptance hash includes the target set and delivery version.
        Changing either for an accepted event conflicts. T7 must provide a separate
        add-target or new-version workflow that preserves receipts and prevents
        duplicate contributions; redelivery cannot silently create or reset work.

        The Prisma adapter opens one interactive transaction per event, including
        two SET LOCAL statements and a source-row lock for costs. I/O is async,
        but awaiting this method inline adds commit latency per event, potentially
        several times per request. T5 owns bounded asynchronous capture and an
        explicit batching strategy; looping over this method does not batch commits.
        Success acknowledges commit, independently of HTTP response acknowledgement.
        """
        record: Final = _prepare(event, targets, envelope, schema_version, version)
        async with self._transaction() as db:
            result: Final = await _accept(db, record)
        match result:
            case "inserted" | "duplicate":
                return result
            case "missing_source":
                raise AccountingConflictError("Cost source must be a durably accepted outbound usage event")
            case "competing_cost":
                raise AccountingConflictError("Usage event already has an authoritative cost")
            case "identity_conflict":
                raise AccountingConflictError("Event identity conflicts with its accepted content or delivery set")
            case _:
                assert_never(result)

    async def purge_completed(
        self,
        *,
        redelivery_before: datetime | None = None,
        recovery_before: datetime | None = None,
        limit: int = 100,
    ) -> int:
        """Delete at most `limit` completed families once both retention owners permit it.

        T8 must supply a coherent recovery watermark including outstanding holds.
        Omitting either watermark disables deletion. Each usage row and its cost
        remain together so a retained charge never loses its source or vice versa.
        Applied receipts are deleted with the family, ending its retained dedup
        protection. The caller must guarantee events recorded before
        redelivery_before cannot be redelivered through supported paths.
        T8's recovery_before must also protect checkpoints,
        outstanding holds and unfinished settlements; age alone is insufficient.
        """
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Accounting retention batch size must be between 1 and 1000")
        if redelivery_before is None or recovery_before is None:
            return 0
        cutoff: Final = min(
            _INSTANT.validate_python(redelivery_before).astimezone(timezone.utc),
            _INSTANT.validate_python(recovery_before).astimezone(timezone.utc),
        ).isoformat()
        async with self._transaction() as db:
            roots: Final = _STRINGS.validate_python(tuple(await db.query_raw(_RETENTION_CANDIDATES, cutoff, limit)))
            deleted: Final = tuple([await _delete_completed(db, root["event_id"], cutoff) for root in roots])
        return sum(deleted)
