"""Durable Lens ledger outbox: cost events and payload bodies, each claimed under its own lease.

Rows are inserted in the spend log delivery transaction and only read here. Claims commit before any
network call, so a crashed exporter's lease expires and another pod retries the same identity. A row
leaves the outbox only when Lens acknowledges it; transient failures reschedule it with capped backoff
and never park or delete it. Parked rows are malformed ones, kept until an operator requeues them.
"""

from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.repositories.accounting_delivery_repository import LedgerDatabase

_TIMEOUT: Final = timedelta(seconds=30)
_BACKOFF: Final = "make_interval(secs => LEAST(300, power(2, LEAST(attempts, 8)) * (0.5 + random())))"
_DUE: Final = (
    "parked_at IS NULL AND next_attempt_at <= CURRENT_TIMESTAMP "
    "AND (lease_until IS NULL OR lease_until < CURRENT_TIMESTAMP)"
)


class ClaimedEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    attempts: int
    event: str


class ClaimedPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    field: str
    team_id: str
    api_key: str
    event_time_ms: int
    content: str
    bytes: int
    sha256: str
    attempts: int


class OutboxBacklog(BaseModel):
    model_config = ConfigDict(frozen=True)

    pending_events: int
    parked_events: int
    pending_payloads: int
    parked_payloads: int


_EVENTS: Final = TypeAdapter(tuple[ClaimedEvent, ...])
_PAYLOADS: Final = TypeAdapter(tuple[ClaimedPayload, ...])

_CLAIM_EVENTS: Final = f"""UPDATE "LiteLLM_LensLedgerExport"
SET lease_owner = $1, lease_until = CURRENT_TIMESTAMP + make_interval(secs => $3::integer)
WHERE event_id IN (
    SELECT event_id FROM "LiteLLM_LensLedgerExport" WHERE {_DUE}
    ORDER BY next_attempt_at, event_id LIMIT $2::integer FOR UPDATE SKIP LOCKED)
RETURNING event_id, attempts, event::text AS event"""
_ACK_EVENTS: Final = 'DELETE FROM "LiteLLM_LensLedgerExport" WHERE lease_owner = $1 AND event_id = ANY($2::text[])'
_RETRY_EVENTS: Final = f"""UPDATE "LiteLLM_LensLedgerExport"
SET attempts = attempts + 1, last_error = $3, next_attempt_at = CURRENT_TIMESTAMP + {_BACKOFF},
    lease_owner = NULL, lease_until = NULL
WHERE lease_owner = $1 AND event_id = ANY($2::text[])"""
_PARK_EVENTS: Final = """UPDATE "LiteLLM_LensLedgerExport"
SET attempts = attempts + 1, last_error = $3, parked_at = CURRENT_TIMESTAMP, lease_owner = NULL, lease_until = NULL
WHERE lease_owner = $1 AND event_id = ANY($2::text[])"""

_CLAIM_PAYLOADS: Final = f"""UPDATE "LiteLLM_LensLedgerPayload" AS payload
SET lease_owner = $1, lease_until = CURRENT_TIMESTAMP + make_interval(secs => $3::integer)
FROM (
    SELECT event_id, field FROM (
        SELECT event_id, field,
               sum(bytes) OVER (ORDER BY next_attempt_at, event_id, field) AS running,
               row_number() OVER (ORDER BY next_attempt_at, event_id, field) AS position
        FROM (
            SELECT event_id, field, bytes, next_attempt_at FROM "LiteLLM_LensLedgerPayload" WHERE {_DUE}
            ORDER BY next_attempt_at, event_id, field LIMIT $2::integer FOR UPDATE SKIP LOCKED
        ) AS locked
    ) AS ranked
    WHERE running <= $4::bigint OR position = 1
) AS selected
WHERE payload.event_id = selected.event_id AND payload.field = selected.field
RETURNING payload.event_id, payload.field, payload.team_id, payload.api_key, payload.event_time_ms,
          payload.content, payload.bytes, payload.sha256, payload.attempts"""
_PAYLOAD_KEYS: Final = "(event_id, field) IN (SELECT * FROM unnest($2::text[], $3::text[]))"
_ACK_PAYLOADS: Final = f'DELETE FROM "LiteLLM_LensLedgerPayload" WHERE lease_owner = $1 AND {_PAYLOAD_KEYS}'
_RETRY_PAYLOADS: Final = f"""UPDATE "LiteLLM_LensLedgerPayload"
SET attempts = attempts + 1, last_error = $4, next_attempt_at = CURRENT_TIMESTAMP + {_BACKOFF},
    lease_owner = NULL, lease_until = NULL
WHERE lease_owner = $1 AND {_PAYLOAD_KEYS}"""
_PARK_PAYLOADS: Final = f"""UPDATE "LiteLLM_LensLedgerPayload"
SET attempts = attempts + 1, last_error = $4, parked_at = CURRENT_TIMESTAMP, lease_owner = NULL, lease_until = NULL
WHERE lease_owner = $1 AND {_PAYLOAD_KEYS}"""
_RELEASE: Final = (
    'UPDATE "LiteLLM_LensLedgerExport" SET lease_owner = NULL, lease_until = NULL WHERE lease_owner = $1',
    'UPDATE "LiteLLM_LensLedgerPayload" SET lease_owner = NULL, lease_until = NULL WHERE lease_owner = $1',
)
_REQUEUE: Final = {
    "events": """UPDATE "LiteLLM_LensLedgerExport"
SET parked_at = NULL, next_attempt_at = CURRENT_TIMESTAMP, last_error = NULL
WHERE parked_at IS NOT NULL AND ($1::text IS NULL OR last_error = $1)""",
    "payloads": """UPDATE "LiteLLM_LensLedgerPayload"
SET parked_at = NULL, next_attempt_at = CURRENT_TIMESTAMP, last_error = NULL
WHERE parked_at IS NOT NULL AND ($1::text IS NULL OR last_error = $1)""",
}
_BACKLOG: Final = """SELECT
    (SELECT COUNT(*) FROM "LiteLLM_LensLedgerExport" WHERE parked_at IS NULL) AS pending_events,
    (SELECT COUNT(*) FROM "LiteLLM_LensLedgerExport" WHERE parked_at IS NOT NULL) AS parked_events,
    (SELECT COUNT(*) FROM "LiteLLM_LensLedgerPayload" WHERE parked_at IS NULL) AS pending_payloads,
    (SELECT COUNT(*) FROM "LiteLLM_LensLedgerPayload" WHERE parked_at IS NOT NULL) AS parked_payloads"""


class LensExportRepository:
    def __init__(self, database: LedgerDatabase, owner: str) -> None:
        self._database: Final = database
        self._owner: Final = owner

    async def _rows(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        async with self._database.tx(timeout=_TIMEOUT) as transaction:
            return await transaction.query_raw(query, *args)

    async def _execute(self, query: str, *args: object) -> int:
        async with self._database.tx(timeout=_TIMEOUT) as transaction:
            return await transaction.execute_raw(query, *args)

    async def claim_events(self, limit: int, lease_seconds: int) -> tuple[ClaimedEvent, ...]:
        return _EVENTS.validate_python(await self._rows(_CLAIM_EVENTS, self._owner, limit, lease_seconds))

    async def acknowledge_events(self, event_ids: Sequence[str]) -> int:
        return await self._execute(_ACK_EVENTS, self._owner, list(event_ids)) if event_ids else 0

    async def retry_events(self, event_ids: Sequence[str], error: str) -> int:
        return await self._execute(_RETRY_EVENTS, self._owner, list(event_ids), error) if event_ids else 0

    async def park_events(self, event_ids: Sequence[str], error: str) -> int:
        return await self._execute(_PARK_EVENTS, self._owner, list(event_ids), error) if event_ids else 0

    async def claim_payloads(self, limit: int, lease_seconds: int, max_bytes: int) -> tuple[ClaimedPayload, ...]:
        return _PAYLOADS.validate_python(
            await self._rows(_CLAIM_PAYLOADS, self._owner, limit, lease_seconds, max_bytes)
        )

    async def acknowledge_payloads(self, payloads: Sequence[ClaimedPayload]) -> int:
        return await self._execute(_ACK_PAYLOADS, self._owner, *_keys(payloads)) if payloads else 0

    async def retry_payloads(self, payloads: Sequence[ClaimedPayload], error: str) -> int:
        return await self._execute(_RETRY_PAYLOADS, self._owner, *_keys(payloads), error) if payloads else 0

    async def park_payloads(self, payloads: Sequence[ClaimedPayload], error: str) -> int:
        return await self._execute(_PARK_PAYLOADS, self._owner, *_keys(payloads), error) if payloads else 0

    async def release(self) -> int:
        return sum([await self._execute(query, self._owner) for query in _RELEASE])

    async def requeue_parked(self, kind: Literal["events", "payloads"], error: str | None = None) -> int:
        return await self._execute(_REQUEUE[kind], error)

    async def backlog(self) -> OutboxBacklog:
        return OutboxBacklog.model_validate((await self._rows(_BACKLOG))[0])


def _keys(payloads: Sequence[ClaimedPayload]) -> tuple[list[str], list[str]]:
    return [payload.event_id for payload in payloads], [payload.field for payload in payloads]
