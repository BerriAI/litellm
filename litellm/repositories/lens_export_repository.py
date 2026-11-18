"""Durable Lens ledger outbox: cost events and payload bodies, each claimed under its own lease.

Rows are inserted in the spend log delivery transaction and only read here. Claims commit before any
network call, so a crashed exporter's lease simply expires and another pod retries the same identity.
"""

from collections.abc import Sequence
from datetime import timedelta
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.repositories.accounting_delivery_repository import LedgerDatabase

_TIMEOUT: Final = timedelta(seconds=30)
_BACKOFF: Final = "make_interval(secs => LEAST(900, 5 + power(2, LEAST(attempts, 10)) * (0.5 + random())))"


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


_EVENTS: Final = TypeAdapter(tuple[ClaimedEvent, ...])
_PAYLOADS: Final = TypeAdapter(tuple[ClaimedPayload, ...])

_CLAIM_EVENTS: Final = """UPDATE "LiteLLM_LensLedgerExport"
SET lease_owner = $1, lease_until = CURRENT_TIMESTAMP + make_interval(secs => $3::integer)
WHERE event_id IN (
    SELECT event_id FROM "LiteLLM_LensLedgerExport"
    WHERE parked_at IS NULL AND next_attempt_at <= CURRENT_TIMESTAMP
      AND (lease_until IS NULL OR lease_until < CURRENT_TIMESTAMP)
    ORDER BY next_attempt_at, event_id LIMIT $2::integer FOR UPDATE SKIP LOCKED)
RETURNING event_id, attempts, event::text AS event"""
_ACK_EVENTS: Final = """DELETE FROM "LiteLLM_LensLedgerExport" WHERE lease_owner = $1 AND event_id = ANY($2::text[])"""
_RETRY_EVENTS: Final = f"""UPDATE "LiteLLM_LensLedgerExport"
SET attempts = attempts + 1, last_error = $3, next_attempt_at = CURRENT_TIMESTAMP + {_BACKOFF},
    parked_at = CASE WHEN attempts + 1 >= $4::integer THEN CURRENT_TIMESTAMP END,
    lease_owner = NULL, lease_until = NULL
WHERE lease_owner = $1 AND event_id = ANY($2::text[])"""
_PARK_EVENTS: Final = """UPDATE "LiteLLM_LensLedgerExport"
SET attempts = attempts + 1, last_error = $3, parked_at = CURRENT_TIMESTAMP, lease_owner = NULL, lease_until = NULL
WHERE lease_owner = $1 AND event_id = ANY($2::text[])"""

_CLAIM_PAYLOADS: Final = """UPDATE "LiteLLM_LensLedgerPayload"
SET lease_owner = $1, lease_until = CURRENT_TIMESTAMP + make_interval(secs => $3::integer)
WHERE (event_id, field) IN (
    SELECT event_id, field FROM "LiteLLM_LensLedgerPayload"
    WHERE parked_at IS NULL AND next_attempt_at <= CURRENT_TIMESTAMP
      AND (lease_until IS NULL OR lease_until < CURRENT_TIMESTAMP)
    ORDER BY next_attempt_at, event_id, field LIMIT $2::integer FOR UPDATE SKIP LOCKED)
RETURNING event_id, field, team_id, api_key, event_time_ms, content, bytes, sha256, attempts"""
_PAYLOAD_KEYS: Final = "(event_id, field) IN (SELECT * FROM unnest($2::text[], $3::text[]))"
_ACK_PAYLOADS: Final = f"""DELETE FROM "LiteLLM_LensLedgerPayload" WHERE lease_owner = $1 AND {_PAYLOAD_KEYS}"""
_RETRY_PAYLOADS: Final = f"""UPDATE "LiteLLM_LensLedgerPayload"
SET attempts = attempts + 1, last_error = $4, next_attempt_at = CURRENT_TIMESTAMP + {_BACKOFF},
    parked_at = CASE WHEN attempts + 1 >= $5::integer THEN CURRENT_TIMESTAMP END,
    lease_owner = NULL, lease_until = NULL
WHERE lease_owner = $1 AND {_PAYLOAD_KEYS}"""
_PARK_PAYLOADS: Final = f"""UPDATE "LiteLLM_LensLedgerPayload"
SET attempts = attempts + 1, last_error = $4, parked_at = CURRENT_TIMESTAMP, lease_owner = NULL, lease_until = NULL
WHERE lease_owner = $1 AND {_PAYLOAD_KEYS}"""
_CLEANUP: Final = (
    """DELETE FROM "LiteLLM_LensLedgerExport" WHERE parked_at < CURRENT_TIMESTAMP - make_interval(days => $1::integer)""",
    """DELETE FROM "LiteLLM_LensLedgerPayload" WHERE parked_at < CURRENT_TIMESTAMP - make_interval(days => $1::integer)""",
)


class LensExportRepository:
    def __init__(self, database: LedgerDatabase, owner: str) -> None:
        self._database: Final = database
        self._owner: Final = owner

    async def _rows(self, query: str, *args: object) -> Sequence[object]:
        async with self._database.tx(timeout=_TIMEOUT) as transaction:
            return await transaction.query_raw(query, *args)

    async def _execute(self, query: str, *args: object) -> int:
        async with self._database.tx(timeout=_TIMEOUT) as transaction:
            return await transaction.execute_raw(query, *args)

    async def claim_events(self, limit: int, lease_seconds: int) -> tuple[ClaimedEvent, ...]:
        return _EVENTS.validate_python(await self._rows(_CLAIM_EVENTS, self._owner, limit, lease_seconds))

    async def acknowledge_events(self, event_ids: Sequence[str]) -> int:
        return await self._execute(_ACK_EVENTS, self._owner, list(event_ids)) if event_ids else 0

    async def retry_events(self, event_ids: Sequence[str], error: str, max_attempts: int) -> int:
        if not event_ids:
            return 0
        return await self._execute(_RETRY_EVENTS, self._owner, list(event_ids), error, max_attempts)

    async def park_events(self, event_ids: Sequence[str], error: str) -> int:
        return await self._execute(_PARK_EVENTS, self._owner, list(event_ids), error) if event_ids else 0

    async def claim_payloads(self, limit: int, lease_seconds: int) -> tuple[ClaimedPayload, ...]:
        return _PAYLOADS.validate_python(await self._rows(_CLAIM_PAYLOADS, self._owner, limit, lease_seconds))

    async def acknowledge_payloads(self, payloads: Sequence[ClaimedPayload]) -> int:
        if not payloads:
            return 0
        return await self._execute(_ACK_PAYLOADS, self._owner, *_keys(payloads))

    async def retry_payloads(self, payloads: Sequence[ClaimedPayload], error: str, max_attempts: int) -> int:
        if not payloads:
            return 0
        return await self._execute(_RETRY_PAYLOADS, self._owner, *_keys(payloads), error, max_attempts)

    async def park_payloads(self, payloads: Sequence[ClaimedPayload], error: str) -> int:
        if not payloads:
            return 0
        return await self._execute(_PARK_PAYLOADS, self._owner, *_keys(payloads), error)

    async def cleanup(self, retention_days: int) -> int:
        if retention_days < 1:
            raise ValueError("Lens export retention must be at least one day")
        return sum([await self._execute(query, retention_days) for query in _CLEANUP])


def _keys(payloads: Sequence[ClaimedPayload]) -> tuple[list[str], list[str]]:
    return [payload.event_id for payload in payloads], [payload.field for payload in payloads]
