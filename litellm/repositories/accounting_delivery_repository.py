from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from itertools import chain
from types import TracebackType
from typing import TYPE_CHECKING, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.repositories.prisma_protocols import TableActions

if TYPE_CHECKING:
    from litellm.proxy.db.db_spend_update_writer import SpendBatchManager


class LedgerTransaction(Protocol):
    @property
    def litellm_dailyguardrailusageunits(self) -> TableActions[object]: ...

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...

    async def execute_raw(self, query: str, *args: object) -> int: ...

    def batch_(self) -> "SpendBatchManager": ...


class _TransactionManager(Protocol):
    async def __aenter__(self) -> LedgerTransaction: ...

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, exc_tb: TracebackType | None
    ) -> None: ...


class LedgerDatabase(Protocol):
    def tx(self, *, timeout: timedelta) -> _TransactionManager: ...


class _Delivery(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str
    attempts: int
    previous_row: Mapping[str, object] | None = None


class _StoredRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    row: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class DeliveryProjection:
    row: Mapping[str, object]
    previous_row: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class DeliveryBatchResult:
    applied: int = 0
    failed: int = 0
    parked: tuple[str, ...] = ()
    error: str | None = None


class DeliveryMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    pending_count: int
    parked_count: int
    oldest_pending_seconds: float


_DELIVERIES: Final = TypeAdapter(tuple[_Delivery, ...])
_ROWS: Final = TypeAdapter(tuple[_StoredRow, ...])
_TIMEOUT: Final = timedelta(seconds=60)
_CLAIM: Final = """SELECT request_id, attempts, previous_row FROM "LiteLLM_AccountingDelivery"
WHERE state = 'pending' AND next_attempt_at <= CURRENT_TIMESTAMP
  AND (attempts > 0) = $2::boolean
ORDER BY next_attempt_at, request_id LIMIT $1::integer FOR UPDATE SKIP LOCKED"""
_LOAD: Final = """SELECT to_jsonb(spend_row) AS row FROM "LiteLLM_SpendLogs" spend_row
WHERE request_id = ANY($1::text[]) ORDER BY request_id"""
_APPLIED: Final = """UPDATE "LiteLLM_AccountingDelivery"
SET state = 'applied', applied_at = CURRENT_TIMESTAMP, previous_row = NULL, last_error = NULL
WHERE request_id = ANY($1::text[]) AND state = 'pending' """
_FAILED: Final = """UPDATE "LiteLLM_AccountingDelivery"
SET attempts = attempts + 1, last_error = $3,
    next_attempt_at = CURRENT_TIMESTAMP + make_interval(secs => LEAST(300, power(2, LEAST(attempts, 8)))::integer),
    state = $4::"AccountingDeliveryState"
WHERE request_id = $1 AND state = 'pending' AND attempts = $2::integer
RETURNING request_id, attempts, previous_row"""


class AccountingDeliveryRepository:
    def __init__(self, database: LedgerDatabase) -> None:
        self._database = database

    async def metrics(self) -> DeliveryMetrics:
        async with self._database.tx(timeout=_TIMEOUT) as transaction:
            rows: Final = await transaction.query_raw(
                "SELECT COUNT(*) FILTER (WHERE state = 'pending') AS pending_count, "
                "COUNT(*) FILTER (WHERE state = 'parked') AS parked_count, "
                "COALESCE(EXTRACT(EPOCH FROM CURRENT_TIMESTAMP - MIN(created_at) FILTER (WHERE state = 'pending')), 0) "
                "AS oldest_pending_seconds FROM \"LiteLLM_AccountingDelivery\" WHERE state IN ('pending', 'parked')"
            )
            return DeliveryMetrics.model_validate(rows[0])

    async def cleanup_applied(self, *, retention_days: int = 30, batch_size: int = 1000) -> int:
        if retention_days < 1 or batch_size < 1:
            raise ValueError("Delivery retention and cleanup batch size must be positive")
        async with self._database.tx(timeout=_TIMEOUT) as transaction:
            return await transaction.execute_raw(
                'DELETE FROM "LiteLLM_AccountingDelivery" WHERE request_id IN ('
                'SELECT request_id FROM "LiteLLM_AccountingDelivery" '
                "WHERE state = 'applied' AND applied_at < CURRENT_TIMESTAMP - make_interval(days => $1::integer) "
                "ORDER BY applied_at, request_id LIMIT $2::integer FOR UPDATE SKIP LOCKED)",
                retention_days,
                batch_size,
            )

    async def apply_pending(
        self,
        apply: Callable[[LedgerTransaction, tuple[DeliveryProjection, ...]], Awaitable[None]],
        *,
        batch_size: int = 100,
        max_attempts: int = 5,
    ) -> DeliveryBatchResult:
        if batch_size < 1 or max_attempts < 1:
            raise ValueError("Delivery batch size and maximum attempts must be positive")
        claimed: tuple[_Delivery, ...] = ()  # rebind-ok: preserve claims for retry accounting after rollback
        try:
            async with self._database.tx(timeout=_TIMEOUT) as transaction:
                retries: Final = _DELIVERIES.validate_python(await transaction.query_raw(_CLAIM, 1, True))
                claimed = retries or _DELIVERIES.validate_python(  # rebind-ok: retain claimed identities after rollback
                    await transaction.query_raw(_CLAIM, batch_size, False)
                )
                if not claimed:
                    return DeliveryBatchResult()
                stored: Final = _ROWS.validate_python(
                    await transaction.query_raw(_LOAD, tuple(delivery.request_id for delivery in claimed))
                )
                rows: Final = {entry.row["request_id"]: entry.row for entry in stored}
                if len(rows) != len(claimed):
                    raise ValueError("Pending delivery has no stored SpendLogs row")
                await apply(
                    transaction,
                    tuple(DeliveryProjection(rows[delivery.request_id], delivery.previous_row) for delivery in claimed),
                )
                await transaction.execute_raw(_APPLIED, tuple(delivery.request_id for delivery in claimed))
            return DeliveryBatchResult(applied=len(claimed))
        except Exception as error:  # noqa: BLE001  # roll back all destinations before recording a retry
            if not claimed:
                raise
            return await self._record_failure(claimed, type(error).__name__, max_attempts)

    async def _record_failure(
        self, claimed: tuple[_Delivery, ...], error: str, max_attempts: int
    ) -> DeliveryBatchResult:
        async with self._database.tx(timeout=_TIMEOUT) as transaction:
            updates: Final = tuple(
                [
                    await self._record_row_failure(
                        transaction,
                        delivery,
                        error,
                        "parked" if len(claimed) == 1 and delivery.attempts + 1 >= max_attempts else "pending",
                    )
                    for delivery in claimed
                ]
            )
        return DeliveryBatchResult(failed=len(claimed), parked=tuple(chain.from_iterable(updates)), error=error)

    @staticmethod
    async def _record_row_failure(
        transaction: LedgerTransaction, delivery: _Delivery, error: str, state: Literal["pending", "parked"]
    ) -> tuple[str, ...]:
        updated: Final = _DELIVERIES.validate_python(
            await transaction.query_raw(_FAILED, delivery.request_id, delivery.attempts, error, state)
        )
        return tuple(row.request_id for row in updated) if state == "parked" else ()
