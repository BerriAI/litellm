from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import timedelta
from itertools import chain
from types import TracebackType
from typing import TYPE_CHECKING, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.repositories.prisma_protocols import TableActions

if TYPE_CHECKING:
    from litellm.accounting.projection import Contribution
    from litellm.proxy.db.db_spend_update_writer import SpendBatchManager


class LedgerTransaction(Protocol):
    @property
    def litellm_dailycustomermemberspend(self) -> TableActions[object]: ...

    @property
    def litellm_customerorganizationtable(self) -> TableActions[object]: ...

    @property
    def litellm_customerteamtable(self) -> TableActions[object]: ...

    @property
    def litellm_customermembertable(self) -> TableActions[object]: ...

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

    delivery_id: str
    attempts: int


class _ClaimedDelivery(_Delivery):
    accounting_payload: str


@dataclass(frozen=True, slots=True)
class DeliveryProjection:
    row: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class DeliveryBatchResult:
    applied: int = 0
    failed: int = 0
    parked: tuple[str, ...] = ()
    error: str | None = None
    repaired: tuple[str, ...] = ()


class DeliveryBacklog(BaseModel):
    model_config = ConfigDict(frozen=True)

    pending_count: int
    parked_count: int
    oldest_pending_seconds: float


class DeliveryStorage(BaseModel):
    model_config = ConfigDict(frozen=True)

    batch_count: int
    table_bytes: int


_DELIVERIES: Final = TypeAdapter(tuple[_Delivery, ...])
_CLAIMED: Final = TypeAdapter(tuple[_ClaimedDelivery, ...])
_PAYLOAD: Final = TypeAdapter(dict[str, object])
_TIMEOUT: Final = timedelta(seconds=60)
_CLAIM: Final = """SELECT delivery_id, attempts, accounting_payload::text AS accounting_payload
FROM "LiteLLM_SpendLogDelivery"
WHERE state = 'pending' AND next_attempt_at <= CURRENT_TIMESTAMP
  AND (attempts > 0) = $2::boolean
ORDER BY next_attempt_at, delivery_id LIMIT $1::integer FOR UPDATE SKIP LOCKED"""
_APPLIED: Final = """UPDATE "LiteLLM_SpendLogDelivery"
SET state = 'applied', applied_at = CURRENT_TIMESTAMP, last_error = NULL
WHERE delivery_id = ANY($1::text[]) AND state = 'pending' """
_FAILED: Final = """UPDATE "LiteLLM_SpendLogDelivery"
SET attempts = attempts + 1, last_error = $3,
    next_attempt_at = CURRENT_TIMESTAMP + make_interval(secs => LEAST(300, power(2, LEAST(attempts, 8)) * (0.5 + random()))),
    state = $4::"AccountingDeliveryState"
WHERE delivery_id = $1 AND state = 'pending' AND attempts = $2::integer
RETURNING delivery_id, attempts"""
_BACKLOG: Final = """SELECT COUNT(*) FILTER (WHERE state = 'pending') AS pending_count,
    COUNT(*) FILTER (WHERE state = 'parked') AS parked_count,
    COALESCE(EXTRACT(EPOCH FROM CURRENT_TIMESTAMP - MIN(created_at) FILTER (WHERE state = 'pending')), 0)
        AS oldest_pending_seconds
FROM "LiteLLM_SpendLogDelivery"
WHERE state IN ('pending', 'parked')"""
_STORAGE: Final = """SELECT (SELECT COUNT(*) FROM "LiteLLM_SpendLogDelivery" WHERE is_batch) AS batch_count,
    pg_total_relation_size('"LiteLLM_SpendLogDelivery"') AS table_bytes"""
_RELOCK: Final = """SELECT delivery_id, attempts FROM "LiteLLM_SpendLogDelivery"
WHERE delivery_id = $1 AND state = 'pending' AND attempts = $2::integer FOR UPDATE SKIP LOCKED"""
_REPAIR: Final = """INSERT INTO "LiteLLM_SpendLogDelivery"
    (delivery_id, request_id, state, attempts, last_error, accounting_payload, is_batch)
SELECT $2, $2, 'parked', 1, 'IndexRowSizeExceeded',
    jsonb_build_object('accounting_repair',
        $3::jsonb || jsonb_build_object('delivery_id', delivery_id, 'request_id', request_id, 'is_batch', is_batch)),
    false
FROM "LiteLLM_SpendLogDelivery" WHERE delivery_id = $1"""
_REPAIR_MARKER: Final = "accounting_repair"
_UNDECODABLE: Final = """UPDATE "LiteLLM_SpendLogDelivery"
SET state = 'parked', attempts = attempts + 1, last_error = 'UndecodablePayload'
WHERE delivery_id = ANY($1::text[]) AND state = 'pending'"""


def _decoded(delivery: _ClaimedDelivery) -> DeliveryProjection | None:
    try:
        payload: Final = _PAYLOAD.validate_json(delivery.accounting_payload)
    except ValueError:
        return None
    return None if _REPAIR_MARKER in payload else DeliveryProjection(payload)


def repair_delivery_id(delivery_id: str) -> str:
    return f"{delivery_id}:index-repair"


def _repair_payload(rejected: "tuple[Contribution, ...]") -> str:
    """The contributions Postgres could not index. The parked record nests them under its marker with the source
    receipt's identity and no top-level charge fields, so even a reader that ignores the marker projects nothing."""
    return safe_dumps(
        {"reason": "IndexRowSizeExceeded", "rejected": [asdict(item) for item in rejected]}, non_finite_as_null=True
    )


def _is_index_row_size_failure(error: Exception) -> bool:
    from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler

    return PrismaDBExceptionHandler.is_index_row_size_error(error)


def _is_permanent_failure(error: Exception) -> bool:
    """A delivery whose own data failed fails the same way on every retry, while any other error is the database's
    and clears once it recovers."""
    from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler

    if PrismaDBExceptionHandler.is_database_service_unavailable_error(error):
        return False
    return isinstance(error, (ValueError, TypeError, KeyError)) or PrismaDBExceptionHandler.is_rejected_row_error(error)


class AccountingDeliveryRepository:
    def __init__(self, database: LedgerDatabase) -> None:
        self._database = database

    async def backlog(self) -> DeliveryBacklog:
        async with self._database.tx(timeout=_TIMEOUT) as transaction:
            return DeliveryBacklog.model_validate((await transaction.query_raw(_BACKLOG))[0])

    async def storage(self) -> DeliveryStorage:
        async with self._database.tx(timeout=_TIMEOUT) as transaction:
            return DeliveryStorage.model_validate((await transaction.query_raw(_STORAGE))[0])

    async def cleanup_applied(self, *, retention_days: int = 1, batch_size: int = 1000) -> int:
        if retention_days < 1 or batch_size < 1:
            raise ValueError("Delivery retention and cleanup batch size must be positive")
        async with self._database.tx(timeout=_TIMEOUT) as transaction:
            return await transaction.execute_raw(
                'DELETE FROM "LiteLLM_SpendLogDelivery" WHERE delivery_id IN ('
                'SELECT delivery_id FROM "LiteLLM_SpendLogDelivery" '
                "WHERE state = 'applied' AND NOT is_batch AND applied_at < CURRENT_TIMESTAMP - make_interval(days => $1::integer) "
                "ORDER BY applied_at, delivery_id LIMIT $2::integer FOR UPDATE SKIP LOCKED)",
                retention_days,
                batch_size,
            )

    async def apply_pending(
        self,
        apply: Callable[[LedgerTransaction, tuple[DeliveryProjection, ...]], Awaitable[None]],
        *,
        isolate: "Callable[[LedgerTransaction, DeliveryProjection], Awaitable[tuple[Contribution, ...]]] | None" = None,
        batch_size: int = 100,
        max_attempts: int = 5,
        retry_only: bool | None = None,
    ) -> DeliveryBatchResult:
        if batch_size < 1 or max_attempts < 1:
            raise ValueError("Delivery batch size and maximum attempts must be positive")
        claimed: tuple[_Delivery, ...] = ()  # rebind-ok: preserve claims for retry accounting after rollback
        projections: tuple[DeliveryProjection, ...] = ()  # rebind-ok: preserve claims for isolation after rollback
        try:
            async with self._database.tx(timeout=_TIMEOUT) as transaction:
                retries: Final = (
                    _CLAIMED.validate_python(await transaction.query_raw(_CLAIM, 1, True))
                    if retry_only is not False
                    else ()
                )
                rows: Final = retries or (
                    _CLAIMED.validate_python(await transaction.query_raw(_CLAIM, batch_size, False))
                    if retry_only is not True
                    else ()
                )
                decoded: Final = tuple((row, _decoded(row)) for row in rows)
                undecodable: Final = tuple(row.delivery_id for row, projection in decoded if projection is None)
                applicable: Final = tuple((row, projection) for row, projection in decoded if projection is not None)
                if undecodable:
                    await transaction.execute_raw(_UNDECODABLE, undecodable)
                claimed = tuple(row for row, _ in applicable)  # rebind-ok: retain claimed identities after rollback
                projections = tuple(projection for _, projection in applicable)  # rebind-ok: retained after rollback
                if claimed:
                    await apply(transaction, tuple(projection for _, projection in applicable))
                    await transaction.execute_raw(_APPLIED, tuple(row.delivery_id for row in claimed))
            return DeliveryBatchResult(applied=len(claimed), parked=undecodable)
        except Exception as error:  # noqa: BLE001  # roll back all destinations before recording a retry
            if not claimed:
                raise
            if isolate is not None and len(claimed) == 1 and _is_index_row_size_failure(error):
                return await self._apply_isolated(claimed[0], projections[0], isolate, max_attempts)
            return await self._record_failure(claimed, error, max_attempts)

    async def _apply_isolated(
        self,
        delivery: _Delivery,
        projection: DeliveryProjection,
        isolate: "Callable[[LedgerTransaction, DeliveryProjection], Awaitable[tuple[Contribution, ...]]]",
        max_attempts: int,
    ) -> DeliveryBatchResult:
        """Commits what Postgres accepts together with a parked record of what it rejected and the delivery's
        receipt, so the delivery is never charged twice and nothing rejected is silently lost."""
        try:
            async with self._database.tx(timeout=_TIMEOUT) as transaction:
                if not await transaction.query_raw(_RELOCK, delivery.delivery_id, delivery.attempts):
                    return DeliveryBatchResult()
                rejected: Final = await isolate(transaction, projection)
                repair_id: Final = repair_delivery_id(delivery.delivery_id)
                recorded: Final = (
                    await transaction.execute_raw(_REPAIR, delivery.delivery_id, repair_id, _repair_payload(rejected))
                    if rejected
                    else 1
                )
                if recorded != 1:
                    raise RuntimeError("Could not record the rejected contributions of an accounting delivery")
                await transaction.execute_raw(_APPLIED, (delivery.delivery_id,))
            return DeliveryBatchResult(applied=1, repaired=(repair_id,) if rejected else ())
        except Exception as error:  # noqa: BLE001  # roll back all destinations before recording a retry
            return await self._record_failure((delivery,), error, max_attempts)

    async def _record_failure(
        self, claimed: tuple[_Delivery, ...], error: Exception, max_attempts: int
    ) -> DeliveryBatchResult:
        parks: Final = len(claimed) == 1 and _is_permanent_failure(error)
        async with self._database.tx(timeout=_TIMEOUT) as transaction:
            updates: Final = tuple(
                [
                    await self._record_row_failure(
                        transaction,
                        delivery,
                        type(error).__name__,
                        "parked" if parks and delivery.attempts + 1 >= max_attempts else "pending",
                    )
                    for delivery in claimed
                ]
            )
        return DeliveryBatchResult(
            failed=len(claimed), parked=tuple(chain.from_iterable(updates)), error=type(error).__name__
        )

    @staticmethod
    async def _record_row_failure(
        transaction: LedgerTransaction, delivery: _Delivery, error: str, state: Literal["pending", "parked"]
    ) -> tuple[str, ...]:
        updated: Final = _DELIVERIES.validate_python(
            await transaction.query_raw(_FAILED, delivery.delivery_id, delivery.attempts, error, state)
        )
        return tuple(row.delivery_id for row in updated) if state == "parked" else ()
