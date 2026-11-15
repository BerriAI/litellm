from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import timedelta
from itertools import chain
from types import TracebackType
from typing import TYPE_CHECKING, Final, Generic, Literal, Protocol, TypeAlias, TypeVar

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


_Applied: Final = TypeVar("_Applied")


@dataclass(frozen=True, slots=True)
class DeliveryBatchResult:
    applied: int = 0
    failed: int = 0
    parked: tuple[str, ...] = ()
    error: str | None = None
    repaired: tuple[str, ...] = ()


def _merged(first: DeliveryBatchResult, second: DeliveryBatchResult) -> DeliveryBatchResult:
    return DeliveryBatchResult(
        applied=first.applied + second.applied,
        failed=first.failed + second.failed,
        parked=first.parked + second.parked,
        error=first.error or second.error,
        repaired=first.repaired + second.repaired,
    )


def _progressed(result: DeliveryBatchResult) -> bool:
    return bool(result.applied or result.failed)


@dataclass(frozen=True, slots=True)
class _Committed(Generic[_Applied]):
    result: DeliveryBatchResult
    applied: tuple[_Applied, ...] = ()
    rejected: "tuple[Contribution, ...]" = ()


@dataclass(frozen=True, slots=True)
class _Failed:
    error: Exception


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
WHERE delivery_id = (
    SELECT delivery_id FROM "LiteLLM_SpendLogDelivery"
    WHERE delivery_id = $1 AND state = 'pending' AND attempts = $2::integer FOR UPDATE SKIP LOCKED
)
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
WHERE (delivery_id, attempts) IN (SELECT * FROM unnest($1::text[], $2::integer[])) AND state = 'pending'
ORDER BY delivery_id FOR UPDATE SKIP LOCKED"""
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


_Isolate: TypeAlias = Callable[
    [LedgerTransaction, DeliveryProjection], Awaitable["tuple[_Applied, tuple[Contribution, ...]]"]
]
_OnCommit: TypeAlias = Callable[[_Applied, "tuple[Contribution, ...]"], Awaitable[None]]


def _rows(claimed: tuple[tuple[_Delivery, DeliveryProjection], ...]) -> tuple[_Delivery, ...]:
    return tuple(row for row, _ in claimed)


def _projections(claimed: tuple[tuple[_Delivery, DeliveryProjection], ...]) -> tuple[DeliveryProjection, ...]:
    return tuple(projection for _, projection in claimed)


def _delivery_ids(claimed: tuple[tuple[_Delivery, DeliveryProjection], ...]) -> tuple[str, ...]:
    return tuple(row.delivery_id for row, _ in claimed)


async def _relocked(
    transaction: LedgerTransaction, claimed: tuple[tuple[_Delivery, DeliveryProjection], ...]
) -> tuple[tuple[_Delivery, DeliveryProjection], ...]:
    """The claims still pending at the attempt count they were claimed at, locked for this transaction. Another
    worker that took one in the meantime changed its state or attempts, so it is left to that worker."""
    rows: Final = _rows(claimed)
    locked: Final = frozenset(
        row.delivery_id
        for row in _DELIVERIES.validate_python(
            await transaction.query_raw(
                _RELOCK, tuple(row.delivery_id for row in rows), tuple(row.attempts for row in rows)
            )
        )
    )
    return tuple(pair for pair in claimed if pair[0].delivery_id in locked)


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
        apply: Callable[[LedgerTransaction, tuple[DeliveryProjection, ...]], Awaitable[_Applied]],
        *,
        isolate: "_Isolate[_Applied] | None" = None,
        on_commit: "_OnCommit[_Applied] | None" = None,
        continue_splitting: Callable[[], bool] = lambda: True,
        batch_size: int = 100,
        max_attempts: int = 5,
        retry_only: bool | None = None,
    ) -> DeliveryBatchResult:
        if batch_size < 1 or max_attempts < 1:
            raise ValueError("Delivery batch size and maximum attempts must be positive")
        claimed: tuple[tuple[_Delivery, DeliveryProjection], ...] = ()  # rebind-ok: claims survive the rollback

        async def claim_and_apply(transaction: LedgerTransaction) -> _Committed[_Applied]:
            nonlocal claimed
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
            if undecodable:
                await transaction.execute_raw(_UNDECODABLE, undecodable)
            claimed = tuple(  # rebind-ok: retain claimed identities after rollback
                (row, projection) for row, projection in decoded if projection is not None
            )
            if not claimed:
                return _Committed(DeliveryBatchResult(parked=undecodable))
            applied: Final = await apply(transaction, _projections(claimed))
            await transaction.execute_raw(_APPLIED, _delivery_ids(claimed))
            return _Committed(DeliveryBatchResult(applied=len(claimed), parked=undecodable), (applied,))

        recovery: Final = _Recovery(self._database, apply, on_commit, continue_splitting, max_attempts)
        outcome: Final = await _transact(self._database, claim_and_apply)
        if isinstance(outcome, _Failed) and not claimed:
            raise outcome.error
        return await recovery.settle(outcome, claimed, isolate, must_progress=True)


async def _transact(
    database: LedgerDatabase, work: Callable[[LedgerTransaction], Awaitable[_Committed[_Applied]]]
) -> _Committed[_Applied] | _Failed:
    """One transaction's outcome as a value, so recovery and notification never run inside an exception handler
    and cannot mistake an unrelated later error for this one."""
    try:
        async with database.tx(timeout=_TIMEOUT) as transaction:
            return await work(transaction)
    except Exception as error:  # noqa: BLE001  # roll back all destinations before recording a retry
        return _Failed(error)


async def _record_failure(
    database: LedgerDatabase, claimed: tuple[_Delivery, ...], error: Exception, max_attempts: int
) -> DeliveryBatchResult:
    parks: Final = len(claimed) == 1 and _is_permanent_failure(error)
    async with database.tx(timeout=_TIMEOUT) as transaction:
        updates: Final = tuple(
            [
                await _record_row_failure(
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


async def _record_row_failure(
    transaction: LedgerTransaction, delivery: _Delivery, error: str, state: Literal["pending", "parked"]
) -> tuple[str, ...]:
    updated: Final = _DELIVERIES.validate_python(
        await transaction.query_raw(_FAILED, delivery.delivery_id, delivery.attempts, error, state)
    )
    return tuple(row.delivery_id for row in updated) if state == "parked" else ()


@dataclass(frozen=True, slots=True)
class _Recovery(Generic[_Applied]):
    """Settles one transaction's outcome. A batch Postgres refused for an oversized index entry is halved until the
    delivery carrying the value is alone and applied contribution by contribution, while continue_splitting allows.
    Once it does not, a call that has not committed anything yet still isolates one delivery so every flush makes
    progress, and the rest stay pending untouched. Each commit is reported after its transaction, never inside it."""

    database: LedgerDatabase
    apply: Callable[[LedgerTransaction, tuple[DeliveryProjection, ...]], Awaitable[_Applied]]
    on_commit: "_OnCommit[_Applied] | None"
    continue_splitting: Callable[[], bool]
    max_attempts: int

    async def settle(
        self,
        outcome: _Committed[_Applied] | _Failed,
        claimed: tuple[tuple[_Delivery, DeliveryProjection], ...],
        isolate: "_Isolate[_Applied] | None",
        *,
        must_progress: bool,
    ) -> DeliveryBatchResult:
        if isinstance(outcome, _Committed):
            return await self._notified(outcome)
        if isolate is None or not _is_index_row_size_failure(outcome.error):
            return await _record_failure(self.database, _rows(claimed), outcome.error, self.max_attempts)
        return await self._split(claimed, isolate, must_progress=must_progress)

    async def _split(
        self,
        claimed: tuple[tuple[_Delivery, DeliveryProjection], ...],
        isolate: "_Isolate[_Applied]",
        *,
        must_progress: bool,
    ) -> DeliveryBatchResult:
        if len(claimed) == 1:
            if must_progress or self.continue_splitting():
                return await self._isolate_one(claimed[0], isolate)
            return DeliveryBatchResult()
        middle: Final = len(claimed) // 2
        first: Final = await self._subset(claimed[:middle], isolate, must_progress=must_progress)
        second: Final = await self._subset(
            claimed[middle:], isolate, must_progress=must_progress and not _progressed(first)
        )
        return _merged(first, second)

    async def _subset(
        self,
        claimed: tuple[tuple[_Delivery, DeliveryProjection], ...],
        isolate: "_Isolate[_Applied]",
        *,
        must_progress: bool,
    ) -> DeliveryBatchResult:
        if not self.continue_splitting():
            return await self._isolate_one(claimed[0], isolate) if must_progress else DeliveryBatchResult()

        async def apply_held(transaction: LedgerTransaction) -> _Committed[_Applied]:
            held: Final = await _relocked(transaction, claimed)
            if not held:
                return _Committed(DeliveryBatchResult())
            applied: Final = await self.apply(transaction, _projections(held))
            await transaction.execute_raw(_APPLIED, _delivery_ids(held))
            return _Committed(DeliveryBatchResult(applied=len(held)), (applied,))

        outcome: Final = await _transact(self.database, apply_held)
        return await self.settle(outcome, claimed, isolate, must_progress=must_progress)

    async def _isolate_one(
        self, claimed: tuple[_Delivery, DeliveryProjection], isolate: "_Isolate[_Applied]"
    ) -> DeliveryBatchResult:
        """Commits what Postgres accepts together with a parked record of what it rejected and the delivery's
        receipt, so the delivery is never charged twice and nothing rejected is silently lost."""
        delivery, projection = claimed

        async def apply_isolated(transaction: LedgerTransaction) -> _Committed[_Applied]:
            if not await _relocked(transaction, (claimed,)):
                return _Committed(DeliveryBatchResult())
            applied, rejected = await isolate(transaction, projection)
            repair_id: Final = repair_delivery_id(delivery.delivery_id)
            if rejected:
                await transaction.execute_raw(_REPAIR, delivery.delivery_id, repair_id, _repair_payload(rejected))
            await transaction.execute_raw(_APPLIED, (delivery.delivery_id,))
            return _Committed(
                DeliveryBatchResult(applied=1, repaired=(repair_id,) if rejected else ()), (applied,), rejected
            )

        outcome: Final = await _transact(self.database, apply_isolated)
        if isinstance(outcome, _Failed):
            return await _record_failure(self.database, (delivery,), outcome.error, self.max_attempts)
        return await self._notified(outcome)

    async def _notified(self, outcome: _Committed[_Applied]) -> DeliveryBatchResult:
        if self.on_commit is not None and outcome.applied:
            await self.on_commit(outcome.applied[0], outcome.rejected)
        return outcome.result
