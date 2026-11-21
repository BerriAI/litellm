"""
Spend logs repository: the ledger write of request logs with their accounting deliveries, and the window
spend sums budget counters reseed from.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from enum import Enum
from itertools import groupby
from typing import TYPE_CHECKING, Final, Protocol

from pydantic import TypeAdapter
from typing_extensions import assert_never

from litellm.litellm_core_utils.core_helpers import require_attributes
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.repositories import rust_database
from litellm.repositories.accounting_delivery_repository import LedgerDatabase, LedgerTransaction
from litellm.repositories.table_repositories import PrismaTableRepository
from litellm.rust_bridge._native import RustSpendLogRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models  # noqa: F401  # used by quoted base-class subscripts


class _LedgerWriterClientView(Protocol):
    @property
    def writer_db(self) -> LedgerDatabase: ...


class _RawQueryWriterClientView(Protocol):
    @property
    def writer_db(self) -> LedgerTransaction: ...


_SPEND_LOG_FIELDS: Final = TypeAdapter(dict[str, object])


def _accounting_payload(value: object) -> Mapping[str, object]:
    return (
        _SPEND_LOG_FIELDS.validate_json(value) if isinstance(value, str) else _SPEND_LOG_FIELDS.validate_python(value)
    )


_ACCOUNTING_ONLY_FIELDS: Final = frozenset({"accounting_payload", "accounting_log_disabled", "accounting_delivery_id"})


def _settles_once(row: Mapping[str, object]) -> bool:
    """A batch or video cost row: update_database gives every other ledger row its own delivery id."""
    return "accounting_delivery_id" not in row


def _delivery_id(row: Mapping[str, object]) -> str:
    return str(row["accounting_delivery_id"]) if not _settles_once(row) else str(row["request_id"])


class SpendLogOwner(str, Enum):
    """Whose spend a window sum adds up, named by the budget window counter label."""

    KEY = "Key"
    TEAM = "Team"
    USER = "User"


class SpendLogsRepository(PrismaTableRepository["prisma_models.LiteLLM_SpendLogs"]):
    table_name = "litellm_spendlogs"

    async def window_spend(self, owner: SpendLogOwner, owner_id: str, since: datetime) -> float:
        """The spend ``owner_id`` logged at or after ``since``, zero when it logged none. Like the Prisma read it
        replaces, a naive ``since`` is UTC and the bound is cut to the millisecond."""
        repository: Final = rust_database.required_repository(RustSpendLogRepository)
        utc: Final = since.replace(tzinfo=timezone.utc) if since.tzinfo is None else since.astimezone(timezone.utc)
        bound: Final = utc.replace(microsecond=utc.microsecond // 1000 * 1000)
        match owner:
            case SpendLogOwner.KEY:
                return await repository.key_window_spend(owner_id, bound)
            case SpendLogOwner.TEAM:
                return await repository.team_window_spend(owner_id, bound)
            case SpendLogOwner.USER:
                return await repository.user_window_spend(owner_id, bound)
            case _:
                assert_never(owner)

    async def insert_with_deliveries(
        self, rows: Sequence[Mapping[str, object]], *, write_spend_logs: bool = True
    ) -> int:
        legacy: Final = tuple(row for row in rows if "accounting_payload" not in row)
        legacy_count: Final = await self.table.create_many(data=legacy, skip_duplicates=True) if legacy else 0
        ledger: Final = tuple(
            sorted(
                {_delivery_id(row): row for row in reversed(rows) if "accounting_payload" in row}.values(),
                key=_delivery_id,
            )
        )
        if not ledger:
            return legacy_count
        database: Final = require_attributes(self._connected_client(), _LedgerWriterClientView, "writer_db").writer_db
        async with database.tx(timeout=timedelta(seconds=60)) as transaction:
            accepted: Final = await self._insert_ledger(transaction, ledger, write_spend_logs=write_spend_logs)
        return legacy_count + accepted

    @staticmethod
    async def _insert_ledger(
        transaction: LedgerTransaction, rows: Sequence[Mapping[str, object]], *, write_spend_logs: bool
    ) -> int:
        from litellm.proxy.db.db_span import db_span
        from litellm.repositories.spend_log_insert import spend_log_insert_statement

        payloads: Final = tuple(
            {
                "delivery_id": _delivery_id(row),
                "request_id": row["request_id"],
                "accounting_payload": _accounting_payload(row["accounting_payload"]),
                "is_batch": _settles_once(row),
            }
            for row in rows
        )
        inserted_deliveries: Final = await transaction.query_raw(
            'INSERT INTO "LiteLLM_SpendLogDelivery" (delivery_id, request_id, accounting_payload, is_batch) '
            "SELECT delivery_id, request_id, accounting_payload, is_batch FROM jsonb_to_recordset($1::jsonb) "
            "AS inputs(delivery_id text, request_id text, accounting_payload jsonb, is_batch boolean) "
            "ON CONFLICT (delivery_id) DO NOTHING RETURNING delivery_id",
            safe_dumps(payloads, non_finite_as_null=True),
        )
        accepted_ids: Final = frozenset(str(row["delivery_id"]) for row in inserted_deliveries)
        charged_by_legacy: Final = await SpendLogsRepository._jobs_charged_by_legacy(
            transaction,
            tuple(_delivery_id(row) for row in rows if _settles_once(row) and _delivery_id(row) in accepted_ids),
        )
        if charged_by_legacy:
            await transaction.execute_raw(
                'DELETE FROM "LiteLLM_SpendLogDelivery" WHERE delivery_id = ANY($1::text[])', list(charged_by_legacy)
            )
        accepted: Final = accepted_ids - charged_by_legacy
        local_rows: Final = tuple(
            {key: value for key, value in row.items() if key not in _ACCOUNTING_ONLY_FIELDS}
            for row in rows
            if _delivery_id(row) in accepted and row.get("accounting_log_disabled") is not True
        )
        if write_spend_logs and local_rows:
            groups: Final = tuple(
                tuple(group)
                for _, group in groupby(
                    sorted(local_rows, key=lambda row: tuple(sorted(row))), key=lambda row: tuple(sorted(row))
                )
            )
            async with db_span("insert_spend_logs", "LiteLLM_SpendLogs"):
                for group in groups:
                    await transaction.query_raw(*spend_log_insert_statement(group))
        return len(accepted)

    @staticmethod
    async def _jobs_charged_by_legacy(transaction: LedgerTransaction, cost_row_ids: Sequence[str]) -> frozenset[str]:
        """Batches and videos a pre-ledger poll already charged through their SpendLogs cost row."""
        if not cost_row_ids:
            return frozenset()
        rows: Final = await transaction.query_raw(
            'SELECT DISTINCT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = ANY($1::text[]) '
            "AND status = 'success' AND spend > 0",
            list(cost_row_ids),
        )
        return frozenset(str(row["request_id"]) for row in rows)

    async def has_delivery(self, delivery_id: str) -> bool:
        database: Final = require_attributes(self._connected_client(), _RawQueryWriterClientView, "writer_db").writer_db
        rows: Final = await database.query_raw(
            'SELECT 1 FROM "LiteLLM_SpendLogDelivery" WHERE delivery_id = $1', delivery_id
        )
        return bool(rows)
