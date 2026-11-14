from typing import TYPE_CHECKING, Final

from litellm.accounting.delivery_projection import entity_transactions, project_deliveries
from litellm.accounting.projection import Contribution
from litellm.proxy.db.budget_window_spend_writer import commit_window_spend_updates
from litellm.proxy.db.daily_spend_bulk_upsert import DAILY_SPEND_TABLES, build_bulk_upsert, merge_by_conflict_key
from litellm.proxy.db.db_span import db_span
from litellm.proxy.db.db_transaction_queue.window_spend_update_queue import WindowSpendTransaction
from litellm.proxy.db.model_usage_rollup import queue_model_usage_upsert
from litellm.proxy.db.rollup_lock_timeout import apply_rollup_lock_timeout
from litellm.repositories.accounting_delivery_repository import DeliveryProjection, LedgerTransaction

if TYPE_CHECKING:
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
    from litellm.proxy.utils import PrismaClient, ProxyLogging


_SAVEPOINT: Final = "accounting_contribution"


async def apply_deliveries(
    transaction: LedgerTransaction,
    deliveries: tuple[DeliveryProjection, ...],
    *,
    prisma_client: "PrismaClient",
    writer: "DBSpendUpdateWriter",
    proxy_logging_obj: "ProxyLogging",
) -> tuple[Contribution, ...]:
    contributions: Final = project_deliveries(deliveries)
    await apply_rollup_lock_timeout(transaction)
    await _apply_contributions(
        transaction, contributions, prisma_client=prisma_client, writer=writer, proxy_logging_obj=proxy_logging_obj
    )
    return contributions


async def apply_delivery_isolated(
    transaction: LedgerTransaction,
    delivery: DeliveryProjection,
    *,
    prisma_client: "PrismaClient",
    writer: "DBSpendUpdateWriter",
    proxy_logging_obj: "ProxyLogging",
) -> tuple[tuple[Contribution, ...], tuple[Contribution, ...]]:
    """Applies one delivery a contribution at a time, so a value Postgres cannot index loses only its own
    contribution. Returns the applied and the rejected contributions; any other failure propagates."""
    contributions: Final = project_deliveries((delivery,))
    await apply_rollup_lock_timeout(transaction)
    await writer.lock_charged_teams(transaction, entity_transactions(contributions))
    outcomes: Final = tuple(
        [
            (
                item,
                await _applied_within_savepoint(
                    transaction, item, prisma_client=prisma_client, writer=writer, proxy_logging_obj=proxy_logging_obj
                ),
            )
            for item in contributions
        ]
    )
    applied: Final = tuple(item for item, was_applied in outcomes if was_applied)
    rejected: Final = tuple(item for item, was_applied in outcomes if not was_applied)
    return applied, rejected


async def _applied_within_savepoint(
    transaction: LedgerTransaction,
    contribution: Contribution,
    *,
    prisma_client: "PrismaClient",
    writer: "DBSpendUpdateWriter",
    proxy_logging_obj: "ProxyLogging",
) -> bool:
    from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler

    await transaction.execute_raw(f"SAVEPOINT {_SAVEPOINT}")
    try:
        await _apply_contributions(
            transaction,
            (contribution,),
            prisma_client=prisma_client,
            writer=writer,
            proxy_logging_obj=proxy_logging_obj,
        )
    except Exception as error:
        if not PrismaDBExceptionHandler.is_index_row_size_error(error):
            raise
        await transaction.execute_raw(f"ROLLBACK TO SAVEPOINT {_SAVEPOINT}")
        return False
    await transaction.execute_raw(f"RELEASE SAVEPOINT {_SAVEPOINT}")
    return True


async def _apply_contributions(
    transaction: LedgerTransaction,
    contributions: tuple[Contribution, ...],
    *,
    prisma_client: "PrismaClient",
    writer: "DBSpendUpdateWriter",
    proxy_logging_obj: "ProxyLogging",
) -> None:
    from litellm.proxy.guardrails.usage_tracking import UsageUnitIncrement, UsageUnitKey, upsert_usage_unit_row

    await writer.apply_accounting_entities(
        prisma_client, proxy_logging_obj, transaction, entity_transactions(contributions)
    )
    for table in sorted(DAILY_SPEND_TABLES.values(), key=lambda item: item.name):
        rows = tuple(
            {**dict(item.key), **dict(item.values), **dict(item.increments)}
            for item in contributions
            if item.table == table.name
        )
        merged = merge_by_conflict_key(table, rows)
        for offset in range(0, len(merged), 100):
            sql, params = build_bulk_upsert(table, merged[offset : offset + 100])
            async with db_span("upsert_daily_spend", table.name):
                await transaction.execute_raw(sql, *params)
    async with transaction.batch_() as batcher:
        for item in contributions:
            if item.table == "LiteLLM_DailyModelUsage":
                queue_model_usage_upsert(batcher, dict(item.key), dict(item.increments))
    for item in contributions:
        if item.table == "LiteLLM_DailyGuardrailUsageUnits":
            key = dict(item.key)
            counters = dict(item.increments)
            await upsert_usage_unit_row(
                prisma_client,
                UsageUnitKey(key["guardrail_id"], key["date"], key["team_id"], key["api_key"], key["usage_unit"]),
                UsageUnitIncrement(int(counters["units"]), float(counters["cost"]), int(counters["untracked_units"])),
                outer_transaction=transaction,
            )
    windows: Final = tuple(_window(item) for item in contributions if item.table == "LiteLLM_BudgetWindowSpend")
    await commit_window_spend_updates(prisma_client, windows, outer_transaction=transaction)


def _window(item: Contribution) -> WindowSpendTransaction:
    key: Final = dict(item.key)
    values: Final = dict(item.values)
    return WindowSpendTransaction(
        entity_type=key["entity_type"],
        entity_id=key["entity_id"],
        window_duration=key["window_duration"],
        window_start=values["window_start"] or "",
        spend=float(dict(item.increments)["spend"]),
        started_at=values["started_at"],
    )
