from typing import TYPE_CHECKING, Final

from litellm.accounting.delivery_projection import entity_transactions, project_deliveries
from litellm.accounting.projection import Contribution
from litellm.proxy.db.budget_window_spend_writer import commit_window_spend_updates
from litellm.proxy.db.daily_spend_bulk_upsert import DAILY_SPEND_TABLES, build_bulk_upsert, merge_by_conflict_key
from litellm.proxy.db.db_transaction_queue.window_spend_update_queue import WindowSpendTransaction
from litellm.proxy.db.model_usage_rollup import queue_model_usage_upsert
from litellm.repositories.accounting_delivery_repository import DeliveryProjection, LedgerTransaction

if TYPE_CHECKING:
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
    from litellm.proxy.utils import PrismaClient, ProxyLogging


async def apply_deliveries(
    transaction: LedgerTransaction,
    deliveries: tuple[DeliveryProjection, ...],
    *,
    prisma_client: "PrismaClient",
    writer: "DBSpendUpdateWriter",
    proxy_logging_obj: "ProxyLogging",
) -> tuple[Contribution, ...]:
    from litellm.proxy.guardrails.usage_tracking import UsageUnitIncrement, UsageUnitKey, upsert_usage_unit_row

    contributions: Final = project_deliveries(deliveries)
    await writer.apply_accounting_entities(
        prisma_client, proxy_logging_obj, transaction, entity_transactions(contributions)
    )
    for table in sorted(DAILY_SPEND_TABLES.values(), key=lambda item: item.name):
        rows = tuple(
            {**dict(item.key), **dict(item.values), **dict(item.increments)}
            for item in contributions
            if item.table == table.name
        )
        if rows:
            sql, params = build_bulk_upsert(table, merge_by_conflict_key(table, rows))
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
    return contributions


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
