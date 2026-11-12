import os
import time
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    Protocol,
    cast,  # noqa: TID251  # generated Prisma transaction signatures cannot express the borrowed transaction protocol
)

from litellm._logging import verbose_proxy_logger
from litellm.accounting.application import apply_deliveries
from litellm.accounting.metrics import record_backlog, record_storage
from litellm.accounting.projection import Contribution
from litellm.repositories.accounting_delivery_repository import (
    AccountingDeliveryRepository,
    DeliveryBatchResult,
    DeliveryProjection,
    LedgerDatabase,
    LedgerTransaction,
)
from litellm.types.integrations.slack_alerting import AlertType

if TYPE_CHECKING:
    from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
    from litellm.proxy.utils import PrismaClient, ProxyLogging


class _AccountingAlerts(Protocol):
    async def alerting_handler(
        self, message: str, level: Literal["Low", "Medium", "High"], alert_type: AlertType
    ) -> None: ...


_JOB: Final = "accounting_delivery"


def _filled(result: DeliveryBatchResult, batch_size: int) -> bool:
    return result.applied + len(result.parked) >= batch_size


def _progressed(result: DeliveryBatchResult) -> bool:
    return bool(result.applied or result.failed)


async def flush_accounting_deliveries(
    writer: "DBSpendUpdateWriter", prisma_client: "PrismaClient", proxy_logging_obj: "ProxyLogging"
) -> None:
    applied: tuple[Contribution, ...] = ()  # rebind-ok: keep committed contributions for cache invalidation

    async def apply(transaction: LedgerTransaction, deliveries: tuple[DeliveryProjection, ...]) -> None:
        nonlocal applied
        applied = await apply_deliveries(  # rebind-ok: invalidate caches only after the outer transaction commits
            transaction, deliveries, prisma_client=prisma_client, writer=writer, proxy_logging_obj=proxy_logging_obj
        )

    database: Final = cast(  # cast-ok: PrismaClient.tx owns writer routing and yields the generated transaction
        LedgerDatabase, prisma_client
    )
    repository: Final = AccountingDeliveryRepository(database)
    started: Final = time.monotonic()
    batch_size: Final = int(os.environ.get("LITELLM_ACCOUNTING_DELIVERY_BATCH_SIZE", "500"))

    async def run_batch(*, retry_only: bool) -> DeliveryBatchResult:
        result: Final = await repository.apply_pending(apply, batch_size=batch_size, retry_only=retry_only)
        if result.applied:
            await writer.invalidate_accounting_entities(applied, proxy_logging_obj)
        if result.parked:
            verbose_proxy_logger.error(
                "Accounting delivery parked %d requests after isolated failures", len(result.parked)
            )
            await _alert(
                proxy_logging_obj,
                message=f"Accounting delivery parked {len(result.parked)} requests. Included spend totals require repair.",
            )
        return result

    fresh_deadline: Final = time.monotonic() + 5
    for _ in range(batch_size):
        if not _filled(await run_batch(retry_only=False), batch_size) or time.monotonic() >= fresh_deadline:
            break
    retry_deadline: Final = time.monotonic() + 5
    for _ in range(batch_size):
        if time.monotonic() >= retry_deadline or not _progressed(await run_batch(retry_only=True)):
            break
    acquired: Final = await writer.pod_lock_manager.acquire_lock(cronjob_id=_JOB, allow_reentrant=False)
    if acquired is False:
        return
    try:
        from litellm.task_insights.observation import task_insights_enabled

        observing: Final = task_insights_enabled()
        if observing:
            await _observe_task_insights(prisma_client, batch_size)
        retention_days: Final = int(os.environ.get("LITELLM_ACCOUNTING_DELIVERY_RETENTION_DAYS", "1"))
        cleanup_deadline: Final = time.monotonic() + 5
        for _ in range(batch_size):
            if (
                await repository.cleanup_applied(
                    retention_days=retention_days, batch_size=batch_size * 2, keep_unobserved=observing
                )
                < batch_size * 2
                or time.monotonic() >= cleanup_deadline
            ):
                break
        await _report(writer, repository, proxy_logging_obj, time.monotonic() - started)
    finally:
        if acquired:
            await writer.pod_lock_manager.release_lock(cronjob_id=_JOB)


async def _observe_task_insights(prisma_client: "PrismaClient", batch_size: int) -> None:
    """Analytics run after billing committed, so a failure here is logged and never touches spend."""
    from litellm.repositories.task_insights_repository import TaskInsightsDatabase, TaskInsightsRepository
    from litellm.task_insights.observation import observe_applied_deliveries

    database: Final = cast(  # cast-ok: the writer wrapper delegates query_raw, execute_raw and tx to Prisma
        TaskInsightsDatabase, prisma_client.writer_db
    )
    try:
        observed: Final = await observe_applied_deliveries(
            TaskInsightsRepository(database),
            now=lambda: datetime.now(timezone.utc),
            batch_size=min(batch_size, 200),
            max_batches=10,
        )
    except Exception:  # noqa: BLE001  # unobserved deliveries stay retained and the next flush retries them
        verbose_proxy_logger.exception("Task Insights observation failed; deliveries are retained for the next flush")
        return
    if observed:
        verbose_proxy_logger.debug("Task Insights observed %d applied deliveries", observed)


async def _report(
    writer: "DBSpendUpdateWriter",
    repository: AccountingDeliveryRepository,
    proxy_logging_obj: _AccountingAlerts,
    seconds: float,
) -> None:
    backlog: Final = await repository.backlog()
    record_backlog(backlog, seconds)
    interval: Final = float(os.environ.get("LITELLM_ACCOUNTING_DELIVERY_CHECK_INTERVAL_SECONDS", "900"))
    if writer.accounting_storage_checks.due(interval):
        storage: Final = await repository.storage()
        record_storage(storage)
        if storage.batch_count >= int(
            os.environ.get("LITELLM_ACCOUNTING_BATCH_DELIVERY_ALERT_COUNT", "1000000")
        ) or storage.table_bytes >= int(os.environ.get("LITELLM_ACCOUNTING_DELIVERY_ALERT_BYTES", "1073741824")):
            await _alert(
                proxy_logging_obj,
                message=f"Accounting delivery storage requires review. Retained batches={storage.batch_count}, table bytes={storage.table_bytes}.",
            )
    if backlog.oldest_pending_seconds > 300 and writer.accounting_delay_alerts.due(interval):
        await _alert(
            proxy_logging_obj,
            message=f"Accounting deliveries are delayed. Pending={backlog.pending_count}, oldest={backlog.oldest_pending_seconds:.0f}s.",
        )


async def _alert(alerts: _AccountingAlerts, message: str) -> None:
    await alerts.alerting_handler(message=message, level="High", alert_type=AlertType.failed_tracking_spend)
