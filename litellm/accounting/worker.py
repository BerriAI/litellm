import time
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    Protocol,
    cast,  # noqa: TID251  # generated Prisma transaction signatures cannot express the borrowed transaction protocol
)

from litellm._logging import verbose_proxy_logger
from litellm.accounting.application import apply_deliveries
from litellm.accounting.metrics import record_deliveries, record_postponed_cleanup
from litellm.accounting.projection import Contribution
from litellm.repositories.accounting_delivery_repository import (
    AccountingDeliveryRepository,
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


async def flush_accounting_deliveries(
    writer: "DBSpendUpdateWriter", prisma_client: "PrismaClient", proxy_logging_obj: "ProxyLogging"
) -> None:
    acquired: Final = await writer.pod_lock_manager.acquire_lock(cronjob_id=_JOB, allow_reentrant=False)
    if acquired is False:
        return
    applied: tuple[Contribution, ...] = ()  # rebind-ok: keep committed contributions for cache invalidation

    async def apply(transaction: LedgerTransaction, deliveries: tuple[DeliveryProjection, ...]) -> None:
        nonlocal applied
        applied = await apply_deliveries(  # rebind-ok: invalidate caches only after the outer transaction commits
            transaction, deliveries, prisma_client=prisma_client, writer=writer, proxy_logging_obj=proxy_logging_obj
        )

    try:
        database: Final = cast(  # cast-ok: PrismaClient.tx owns writer routing and yields the generated transaction
            LedgerDatabase, prisma_client
        )
        repository: Final = AccountingDeliveryRepository(database)
        started: Final = time.monotonic()
        result: Final = await repository.apply_pending(apply)
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
        await repository.cleanup_applied()
        metrics: Final = await repository.metrics()
        record_deliveries(metrics, time.monotonic() - started)
        if metrics.oldest_pending_seconds > 300:
            await _alert(
                proxy_logging_obj,
                message=f"Accounting deliveries are delayed. Pending={metrics.pending_count}, oldest={metrics.oldest_pending_seconds:.0f}s.",
            )
    finally:
        if acquired:
            await writer.pod_lock_manager.release_lock(cronjob_id=_JOB)


async def _alert(alerts: _AccountingAlerts, message: str) -> None:
    await alerts.alerting_handler(message=message, level="High", alert_type=AlertType.failed_tracking_spend)


async def alert_postponed_cleanup(kind: Literal["rows", "partition"]) -> None:
    from litellm.proxy.proxy_server import proxy_logging_obj

    record_postponed_cleanup(kind)
    await _alert(proxy_logging_obj, f"SpendLogs {kind} cleanup postponed for pending or parked accounting deliveries.")
