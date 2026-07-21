"""
Daily rollup for admin-registered PTU reservations.

Writes prorated flat cost to LiteLLM_DailyTeamSpend using a sentinel api_key
so the rows are distinguishable from real per-request rows and share the
existing unique constraint.
"""

from calendar import monthrange
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Optional, Protocol

from litellm._logging import verbose_proxy_logger
from litellm.constants import PTU_ROLLUP_JOB_ID, PTU_SENTINEL_API_KEY
from litellm.repositories.ptu_reservation_repository import PTUReservationRepository


class AzureCostFetcher(Protocol):
    """Dependency injected into the rollup for azure_billing reservations.

    Implemented by ``AzureCostManagementClient`` in production, mocked in tests.
    """

    async def get_daily_cost(self, resource_id: str, day: date) -> float: ...


@dataclass(frozen=True, slots=True)
class RollupResult:
    day: date
    reservations_processed: int
    rows_written: int
    skipped_flag_off: bool = False


def _days_in_month(day: date) -> int:
    return monthrange(day.year, day.month)[1]


async def _compute_daily_flat_cost(
    reservation: Any,
    day: date,
    *,
    azure_fetcher: Optional[AzureCostFetcher] = None,
) -> float:
    """Return the flat cost attributable to ``day`` for a single reservation.

    manual: prorated (ptu_count * cost_per_ptu) / days_in_month.
    azure_billing: live fetch via ``azure_fetcher`` when provided; 0.0 when
    the pull is not configured (rollup logs a skip).
    """
    if reservation.cost_source == "manual":
        if reservation.ptu_count is None or reservation.cost_per_ptu is None:
            return 0.0
        monthly_total = float(reservation.ptu_count) * float(reservation.cost_per_ptu)
        return monthly_total / float(_days_in_month(day))
    if reservation.cost_source == "azure_billing":
        if azure_fetcher is None or reservation.azure_resource_id is None:
            verbose_proxy_logger.warning(
                "PTU rollup: reservation=%s cost_source=azure_billing skipped "
                "(azure_ptu_billing.subscription_id / Entra ID env vars not configured, "
                "or azure_resource_id missing on the reservation)",
                getattr(reservation, "id", "?"),
            )
            return 0.0
        try:
            fetched = await azure_fetcher.get_daily_cost(reservation.azure_resource_id, day)
        except Exception as exc:  # noqa: BLE001  # log and continue; one bad reservation must not stop the batch
            verbose_proxy_logger.error(
                "PTU rollup: azure fetch failed for reservation=%s day=%s: %s",
                getattr(reservation, "id", "?"),
                day.isoformat(),
                exc,
            )
            return 0.0
        currency = getattr(azure_fetcher, "last_currency", None)
        if currency is not None and currency.upper() != "USD":
            verbose_proxy_logger.warning(
                "PTU rollup: azure_billing reservation=%s day=%s returned currency=%s; "
                "skipping write to avoid storing non-USD amount as USD (LiteLLM_DailyTeamSpend "
                "assumes USD). File a follow-up for currency conversion.",
                getattr(reservation, "id", "?"),
                day.isoformat(),
                currency,
            )
            return 0.0
        verbose_proxy_logger.info(
            "PTU rollup: azure_billing reservation=%s day=%s resource=%s returned $%.4f",
            getattr(reservation, "id", "?"),
            day.isoformat(),
            reservation.azure_resource_id,
            fetched,
        )
        return fetched
    verbose_proxy_logger.warning(
        "PTU rollup: unknown cost_source=%s on reservation=%s; skipping",
        reservation.cost_source,
        getattr(reservation, "id", "?"),
    )
    return 0.0


async def _upsert_ptu_daily_row(
    prisma_client: Any,
    *,
    team_id: str,
    model: str,
    date_str: str,
    reservation_id: str,
    flat_cost: float,
) -> None:
    """Idempotent upsert of a sentinel-api_key row on LiteLLM_DailyTeamSpend."""
    where = {
        "team_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name_endpoint": {
            "team_id": team_id,
            "date": date_str,
            "api_key": PTU_SENTINEL_API_KEY,
            "model": model,
            "custom_llm_provider": "",
            "mcp_namespaced_tool_name": "",
            "endpoint": "",
        }
    }
    now = datetime.now(timezone.utc)
    await prisma_client.db.litellm_dailyteamspend.upsert(
        where=where,
        data={
            "create": {
                "team_id": team_id,
                "date": date_str,
                "api_key": PTU_SENTINEL_API_KEY,
                "model": model,
                "custom_llm_provider": "",
                "mcp_namespaced_tool_name": "",
                "endpoint": "",
                "ptu_flat_cost": flat_cost,
                "ptu_reservation_id": reservation_id,
            },
            "update": {
                "ptu_flat_cost": flat_cost,
                "ptu_reservation_id": reservation_id,
                "updated_at": now,
            },
        },
    )


async def run_ptu_reservation_rollup(
    prisma_client: Any,
    target_date: date | None = None,
    *,
    force: bool = False,
    azure_fetcher: Optional[AzureCostFetcher] = None,
) -> RollupResult:
    """Rollup one UTC day of flat PTU cost across all active reservations.

    Defaults to yesterday UTC. ``force=True`` bypasses the feature-flag check
    so the CLI backfill can run when the scheduler is off. ``azure_fetcher``
    is injected by the proxy startup wiring when
    ``general_settings.azure_ptu_billing.subscription_id`` and the Entra ID env
    vars are configured; azure_billing reservations no-op with a warning when
    it is None.
    Idempotent under the LiteLLM_DailyTeamSpend unique constraint on every
    invocation path.
    """
    if not force:
        from litellm.proxy.proxy_server import general_settings

        if not general_settings.get("enable_ptu_cost_attribution", False):
            verbose_proxy_logger.debug("PTU rollup: feature flag off, skipping")
            return RollupResult(
                day=target_date or (datetime.now(timezone.utc).date() - timedelta(days=1)),
                reservations_processed=0,
                rows_written=0,
                skipped_flag_off=True,
            )

    if prisma_client is None:
        verbose_proxy_logger.warning("PTU rollup: prisma_client is None, skipping")
        return RollupResult(
            day=target_date or (datetime.now(timezone.utc).date() - timedelta(days=1)),
            reservations_processed=0,
            rows_written=0,
        )

    day = target_date or (datetime.now(timezone.utc).date() - timedelta(days=1))
    day_start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    date_str = day.isoformat()

    repo = PTUReservationRepository(prisma_client)
    reservations = await repo.find_active(as_of=day_start)

    rows_written = 0
    for reservation in reservations:
        flat_cost = await _compute_daily_flat_cost(reservation, day, azure_fetcher=azure_fetcher)
        if flat_cost <= 0:
            continue
        try:
            await _upsert_ptu_daily_row(
                prisma_client,
                team_id=reservation.team_id,
                model=reservation.model,
                date_str=date_str,
                reservation_id=reservation.id,
                flat_cost=flat_cost,
            )
            rows_written += 1
        except Exception as exc:  # noqa: BLE001  # one bad reservation must not stop the batch; logged and continued
            verbose_proxy_logger.error(
                "PTU rollup: upsert failed for reservation=%s day=%s: %s",
                reservation.id,
                date_str,
                exc,
            )

    verbose_proxy_logger.info(
        "PTU rollup for %s: %d reservations processed, %d rows written",
        date_str,
        len(reservations),
        rows_written,
    )
    return RollupResult(
        day=day,
        reservations_processed=len(reservations),
        rows_written=rows_written,
    )


__all__ = [
    "AzureCostFetcher",
    "PTU_ROLLUP_JOB_ID",
    "PTU_SENTINEL_API_KEY",
    "RollupResult",
    "run_ptu_reservation_rollup",
]
