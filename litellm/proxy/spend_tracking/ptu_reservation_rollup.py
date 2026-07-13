"""
Daily rollup for admin-registered PTU reservations.

Writes prorated flat cost to LiteLLM_DailyTeamSpend using a sentinel api_key
so the rows are distinguishable from real per-request rows and share the
existing unique constraint.
"""

from calendar import monthrange
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from litellm._logging import verbose_proxy_logger
from litellm.repositories.ptu_reservation_repository import PTUReservationRepository

PTU_SENTINEL_API_KEY = "__ptu_reservation__"
PTU_ROLLUP_JOB_ID = "ptu_reservation_rollup_job"


@dataclass(frozen=True, slots=True)
class RollupResult:
    day: date
    reservations_processed: int
    rows_written: int
    skipped_flag_off: bool = False


def _days_in_month(day: date) -> int:
    return monthrange(day.year, day.month)[1]


def _compute_daily_flat_cost(reservation: Any, day: date) -> float:
    """Return the flat cost attributable to ``day`` for a single reservation."""
    if reservation.cost_source != "manual":
        return 0.0
    if reservation.ptu_count is None or reservation.cost_per_ptu is None:
        return 0.0
    monthly_total = float(reservation.ptu_count) * float(reservation.cost_per_ptu)
    return monthly_total / float(_days_in_month(day))


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
) -> RollupResult:
    """Rollup one UTC day of flat PTU cost across all active reservations.

    Defaults to yesterday UTC. Callable from the scheduler and from the CLI
    backfill helper; both paths are idempotent under the ``LiteLLM_DailyTeamSpend``
    unique constraint.
    """
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
        flat_cost = _compute_daily_flat_cost(reservation, day)
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
        except Exception as exc:
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
    "PTU_ROLLUP_JOB_ID",
    "PTU_SENTINEL_API_KEY",
    "RollupResult",
    "run_ptu_reservation_rollup",
]
