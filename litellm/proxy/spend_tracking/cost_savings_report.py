"""
Scheduled weekly/monthly cost-savings report emails.

Reuses the daily savings rollups (autorouter/compression/prompt-caching, see
``litellm/proxy/spend_tracking/savings.py``) already aggregated for the Cost
Optimization dashboard, and reports them to a configured recipient list on a
cadence, instead of only surfacing them in the UI.
"""

import datetime
from calendar import monthrange
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.proxy.utils import PrismaClient


@dataclass(frozen=True, slots=True)
class CostSavingsReport:
    start_date: datetime.date
    end_date: datetime.date
    total_spend: float
    autorouter_savings_spend: float
    compression_savings_spend: float
    prompt_caching_savings_spend: float

    @property
    def total_savings(self) -> float:
        return self.autorouter_savings_spend + self.compression_savings_spend + self.prompt_caching_savings_spend


async def get_cost_savings_report_for_time_range(
    prisma_client: PrismaClient,
    start_date: datetime.date,
    end_date: datetime.date,
) -> CostSavingsReport | None:
    """
    Returns the proxy-wide cost-savings report for ``[start_date, end_date]``, or
    ``None`` when there is nothing to report (no spend, no savings).
    """
    from litellm.proxy.management_endpoints.common_daily_activity import (
        get_daily_activity_aggregated,
    )

    resp: Final = await get_daily_activity_aggregated(
        prisma_client=prisma_client,
        table_name="litellm_dailyuserspend",
        entity_id_field="user_id",
        entity_id=None,
        entity_metadata_field=None,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        model=None,
        api_key=None,
    )

    total_spend: Final = resp.metadata.total_spend or 0.0
    autorouter_savings_spend: Final = resp.metadata.total_autorouter_savings_spend or 0.0
    compression_savings_spend: Final = resp.metadata.total_compression_savings_spend or 0.0
    prompt_caching_savings_spend: Final = resp.metadata.total_prompt_caching_savings_spend or 0.0

    nothing_to_report: Final = (
        total_spend == 0.0
        and autorouter_savings_spend == 0.0
        and compression_savings_spend == 0.0
        and prompt_caching_savings_spend == 0.0
    )
    if nothing_to_report:
        return None

    return CostSavingsReport(
        start_date=start_date,
        end_date=end_date,
        total_spend=total_spend,
        autorouter_savings_spend=autorouter_savings_spend,
        compression_savings_spend=compression_savings_spend,
        prompt_caching_savings_spend=prompt_caching_savings_spend,
    )


async def send_weekly_cost_savings_report(
    prisma_client: PrismaClient,
    internal_usage_cache: DualCache,
    recipient_emails: Sequence[str],
    time_range: str = "7d",
) -> None:
    """
    Args:
        time_range: A string specifying the time range for the report, e.g., "1d", "7d", "30d"
    """
    from litellm.integrations.email_alerting import send_cost_savings_report_email

    try:
        days: Final = int(time_range[:-1])
        if time_range[-1].lower() != "d":
            raise ValueError("Time range must be specified in days, e.g., '7d'")

        end_date: Final = datetime.datetime.now().date()  # noqa: DTZ005  # naive local time, matches spend report job
        start_date: Final = end_date - datetime.timedelta(days=days)

        event_cache_key: Final = (
            f"weekly_cost_savings_report_sent_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
        )
        if await internal_usage_cache.async_get_cache(key=event_cache_key):
            return

        report: Final = await get_cost_savings_report_for_time_range(
            prisma_client=prisma_client,
            start_date=start_date,
            end_date=end_date,
        )
        if report is None:
            return

        await send_cost_savings_report_email(recipient_emails=list(recipient_emails), report=report)

        await internal_usage_cache.async_set_cache(
            key=event_cache_key,
            value="SENT",
            ttl=duration_in_seconds(time_range),
        )
    except ValueError as ve:
        verbose_proxy_logger.error("Invalid time range format: %s", ve)
    except Exception as e:  # noqa: BLE001  # scheduled job must never crash the scheduler loop
        verbose_proxy_logger.error("Error sending weekly cost savings report: %s", e)


async def send_monthly_cost_savings_report(
    prisma_client: PrismaClient,
    internal_usage_cache: DualCache,
    recipient_emails: Sequence[str],
) -> None:
    from litellm.integrations.email_alerting import send_cost_savings_report_email

    try:
        end_date: Final = datetime.datetime.now().date()  # noqa: DTZ005  # naive local time, matches spend report job
        start_date: Final = end_date.replace(day=1)

        event_cache_key: Final = (
            f"monthly_cost_savings_report_sent_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
        )
        if await internal_usage_cache.async_get_cache(key=event_cache_key):
            return

        report: Final = await get_cost_savings_report_for_time_range(
            prisma_client=prisma_client,
            start_date=start_date,
            end_date=end_date,
        )
        if report is None:
            return

        await send_cost_savings_report_email(recipient_emails=list(recipient_emails), report=report)

        _, days_in_month = monthrange(end_date.year, end_date.month)
        await internal_usage_cache.async_set_cache(
            key=event_cache_key,
            value="SENT",
            ttl=duration_in_seconds(f"{days_in_month}d"),
        )
    except Exception as e:  # noqa: BLE001  # scheduled job must never crash the scheduler loop
        verbose_proxy_logger.error("Error sending monthly cost savings report: %s", e)
