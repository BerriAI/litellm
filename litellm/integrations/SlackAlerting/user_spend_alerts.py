"""Per-user daily/monthly spend threshold alerts and spend anomaly detection."""

import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from pydantic import TypeAdapter

from litellm.constants import HOURS_IN_A_DAY
from litellm.types.integrations.slack_alerting import AlertType, SlackAlertingArgs

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

DAY_SECONDS: Final = HOURS_IN_A_DAY * 60 * 60
MONTHLY_ALERT_TTL_SECONDS: Final = 32 * DAY_SECONDS

USER_SPEND_QUERY: Final = """
SELECT
    user_id,
    COALESCE(SUM(spend) FILTER (WHERE date = $1), 0)::float AS daily_spend,
    COALESCE(SUM(spend) FILTER (WHERE date >= $2), 0)::float AS monthly_spend,
    COALESCE(SUM(spend) FILTER (WHERE date >= $3 AND date < $1), 0)::float AS baseline_spend,
    COUNT(DISTINCT date) FILTER (WHERE date >= $3 AND date < $1 AND spend > 0)::int AS baseline_days
FROM "LiteLLM_DailyUserSpend"
WHERE date >= LEAST($2, $3) AND user_id IS NOT NULL
GROUP BY user_id
HAVING COALESCE(SUM(spend) FILTER (WHERE date >= $2), 0) > 0
"""


@dataclass(frozen=True, slots=True)
class UserSpendRow:
    user_id: str
    daily_spend: float
    monthly_spend: float
    baseline_spend: float
    baseline_days: int


@dataclass(frozen=True, slots=True)
class UserSpendAlertEvent:
    kind: Literal["daily_threshold", "monthly_threshold", "anomaly"]
    alert_type: AlertType
    message: str
    cache_key: str
    cache_ttl: int


USER_SPEND_ROWS_ADAPTER: Final = TypeAdapter(tuple[UserSpendRow, ...])


async def fetch_user_spend_rows(
    prisma_client: "PrismaClient",
    today: datetime.date,
    baseline_days: int,
) -> tuple[UserSpendRow, ...]:
    today_str: Final = today.strftime("%Y-%m-%d")
    month_start_str: Final = today.replace(day=1).strftime("%Y-%m-%d")
    baseline_start_str: Final = (today - datetime.timedelta(days=max(baseline_days, 1))).strftime("%Y-%m-%d")
    raw: Final = await prisma_client.db.query_raw(USER_SPEND_QUERY, today_str, month_start_str, baseline_start_str)
    return USER_SPEND_ROWS_ADAPTER.validate_python(raw)


def _daily_threshold_event(row: UserSpendRow, args: SlackAlertingArgs, today_str: str) -> UserSpendAlertEvent | None:
    threshold: Final = args.daily_spend_per_user_threshold
    if threshold is None or row.daily_spend < threshold:
        return None
    return UserSpendAlertEvent(
        kind="daily_threshold",
        alert_type=AlertType.user_spend_thresholds,
        message=(
            f"User Daily Spend Threshold Crossed:\n"
            f"User: `{row.user_id}`\n"
            f"Spend Today: `${row.daily_spend:.2f}`\n"
            f"Daily Threshold: `${threshold:.2f}`"
        ),
        cache_key=f"user_spend_alert_daily_{row.user_id}_{today_str}",
        cache_ttl=DAY_SECONDS,
    )


def _monthly_threshold_event(row: UserSpendRow, args: SlackAlertingArgs, month_str: str) -> UserSpendAlertEvent | None:
    threshold: Final = args.monthly_spend_per_user_threshold
    if threshold is None or row.monthly_spend < threshold:
        return None
    return UserSpendAlertEvent(
        kind="monthly_threshold",
        alert_type=AlertType.user_spend_thresholds,
        message=(
            f"User Monthly Spend Threshold Crossed:\n"
            f"User: `{row.user_id}`\n"
            f"Spend This Month: `${row.monthly_spend:.2f}`\n"
            f"Monthly Threshold: `${threshold:.2f}`"
        ),
        cache_key=f"user_spend_alert_monthly_{row.user_id}_{month_str}",
        cache_ttl=MONTHLY_ALERT_TTL_SECONDS,
    )


def _anomaly_event(row: UserSpendRow, args: SlackAlertingArgs, today_str: str) -> UserSpendAlertEvent | None:
    if row.daily_spend < args.spend_anomaly_min_spend:
        return None
    baseline_daily_avg: Final = row.baseline_spend / row.baseline_days if row.baseline_days > 0 else 0.0
    if row.baseline_days > 0 and row.daily_spend <= args.spend_anomaly_multiplier * baseline_daily_avg:
        return None
    return UserSpendAlertEvent(
        kind="anomaly",
        alert_type=AlertType.user_spend_anomalies,
        message=(
            f"User Spend Anomaly Detected:\n"
            f"User: `{row.user_id}`\n"
            f"Spend Today: `${row.daily_spend:.2f}`\n"
            f"Daily Average (last {args.spend_anomaly_baseline_days} days): `${baseline_daily_avg:.2f}`\n"
            f"Trigger: spend above `{args.spend_anomaly_multiplier}x` the daily average "
            f"(minimum `${args.spend_anomaly_min_spend:.2f}`)"
        ),
        cache_key=f"user_spend_alert_anomaly_{row.user_id}_{today_str}",
        cache_ttl=DAY_SECONDS,
    )


def evaluate_user_spend(
    row: UserSpendRow,
    args: SlackAlertingArgs,
    today: datetime.date,
    thresholds_enabled: bool,
    anomalies_enabled: bool,
) -> tuple[UserSpendAlertEvent, ...]:
    today_str: Final = today.strftime("%Y-%m-%d")
    month_str: Final = today.strftime("%Y-%m")
    threshold_events: Final = (
        (
            _daily_threshold_event(row=row, args=args, today_str=today_str),
            _monthly_threshold_event(row=row, args=args, month_str=month_str),
        )
        if thresholds_enabled
        else ()
    )
    anomaly_events: Final = (_anomaly_event(row=row, args=args, today_str=today_str),) if anomalies_enabled else ()
    return tuple(event for event in (*threshold_events, *anomaly_events) if event is not None)
