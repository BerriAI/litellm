import datetime
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.integrations.SlackAlerting.user_spend_alerts import (
    UserSpendRow,
    evaluate_user_spend,
)
from litellm.types.integrations.slack_alerting import (
    DEFAULT_ALERT_TYPES,
    AlertType,
    SlackAlertingArgs,
)

TODAY: Final = datetime.date(2026, 8, 15)


def _row(
    daily_spend: float = 0.0,
    monthly_spend: float = 0.0,
    baseline_spend: float = 0.0,
) -> UserSpendRow:
    return UserSpendRow(
        user_id="user-1",
        daily_spend=daily_spend,
        monthly_spend=monthly_spend,
        baseline_spend=baseline_spend,
    )


def _evaluate(row: UserSpendRow, args: SlackAlertingArgs, thresholds: bool = True, anomalies: bool = True):
    return evaluate_user_spend(
        row=row,
        args=args,
        today=TODAY,
        thresholds_enabled=thresholds,
        anomalies_enabled=anomalies,
    )


def test_daily_threshold_crossed():
    args: Final = SlackAlertingArgs(daily_spend_per_user_threshold=50.0, spend_anomaly_min_spend=1000.0)
    events: Final = _evaluate(_row(daily_spend=75.0, monthly_spend=75.0), args)
    assert [e.kind for e in events] == ["daily_threshold"]
    assert "`$75.00`" in events[0].message
    assert "`$50.00`" in events[0].message
    assert events[0].alert_type == AlertType.user_spend_thresholds
    assert events[0].cache_key == "user_spend_alert_daily_user-1_2026-08-15"


def test_daily_threshold_not_crossed():
    args: Final = SlackAlertingArgs(daily_spend_per_user_threshold=50.0, spend_anomaly_min_spend=1000.0)
    assert _evaluate(_row(daily_spend=49.99, monthly_spend=49.99), args) == ()


def test_thresholds_unset_by_default():
    args: Final = SlackAlertingArgs(spend_anomaly_min_spend=1000.0)
    assert _evaluate(_row(daily_spend=999.0, monthly_spend=999.0), args) == ()


def test_monthly_threshold_crossed():
    args: Final = SlackAlertingArgs(monthly_spend_per_user_threshold=200.0, spend_anomaly_min_spend=1000.0)
    events: Final = _evaluate(_row(daily_spend=5.0, monthly_spend=250.0), args)
    assert [e.kind for e in events] == ["monthly_threshold"]
    assert events[0].cache_key == "user_spend_alert_monthly_user-1_2026-08"


def test_thresholds_disabled_suppresses_threshold_events():
    args: Final = SlackAlertingArgs(
        daily_spend_per_user_threshold=50.0,
        monthly_spend_per_user_threshold=200.0,
        spend_anomaly_min_spend=1000.0,
    )
    assert _evaluate(_row(daily_spend=75.0, monthly_spend=250.0), args, thresholds=False) == ()


def test_anomaly_detected_above_multiple_of_baseline():
    args: Final = SlackAlertingArgs(spend_anomaly_multiplier=3.0, spend_anomaly_min_spend=10.0)
    events: Final = _evaluate(
        _row(daily_spend=70.0, monthly_spend=100.0, baseline_spend=70.0), args
    )
    assert [e.kind for e in events] == ["anomaly"]
    assert events[0].alert_type == AlertType.user_spend_anomalies
    assert "`$10.00`" in events[0].message
    assert events[0].cache_key == "user_spend_alert_anomaly_user-1_2026-08-15"


def test_no_anomaly_within_baseline_multiple():
    args: Final = SlackAlertingArgs(spend_anomaly_multiplier=3.0, spend_anomaly_min_spend=10.0)
    assert (
        _evaluate(_row(daily_spend=25.0, monthly_spend=100.0, baseline_spend=70.0), args) == ()
    )


def test_no_anomaly_below_min_spend_floor():
    args: Final = SlackAlertingArgs(spend_anomaly_multiplier=3.0, spend_anomaly_min_spend=10.0)
    assert _evaluate(_row(daily_spend=9.0, monthly_spend=9.0, baseline_spend=0.1), args) == ()


def test_anomaly_for_new_user_without_baseline():
    args: Final = SlackAlertingArgs(spend_anomaly_multiplier=3.0, spend_anomaly_min_spend=10.0)
    events: Final = _evaluate(_row(daily_spend=15.0, monthly_spend=15.0), args)
    assert [e.kind for e in events] == ["anomaly"]


def test_sparse_baseline_averages_over_full_window():
    args: Final = SlackAlertingArgs(
        spend_anomaly_multiplier=3.0, spend_anomaly_min_spend=10.0, spend_anomaly_baseline_days=7
    )
    events: Final = _evaluate(_row(daily_spend=13.0, monthly_spend=20.0, baseline_spend=7.0), args)
    assert [e.kind for e in events] == ["anomaly"]


def test_anomalies_not_in_default_alert_types():
    assert AlertType.user_spend_anomalies not in DEFAULT_ALERT_TYPES
    assert AlertType.user_spend_thresholds in DEFAULT_ALERT_TYPES


def test_invalid_config_rejected():
    with pytest.raises(ValidationError, match="daily_spend_per_user_threshold"):
        SlackAlertingArgs(daily_spend_per_user_threshold=0)
    with pytest.raises(ValidationError, match="spend_anomaly_baseline_days"):
        SlackAlertingArgs(spend_anomaly_baseline_days=0)
    with pytest.raises(ValidationError, match="user_spend_check_interval"):
        SlackAlertingArgs(user_spend_check_interval=10)


def test_non_finite_config_rejected():
    with pytest.raises(ValidationError, match="daily_spend_per_user_threshold"):
        SlackAlertingArgs(daily_spend_per_user_threshold=float("inf"))
    with pytest.raises(ValidationError, match="spend_anomaly_multiplier"):
        SlackAlertingArgs(spend_anomaly_multiplier=float("nan"))
    with pytest.raises(ValidationError, match="spend_anomaly_min_spend"):
        SlackAlertingArgs(spend_anomaly_min_spend=float("inf"))
    with pytest.raises(ValidationError, match="user_spend_check_interval"):
        SlackAlertingArgs(user_spend_check_interval=float("inf"))


def test_anomalies_disabled_suppresses_anomaly_events():
    args: Final = SlackAlertingArgs(spend_anomaly_multiplier=3.0, spend_anomaly_min_spend=10.0)
    assert _evaluate(_row(daily_spend=500.0, monthly_spend=500.0), args, anomalies=False) == ()


@pytest.mark.asyncio
async def test_send_user_spend_alerts_sends_and_dedupes():
    slack_alerting: Final = SlackAlerting(
        alerting=["slack"],
        alerting_args={"daily_spend_per_user_threshold": 50.0, "spend_anomaly_min_spend": 1000.0},
    )
    mock_prisma: Final = AsyncMock()
    mock_prisma.db.query_raw = AsyncMock(
        return_value=[
            {
                "user_id": "user-1",
                "daily_spend": 75.0,
                "monthly_spend": 75.0,
                "baseline_spend": 0.0,
            },
            {
                "user_id": "user-2",
                "daily_spend": 60.0,
                "monthly_spend": 60.0,
                "baseline_spend": 0.0,
            },
        ]
    )
    with patch.object(slack_alerting, "send_alert", new_callable=AsyncMock) as mock_send_alert:
        await slack_alerting.send_user_spend_alerts(prisma_client=mock_prisma)
        assert mock_send_alert.call_count == 1
        sent_kwargs: Final = mock_send_alert.call_args.kwargs
        assert sent_kwargs["alert_type"] == AlertType.user_spend_thresholds
        assert "User Daily Spend Threshold Crossed" in sent_kwargs["message"]
        assert "`user-1`" in sent_kwargs["message"]
        assert "`user-2`" in sent_kwargs["message"]

        await slack_alerting.send_user_spend_alerts(prisma_client=mock_prisma)
        assert mock_send_alert.call_count == 1


@pytest.mark.asyncio
async def test_send_user_spend_alerts_noop_when_alert_types_disabled():
    slack_alerting: Final = SlackAlerting(
        alerting=["slack"],
        alert_types=[AlertType.budget_alerts],
        alerting_args={"daily_spend_per_user_threshold": 50.0},
    )
    mock_prisma: Final = AsyncMock()
    await slack_alerting.send_user_spend_alerts(prisma_client=mock_prisma)
    mock_prisma.db.query_raw.assert_not_called()
