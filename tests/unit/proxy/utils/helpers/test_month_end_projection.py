from datetime import date, datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.proxy.utils import (
    _get_month_end_date,
    get_projected_spend_over_limit,
    is_projected_spend_over_limit,
)


def normalize(value):
    return value


def _freeze_today(monkeypatch, frozen):
    class _FrozenDate(date):
        @classmethod
        def today(cls):
            return frozen

    monkeypatch.setattr("litellm.proxy.utils.date", _FrozenDate)


@pytest.mark.parametrize(
    "today, expected",
    [
        (date(2024, 1, 15), date(2024, 1, 31)),
        (date(2024, 2, 1), date(2024, 2, 29)),
        (date(2023, 2, 1), date(2023, 2, 28)),
        (date(2024, 4, 10), date(2024, 4, 30)),
        (date(2024, 12, 1), date(2024, 12, 31)),
    ],
)
def test_get_month_end_date_happy_path(today, expected):
    result = _get_month_end_date(today)
    assert normalize(
        {
            "year": result.year,
            "month": result.month,
            "day": result.day,
            "expected": expected.isoformat(),
            "input": today.isoformat(),
        }
    ) == {
        "year": expected.year,
        "month": expected.month,
        "day": expected.day,
        "expected": expected.isoformat(),
        "input": today.isoformat(),
    }


def test_get_month_end_date_raises_on_non_date_input():
    with pytest.raises(AttributeError):
        _get_month_end_date("2024-01-15")


def test_is_projected_spend_over_limit_happy_path_under_budget(monkeypatch):
    _freeze_today(monkeypatch, date(2024, 1, 11))
    summary = {
        "result": is_projected_spend_over_limit(current_spend=10.0, soft_budget_limit=1_000_000.0),
        "current_spend": 10.0,
        "soft_budget_limit": 1_000_000.0,
    }
    assert summary == {
        "result": False,
        "current_spend": 10.0,
        "soft_budget_limit": 1_000_000.0,
    }


def test_is_projected_spend_over_limit_happy_path_over_budget(monkeypatch):
    _freeze_today(monkeypatch, date(2024, 1, 11))
    summary = {
        "result": is_projected_spend_over_limit(current_spend=100.0, soft_budget_limit=50.0),
        "current_spend": 100.0,
        "soft_budget_limit": 50.0,
    }
    assert summary == {
        "result": True,
        "current_spend": 100.0,
        "soft_budget_limit": 50.0,
    }


def test_is_projected_spend_over_limit_first_of_month_no_division_by_zero(monkeypatch):
    _freeze_today(monkeypatch, date(2024, 1, 1))
    summary = {
        "result": is_projected_spend_over_limit(current_spend=5.0, soft_budget_limit=10.0),
        "current_spend": 5.0,
        "soft_budget_limit": 10.0,
    }
    assert summary == {
        "result": True,
        "current_spend": 5.0,
        "soft_budget_limit": 10.0,
    }


def test_is_projected_spend_over_limit_none_limit_returns_false():
    assert is_projected_spend_over_limit(current_spend=10_000.0, soft_budget_limit=None) is False


def test_is_projected_spend_over_limit_raises_when_today_missing(monkeypatch):
    class _Broken:
        @classmethod
        def today(cls):
            raise RuntimeError("clock unavailable")

    monkeypatch.setattr("litellm.proxy.utils.date", _Broken)
    with pytest.raises(RuntimeError):
        is_projected_spend_over_limit(current_spend=1.0, soft_budget_limit=1.0)


def test_get_projected_spend_over_limit_happy_path_over_budget(monkeypatch):
    _freeze_today(monkeypatch, date(2024, 1, 11))
    result = get_projected_spend_over_limit(current_spend=100.0, soft_budget_limit=50.0)
    assert result is not None
    projected, exceed_date = result
    summary = {
        "projected_spend": projected,
        "exceed_date": exceed_date.isoformat(),
        "current_spend": 100.0,
        "soft_budget_limit": 50.0,
    }
    assert summary == {
        "projected_spend": 300.0,
        "exceed_date": "2024-01-11",
        "current_spend": 100.0,
        "soft_budget_limit": 50.0,
    }


def test_get_projected_spend_over_limit_first_of_month_uses_current_as_daily(
    monkeypatch,
):
    _freeze_today(monkeypatch, date(2024, 1, 1))
    result = get_projected_spend_over_limit(current_spend=5.0, soft_budget_limit=10.0)
    assert result is not None
    projected, exceed_date = result
    expected_exceed = date(2024, 1, 1) + timedelta(days=1.0)
    summary = {
        "projected_spend": projected,
        "exceed_date": exceed_date.isoformat(),
        "expected_exceed_date": expected_exceed.isoformat(),
        "soft_budget_limit": 10.0,
    }
    assert summary == {
        "projected_spend": 155.0,
        "exceed_date": expected_exceed.isoformat(),
        "expected_exceed_date": expected_exceed.isoformat(),
        "soft_budget_limit": 10.0,
    }


def test_get_projected_spend_over_limit_zero_daily_spend_exceed_today(monkeypatch):
    _freeze_today(monkeypatch, date(2024, 1, 11))
    result = get_projected_spend_over_limit(current_spend=0.0, soft_budget_limit=-1.0)
    assert result is not None
    projected, exceed_date = result
    summary = {
        "projected_spend": projected,
        "exceed_date": exceed_date.isoformat(),
        "soft_budget_limit": -1.0,
    }
    assert summary == {
        "projected_spend": 0.0,
        "exceed_date": "2024-01-11",
        "soft_budget_limit": -1.0,
    }


def test_get_projected_spend_over_limit_under_budget_returns_none(monkeypatch):
    _freeze_today(monkeypatch, date(2024, 1, 11))
    assert get_projected_spend_over_limit(current_spend=1.0, soft_budget_limit=1_000_000.0) is None


def test_get_projected_spend_over_limit_exceed_date_uses_remaining_budget(monkeypatch):
    _freeze_today(monkeypatch, date(2024, 1, 11))
    result = get_projected_spend_over_limit(current_spend=20.0, soft_budget_limit=30.0)
    assert result is not None
    projected, exceed_date = result
    daily = 20.0 / 10
    remaining_budget = 30.0 - 20.0
    expected_exceed = date(2024, 1, 11) + timedelta(days=remaining_budget / daily)
    summary = {
        "projected_spend": projected,
        "exceed_date": exceed_date.isoformat(),
        "expected_exceed_date": expected_exceed.isoformat(),
        "soft_budget_limit": 30.0,
    }
    assert summary == {
        "projected_spend": 60.0,
        "exceed_date": expected_exceed.isoformat(),
        "expected_exceed_date": expected_exceed.isoformat(),
        "soft_budget_limit": 30.0,
    }


def test_get_projected_spend_over_limit_none_limit_returns_none():
    assert get_projected_spend_over_limit(current_spend=1.0, soft_budget_limit=None) is None


def test_get_projected_spend_over_limit_raises_when_today_missing(monkeypatch):
    class _Broken:
        @classmethod
        def today(cls):
            raise RuntimeError("clock unavailable")

    monkeypatch.setattr("litellm.proxy.utils.date", _Broken)
    with pytest.raises(RuntimeError):
        get_projected_spend_over_limit(current_spend=1.0, soft_budget_limit=1.0)


UTC_RESET_JAN_16: Final = datetime(2024, 1, 16, tzinfo=timezone.utc)


def test_daily_budget_projects_pace_to_the_reset_not_month_end():
    result: Final = get_projected_spend_over_limit(
        current_spend=8.0,
        soft_budget_limit=10.0,
        budget_duration="1d",
        budget_reset_at=UTC_RESET_JAN_16,
        now=datetime(2024, 1, 15, 12, tzinfo=timezone.utc),
    )
    assert result == (16.0, date(2024, 1, 15))


def test_daily_budget_late_in_the_day_does_not_double_todays_spend():
    assert (
        get_projected_spend_over_limit(
            current_spend=6.0,
            soft_budget_limit=10.0,
            budget_duration="1d",
            budget_reset_at=UTC_RESET_JAN_16,
            now=datetime(2024, 1, 15, 23, tzinfo=timezone.utc),
        )
        is None
    )


def test_naive_reset_time_is_read_as_utc():
    result: Final = get_projected_spend_over_limit(
        current_spend=8.0,
        soft_budget_limit=10.0,
        budget_duration="1d",
        budget_reset_at=datetime(2024, 1, 16),
        now=datetime(2024, 1, 15, 12),
    )
    assert result == (16.0, date(2024, 1, 15))


def test_weekly_budget_measures_pace_from_the_previous_reset():
    result: Final = get_projected_spend_over_limit(
        current_spend=8.0,
        soft_budget_limit=10.0,
        budget_duration="7d",
        budget_reset_at=datetime(2024, 1, 18, tzinfo=timezone.utc),
        now=datetime(2024, 1, 15, tzinfo=timezone.utc),
    )
    assert result == (14.0, date(2024, 1, 16))


def test_monthly_budget_window_starts_on_the_previous_reset_day():
    result: Final = get_projected_spend_over_limit(
        current_spend=8.0,
        soft_budget_limit=10.0,
        budget_duration="1mo",
        budget_reset_at=datetime(2024, 3, 1, tzinfo=timezone.utc),
        now=datetime(2024, 2, 5, tzinfo=timezone.utc),
    )
    assert result == (pytest.approx(58.0), date(2024, 2, 6))


def test_sub_day_budget_is_measured_in_hours_not_days():
    result: Final = get_projected_spend_over_limit(
        current_spend=3.0,
        soft_budget_limit=5.0,
        budget_duration="4h",
        budget_reset_at=datetime(2024, 1, 15, 12, tzinfo=timezone.utc),
        now=datetime(2024, 1, 15, 10, tzinfo=timezone.utc),
    )
    assert result == (pytest.approx(6.0), date(2024, 1, 15))


def test_first_minutes_after_a_reset_do_not_project_a_burst_across_the_window():
    assert (
        get_projected_spend_over_limit(
            current_spend=0.10,
            soft_budget_limit=10.0,
            budget_duration="1d",
            budget_reset_at=UTC_RESET_JAN_16,
            now=datetime(2024, 1, 15, 0, 1, tzinfo=timezone.utc),
        )
        is None
    )


def test_exceed_date_is_reported_in_the_reset_timezone():
    result: Final = get_projected_spend_over_limit(
        current_spend=9.5,
        soft_budget_limit=10.0,
        budget_duration="1d",
        budget_reset_at=datetime(2024, 1, 17, tzinfo=timezone(timedelta(hours=2))),
        now=datetime(2024, 1, 15, 22, 10, tzinfo=timezone.utc),
    )
    assert result is not None
    assert result[1] == date(2024, 1, 16)


@pytest.mark.parametrize("soft_budget_limit", [9.99, 13.9, 13.99])
def test_exceed_date_never_lands_after_the_reset(soft_budget_limit):
    result: Final = get_projected_spend_over_limit(
        current_spend=8.0,
        soft_budget_limit=soft_budget_limit,
        budget_duration="7d",
        budget_reset_at=datetime(2024, 1, 18, tzinfo=timezone.utc),
        now=datetime(2024, 1, 15, tzinfo=timezone.utc),
    )
    assert result is not None
    assert result[1] <= date(2024, 1, 18)


@pytest.mark.parametrize("current_spend, expected", [(4.0, False), (8.0, True)])
def test_is_projected_spend_over_limit_follows_the_reset_window(current_spend, expected):
    assert (
        is_projected_spend_over_limit(
            current_spend=current_spend,
            soft_budget_limit=10.0,
            budget_duration="1d",
            budget_reset_at=UTC_RESET_JAN_16,
            now=datetime(2024, 1, 15, 12, tzinfo=timezone.utc),
        )
        is expected
    )


def test_without_duration_keeps_month_end_behavior(monkeypatch):
    _freeze_today(monkeypatch, date(2024, 1, 15))
    result: Final = get_projected_spend_over_limit(current_spend=8.0, soft_budget_limit=10.0)
    assert result is not None
    assert result[0] == pytest.approx(8.0 + (8.0 / 14) * 16)


def test_thirty_day_budget_measures_pace_from_the_first_of_the_month():
    result: Final = get_projected_spend_over_limit(
        current_spend=1.0,
        soft_budget_limit=10.0,
        budget_duration="30d",
        budget_reset_at=datetime(2024, 11, 1, tzinfo=timezone.utc),
        now=datetime(2024, 10, 4, tzinfo=timezone.utc),
    )
    assert result == (pytest.approx(1.0 + 672 / 72), date(2024, 10, 31))


def test_hour_spelling_the_scheduler_resets_at_midnight_does_not_alert_before_midnight():
    assert (
        get_projected_spend_over_limit(
            current_spend=1.0,
            soft_budget_limit=10.0,
            budget_duration="1hr",
            budget_reset_at=UTC_RESET_JAN_16,
            now=datetime(2024, 1, 15, 22, tzinfo=timezone.utc),
        )
        is None
    )


def test_unrecognized_duration_projects_within_the_daily_window_it_resets_on():
    result: Final = get_projected_spend_over_limit(
        current_spend=8.0,
        soft_budget_limit=10.0,
        budget_duration="fortnightly",
        budget_reset_at=UTC_RESET_JAN_16,
        now=datetime(2024, 1, 15, 12, tzinfo=timezone.utc),
    )
    assert result == (16.0, date(2024, 1, 15))


def test_reset_more_than_one_window_ahead_does_not_project():
    assert (
        get_projected_spend_over_limit(
            current_spend=9.0,
            soft_budget_limit=10.0,
            budget_duration="1d",
            budget_reset_at=UTC_RESET_JAN_16,
            now=datetime(2024, 1, 14, 12, tzinfo=timezone.utc),
        )
        is None
    )
