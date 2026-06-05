from datetime import date, timedelta

import pytest

from litellm.proxy.utils import (
    _get_month_end_date,
    _get_projected_spend_over_limit,
    _is_projected_spend_over_limit,
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
        "result": _is_projected_spend_over_limit(
            current_spend=10.0, soft_budget_limit=1_000_000.0
        ),
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
        "result": _is_projected_spend_over_limit(
            current_spend=100.0, soft_budget_limit=50.0
        ),
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
        "result": _is_projected_spend_over_limit(
            current_spend=5.0, soft_budget_limit=10.0
        ),
        "current_spend": 5.0,
        "soft_budget_limit": 10.0,
    }
    assert summary == {
        "result": True,
        "current_spend": 5.0,
        "soft_budget_limit": 10.0,
    }


def test_is_projected_spend_over_limit_none_limit_returns_false():
    assert (
        _is_projected_spend_over_limit(current_spend=10_000.0, soft_budget_limit=None)
        is False
    )


def test_is_projected_spend_over_limit_raises_when_today_missing(monkeypatch):
    class _Broken:
        @classmethod
        def today(cls):
            raise RuntimeError("clock unavailable")

    monkeypatch.setattr("litellm.proxy.utils.date", _Broken)
    with pytest.raises(RuntimeError):
        _is_projected_spend_over_limit(current_spend=1.0, soft_budget_limit=1.0)


def test_get_projected_spend_over_limit_happy_path_over_budget(monkeypatch):
    _freeze_today(monkeypatch, date(2024, 1, 11))
    result = _get_projected_spend_over_limit(
        current_spend=100.0, soft_budget_limit=50.0
    )
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
    result = _get_projected_spend_over_limit(current_spend=5.0, soft_budget_limit=10.0)
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
    result = _get_projected_spend_over_limit(current_spend=0.0, soft_budget_limit=-1.0)
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
    assert (
        _get_projected_spend_over_limit(
            current_spend=1.0, soft_budget_limit=1_000_000.0
        )
        is None
    )


def test_get_projected_spend_over_limit_exceed_date_uses_remaining_budget(monkeypatch):
    _freeze_today(monkeypatch, date(2024, 1, 11))
    result = _get_projected_spend_over_limit(current_spend=20.0, soft_budget_limit=30.0)
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
    assert (
        _get_projected_spend_over_limit(current_spend=1.0, soft_budget_limit=None)
        is None
    )


def test_get_projected_spend_over_limit_raises_when_today_missing(monkeypatch):
    class _Broken:
        @classmethod
        def today(cls):
            raise RuntimeError("clock unavailable")

    monkeypatch.setattr("litellm.proxy.utils.date", _Broken)
    with pytest.raises(RuntimeError):
        _get_projected_spend_over_limit(current_spend=1.0, soft_budget_limit=1.0)
