"""
Tests for the SGR (successful gateway requests) soft/hard limit: where the
allowance is resolved from, how the window is bounded, and what state the
window's count lands in.
"""

import asyncio
from datetime import datetime, timezone

import pytest

from litellm.proxy.db.gateway_request_limits import (
    evaluate_sgr_limit,
    get_sgr_limit_status,
    resolve_sgr_limit,
    sgr_window_start,
)
from litellm.types.proxy.gateway_requests import (
    SGRLimitConfig,
    SGRLimitState,
    SGRLimitWindow,
)

# ── resolving the allowance ───────────────────────────────────────────────────


def test_no_allowance_without_a_license_or_config():
    assert resolve_sgr_limit(general_settings={}, license_data=None) is None


def test_license_max_sgr_is_the_allowance():
    config = resolve_sgr_limit(general_settings={}, license_data={"max_sgr": 1_000_000})
    assert config == SGRLimitConfig(limit=1_000_000, soft_limit=800_000, window=SGRLimitWindow.YEAR)


def test_config_overrides_the_license_so_a_customer_can_alert_earlier():
    config = resolve_sgr_limit(general_settings={"sgr_limit": 250_000}, license_data={"max_sgr": 1_000_000})
    assert config is not None
    assert config.limit == 250_000


def test_soft_limit_percent_moves_the_soft_threshold():
    config = resolve_sgr_limit(
        general_settings={"sgr_limit": 1000, "sgr_soft_limit_percent": 0.5},
        license_data=None,
    )
    assert config is not None
    assert config.soft_limit == 500


def test_window_can_be_narrowed_to_the_calendar_month():
    config = resolve_sgr_limit(general_settings={"sgr_limit": 10, "sgr_limit_window": "month"}, license_data=None)
    assert config is not None
    assert config.window is SGRLimitWindow.MONTH


def test_the_license_can_carry_the_window():
    config = resolve_sgr_limit(general_settings={}, license_data={"max_sgr": 100, "sgr_window": "month"})
    assert config is not None
    assert config.window is SGRLimitWindow.MONTH


def test_config_window_wins_over_the_license_window():
    config = resolve_sgr_limit(
        general_settings={"sgr_limit_window": "year"},
        license_data={"max_sgr": 100, "sgr_window": "month"},
    )
    assert config is not None
    assert config.window is SGRLimitWindow.YEAR


def test_a_nonsensical_license_window_falls_back_to_the_calendar_year():
    config = resolve_sgr_limit(general_settings={}, license_data={"max_sgr": 100, "sgr_window": "fortnight"})
    assert config is not None
    assert config.window is SGRLimitWindow.YEAR


def test_a_license_without_max_sgr_carries_no_allowance():
    assert resolve_sgr_limit(general_settings={}, license_data={"max_users": 5}) is None


@pytest.mark.parametrize("max_sgr", [0, -1, "1000", None])
def test_a_nonsense_license_value_is_not_an_allowance(max_sgr: object):
    assert resolve_sgr_limit(general_settings={}, license_data={"max_sgr": max_sgr}) is None


@pytest.mark.parametrize(
    "general_settings",
    [
        {"sgr_limit": 0},
        {"sgr_limit": -5},
        {"sgr_limit": "lots"},
        {"sgr_limit": 100, "sgr_soft_limit_percent": 1.5},
        {"sgr_limit": 100, "sgr_soft_limit_percent": 0},
        {"sgr_limit": 100, "sgr_limit_window": "fortnight"},
    ],
)
def test_invalid_settings_disable_the_limit_instead_of_crashing_the_proxy(general_settings: dict):
    assert resolve_sgr_limit(general_settings=general_settings, license_data=None) is None


def test_settings_left_unset_fall_back_to_the_license():
    config = resolve_sgr_limit(
        general_settings={"sgr_limit": None, "sgr_soft_limit_percent": None, "sgr_limit_window": None},
        license_data={"max_sgr": 100},
    )
    assert config == SGRLimitConfig(limit=100, soft_limit=80, window=SGRLimitWindow.YEAR)


def test_a_tiny_allowance_still_has_a_soft_threshold_of_at_least_one():
    config = resolve_sgr_limit(general_settings={"sgr_limit": 1}, license_data=None)
    assert config is not None
    assert config.soft_limit == 1


# ── the window ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "window, expected",
    [(SGRLimitWindow.MONTH, "2026-08-01"), (SGRLimitWindow.YEAR, "2026-01-01")],
)
def test_window_starts_on_the_calendar_boundary(window: SGRLimitWindow, expected: str):
    assert sgr_window_start(window, datetime(2026, 8, 8, 13, 45, tzinfo=timezone.utc)) == expected


# ── evaluating a count ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "successful_requests, expected",
    [
        (0, SGRLimitState.UNDER),
        (799, SGRLimitState.UNDER),
        (800, SGRLimitState.SOFT_EXCEEDED),
        (999, SGRLimitState.SOFT_EXCEEDED),
        (1000, SGRLimitState.HARD_EXCEEDED),
        (5000, SGRLimitState.HARD_EXCEEDED),
    ],
)
def test_thresholds_are_inclusive(successful_requests: int, expected: SGRLimitState):
    config = SGRLimitConfig(limit=1000, soft_limit=800, window=SGRLimitWindow.MONTH)
    assert evaluate_sgr_limit(config, successful_requests) is expected


# ── reading the window's count ────────────────────────────────────────────────


class FakeDB:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.queries: list[tuple[str, tuple[object, ...]]] = []

    async def query_raw(self, sql: str, *args: object) -> list[dict]:
        self.queries.append((sql, args))
        return self.rows


class FakePrismaClient:
    def __init__(self, rows: list[dict]) -> None:
        self.db = FakeDB(rows)


def _status(rows: list[dict], config: SGRLimitConfig):
    client = FakePrismaClient(rows)
    status = asyncio.run(
        get_sgr_limit_status(
            prisma_client=client,
            config=config,
            now=datetime(2026, 8, 8, tzinfo=timezone.utc),
        )
    )
    return status, client


def test_status_counts_only_the_current_window():
    config = SGRLimitConfig(limit=1000, soft_limit=800, window=SGRLimitWindow.MONTH)
    status, client = _status([{"successful_requests": 900}], config)

    assert client.db.queries[0][1] == ("2026-08-01",)
    assert status.successful_requests == 900
    assert status.window_start == "2026-08-01"
    assert status.state is SGRLimitState.SOFT_EXCEEDED


def test_status_of_a_deployment_that_has_served_nothing_yet():
    config = SGRLimitConfig(limit=1000, soft_limit=800, window=SGRLimitWindow.MONTH)
    status, _ = _status([], config)

    assert status.successful_requests == 0
    assert status.state is SGRLimitState.UNDER
