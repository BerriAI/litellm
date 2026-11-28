"""Contract: when a `budget_duration` resets, as stored in `budget_reset_at`, and what a reset clears.

Reset times are not "now plus the duration". `litellm/litellm_core_utils/duration_parser.py::
get_next_standardized_reset_time` aligns them to calendar boundaries in `litellm_settings.timezone` (default UTC)
at `litellm_settings.budget_reset_time` (default midnight, day and longer durations only):

    1d            next reset_time_of_day
    7d            next Monday at reset_time_of_day
    30d, 1mo      the 1st of the next month at reset_time_of_day
    12mo          January 1st of the next year (annual resets, #51508)
    Nh            next hour that is a multiple of N
    daily/weekly/monthly/hourly   aliases of 24h, 7d, 30d, 1h

Durations that cannot be persisted (`0d`, `2mo`, garbage) are refused by the management API with HTTP 400 and
`Invalid budget_duration '<value>'. Use a format like '1h', '24h', '7d', or '30d'.`
(`timezone_utils.py::budget_duration_error`). Expected instants are computed in the test from the wall clock
with the same rules, which is acceptable because the rules are the contract being pinned; a test that straddles
a boundary retries once.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo

import pytest

from tests.integration._support.client import Gateway, Scenario, eventually
from tests.integration._support.process import owned_proxy
from tests.integration.spend.dec1_budgets_support import counter, redis_client
from tests.integration.spend.dec1_spend_support import (
    CHAT_COST,
    LANDING_SECONDS,
    ask,
    chat_model,
    hashed,
    new_customer,
    proxy_config,
    scripted_upstream,
    served,
    spend_of,
    spend_reaches,
)

pytestmark: Final = pytest.mark.shares_worker_state

_FAST_RESET: Final = {"PROXY_BUDGET_RESCHEDULER_MIN_TIME": "2", "PROXY_BUDGET_RESCHEDULER_MAX_TIME": "3"}


def _key_reset_at(gateway: Gateway, key: str) -> datetime:
    rows: Final = gateway.rows(
        "SELECT to_char(budget_reset_at, 'YYYY-MM-DD HH24:MI:SS.US') AS reset_at FROM \"LiteLLM_VerificationToken\" WHERE token = %s",
        (hashed(key),),
    )
    value: Final = rows[0]["reset_at"]
    assert value is not None, rows
    return datetime.fromisoformat(str(value)).replace(tzinfo=timezone.utc)


def _midnight_after(now: datetime, days: int = 1) -> datetime:
    return now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=days)


def _first_of_next_month(now: datetime) -> datetime:
    first: Final = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return first.replace(year=first.year + 1, month=1) if first.month == 12 else first.replace(month=first.month + 1)


def _reset_for(scenario: Scenario, gateway: Gateway, duration: str) -> tuple[datetime, datetime]:
    before: Final = datetime.now(timezone.utc)
    key: Final = scenario.key(max_budget=1.0, budget_duration=duration)
    return before, _key_reset_at(gateway, key)


def test_a_one_day_budget_resets_at_the_next_utc_midnight(gateway: Gateway) -> None:
    """`/key/generate` with `budget_duration: "1d"` stores `budget_reset_at` at the next 00:00:00 UTC, not 24 hours
    from now. Every key with a daily budget therefore resets at the same instant."""
    with gateway.scenario() as scenario:
        before, reset = _reset_for(scenario, gateway, "1d")
        assert reset == _midnight_after(before), (before, reset)


def test_a_seven_day_budget_resets_on_the_next_monday_at_midnight(gateway: Gateway) -> None:
    """`budget_duration: "7d"` resets at 00:00 UTC on the next Monday (`_handle_day_reset`, value 7), so weekly
    budgets follow the calendar week. On a Monday after midnight that is the Monday a week later."""
    with gateway.scenario() as scenario:
        before, reset = _reset_for(scenario, gateway, "7d")
        days: Final = (7 - before.weekday()) % 7 or 7
        assert reset == _midnight_after(before, days), (before, reset)
        assert reset.weekday() == 0


@pytest.mark.parametrize("duration", ["30d", "1mo", "monthly"])
def test_a_monthly_budget_resets_on_the_first_of_the_next_month(gateway: Gateway, duration: str) -> None:
    """`30d`, `1mo` and the alias `monthly` all reset at 00:00 UTC on the 1st of the next calendar month, so a
    "30 day" budget is really a calendar month budget (`_handle_day_reset` sends 30 to `_handle_month_reset`)."""
    with gateway.scenario() as scenario:
        before, reset = _reset_for(scenario, gateway, duration)
        assert reset == _first_of_next_month(before), (duration, before, reset)


def test_an_annual_budget_resets_on_january_first(gateway: Gateway) -> None:
    """`budget_duration: "12mo"` (annual resets, #51508) stores 00:00 UTC on January 1st of the next year."""
    with gateway.scenario() as scenario:
        before, reset = _reset_for(scenario, gateway, "12mo")
        assert reset == datetime(before.year + 1, 1, 1, tzinfo=timezone.utc), (before, reset)


def test_an_hourly_budget_resets_on_the_next_whole_hour(gateway: Gateway) -> None:
    """`budget_duration: "1h"` resets at the next whole hour (minute, second and microsecond zero)."""
    with gateway.scenario() as scenario:
        before, reset = _reset_for(scenario, gateway, "1h")
        assert reset == before.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1), (before, reset)


def test_the_daily_alias_resets_at_the_next_midnight(gateway: Gateway) -> None:
    """`budget_duration: "daily"` is an alias of `24h`; 24 hour alignment lands on the next 00:00 UTC."""
    with gateway.scenario() as scenario:
        before, reset = _reset_for(scenario, gateway, "daily")
        assert reset == _midnight_after(before), (before, reset)


@pytest.mark.parametrize("duration", ["0d", "2mo", "soon"])
def test_a_budget_duration_that_cannot_be_scheduled_is_refused_with_400(gateway: Gateway, duration: str) -> None:
    """`/key/generate` with `budget_duration` `0d` (would be permanently due and starve the reset job), `2mo`
    (calendar resets support only 1 or 12 months) or a word that is not a duration is refused with HTTP 400 and a
    body containing `Invalid budget_duration '<value>'. Use a format like '1h', '24h', '7d', or '30d'.`
    UNVERIFIED: assumes `/key/generate` runs `validate_budget_duration` like `/budget/new` and `/customer/new`
    (`management_endpoints/common_utils.py:28`) and that `soon` fails parsing rather than defaulting to midnight."""
    response: Final = gateway.request("POST", "/key/generate", {"max_budget": 1.0, "budget_duration": duration})
    assert response.status_code == 400, response.text
    assert f"Invalid budget_duration '{duration}'. Use a format like '1h', '24h', '7d', or '30d'." in response.text


def test_a_team_budget_duration_gets_the_same_calendar_reset(gateway: Gateway) -> None:
    """`/team/new` with `max_budget` 1.0 and `budget_duration: "1d"` stores the team's `budget_reset_at` at the
    next 00:00 UTC, the same rule as keys."""
    with gateway.scenario() as scenario:
        before: Final = datetime.now(timezone.utc)
        team: Final = scenario.team(max_budget=1.0, budget_duration="1d")
        rows: Final = gateway.rows(
            "SELECT to_char(budget_reset_at, 'YYYY-MM-DD HH24:MI:SS.US') AS reset_at FROM \"LiteLLM_TeamTable\" WHERE team_id = %s",
            (team,),
        )
        assert datetime.fromisoformat(str(rows[0]["reset_at"])).replace(tzinfo=timezone.utc) == _midnight_after(before)


def test_a_customer_budget_duration_is_scheduled_on_its_budget_row(gateway: Gateway) -> None:
    """`/customer/new` with `max_budget` 1.0 and `budget_duration: "1d"` creates a `LiteLLM_BudgetTable` row whose
    `budget_reset_at` is the next 00:00 UTC; customer budgets reset on the budget row's window (#51582)."""
    with gateway.scenario() as scenario:
        before: Final = datetime.now(timezone.utc)
        customer: Final = new_customer(gateway, scenario, max_budget=1.0, budget_duration="1d")
        rows: Final = gateway.rows(
            "SELECT to_char(b.budget_reset_at, 'YYYY-MM-DD HH24:MI:SS.US') AS reset_at FROM \"LiteLLM_EndUserTable\" e "
            'JOIN "LiteLLM_BudgetTable" b ON b.budget_id = e.budget_id WHERE e.user_id = %s',
            (customer,),
        )
        assert len(rows) == 1, rows
        assert datetime.fromisoformat(str(rows[0]["reset_at"])).replace(tzinfo=timezone.utc) == _midnight_after(before)


@pytest.mark.timeout(240)
def test_the_reset_timezone_setting_moves_a_daily_reset_to_local_midnight(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy started with `litellm_settings.timezone: "Asia/Kolkata"` stores a `1d` key reset at the next local
    midnight in Kolkata, which is 18:30 UTC. Operators outside UTC rely on this to reset at their own midnight."""
    config: Final = proxy_config(tmp_path, litellm_settings={"timezone": "Asia/Kolkata"})
    with owned_proxy(gateway, tmp_path, {}, config=config) as owned, owned.scenario() as scenario:
        before, reset = _reset_for(scenario, owned, "1d")
        local: Final = before.astimezone(ZoneInfo("Asia/Kolkata"))
        expected: Final = (local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).astimezone(
            timezone.utc
        )
        assert reset == expected, (before, reset)
        assert (reset.hour, reset.minute) == (18, 30)


@pytest.mark.timeout(240)
def test_the_budget_reset_time_setting_moves_a_daily_reset_to_that_wall_clock_time(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A proxy started with `litellm_settings.budget_reset_time: "12:00"` stores a `1d` key reset at the next
    12:00:00 UTC (today if it is still morning, tomorrow otherwise)."""
    config: Final = proxy_config(tmp_path, litellm_settings={"budget_reset_time": "12:00"})
    with owned_proxy(gateway, tmp_path, {}, config=config) as owned, owned.scenario() as scenario:
        before, reset = _reset_for(scenario, owned, "1d")
        noon: Final = before.replace(hour=12, minute=0, second=0, microsecond=0)
        assert reset == (noon if noon > before else noon + timedelta(days=1)), (before, reset)


@pytest.mark.timeout(240)
def test_a_scheduled_reset_also_clears_the_cross_instance_spend_counter(gateway: Gateway, tmp_path: Path) -> None:
    """A key with `max_budget` 1.0 and `budget_duration` `5s` spends 0.02 on a proxy whose reset job runs every
    few seconds. After the reset, the stored spend is 0 and the Redis counter `spend:key:<hash>` that every
    instance reads for the budget check is gone or 0 as well. A reset that only zeroed the database row would
    leave other instances refusing the key until the counter's TTL expired."""
    with (
        scripted_upstream() as wire,
        owned_proxy(gateway, tmp_path, _FAST_RESET) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=1.0, budget_duration="5s")
        served(ask(owned, model, key))
        spend_reaches(owned, "key", hashed(key), CHAT_COST)
        eventually(lambda: spend_of(owned, "key", hashed(key)), lambda spend: spend == 0, seconds=LANDING_SECONDS)
        cleared: Final = eventually(
            lambda: counter(cache, f"spend:key:{hashed(key)}"),
            lambda value: value is None or value == pytest.approx(0.0),
            seconds=LANDING_SECONDS,
        )
        assert cleared is None or cleared == pytest.approx(0.0)


@pytest.mark.timeout(240)
def test_a_scheduled_reset_moves_budget_reset_at_to_the_next_aligned_boundary(gateway: Gateway, tmp_path: Path) -> None:
    """After the reset job resets a `5s` key, its new `budget_reset_at` is later than the old one and lands on a
    whole-second boundary that is a multiple of 5 seconds (`_handle_second_reset`), not "reset time + 5s" drifted
    by however late the job ran."""
    with owned_proxy(gateway, tmp_path, _FAST_RESET) as owned, owned.scenario() as scenario:
        key: Final = scenario.key(max_budget=1.0, budget_duration="5s")
        first: Final = _key_reset_at(owned, key)
        later: Final = eventually(
            lambda: _key_reset_at(owned, key), lambda value: value > first, seconds=LANDING_SECONDS
        )
        assert later.microsecond == 0 and later.second % 5 == 0, later
