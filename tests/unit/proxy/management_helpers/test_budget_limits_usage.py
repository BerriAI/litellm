from datetime import datetime, timezone
from typing import Final

import pytest

from litellm.models.budget_schedule import resolve_schedules_at
from litellm.proxy import proxy_server
from litellm.proxy.management_helpers.budget_limits_usage import build_budget_limits_usage
from litellm.redis.stores.spend_counters import team_window_counter
from tests.unit.redis.stores.spend_counters_double import FakeSpendCounters

_MORNING: Final = "sched:Europe/Berlin:mon,tue,wed,thu,fri:08:00-13:00"
_WINDOWS: Final = (
    {"budget_duration": _MORNING, "max_budget": 10.0, "reset_at": "2026-11-25T12:00:00+00:00"},
    {"budget_duration": "7d", "max_budget": 50.0, "reset_at": "2026-11-30T00:00:00+00:00"},
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "now, morning",
    [
        (
            datetime(2026, 11, 25, 9, 0, tzinfo=timezone.utc),
            {"current_spend": 3.0, "active": True, "next_start": "2026-11-26T07:00:00+00:00"},
        ),
        (
            datetime(2026, 11, 27, 13, 0, tzinfo=timezone.utc),
            {"current_spend": 0.0, "active": False, "next_start": "2026-11-30T07:00:00+00:00"},
        ),
    ],
    ids=["inside_the_period", "after_friday_morning"],
)
async def test_scheduled_window_usage_reports_the_active_period(
    monkeypatch: pytest.MonkeyPatch, now: datetime, morning: dict[str, object]
) -> None:
    counters: Final = FakeSpendCounters(
        redis={
            team_window_counter("t", f"{_MORNING}@2026-11-25T07:00:00+00:00"): 3.0,
            team_window_counter("t", f"{_MORNING}@2026-11-27T07:00:00+00:00"): 9.0,
            team_window_counter("t", "7d"): 12.0,
        }
    )
    monkeypatch.setattr(proxy_server, "spend_counter_store", counters.store())
    monkeypatch.setattr(proxy_server, "prisma_client", None)

    with resolve_schedules_at(now):
        usage: Final = await build_budget_limits_usage(
            _WINDOWS, lambda name: team_window_counter("t", name), "Team", "t"
        )

    assert usage == {_MORNING: morning, "7d": {"current_spend": 12.0}}
