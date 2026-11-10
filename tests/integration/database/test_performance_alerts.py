import asyncio
import hashlib
import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Final

import psycopg
import pytest
import pytest_asyncio
from psycopg import sql

from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.spend_tracking.performance_alerts import run_performance_alerts
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.repositories.alert_repository import AlertRepository
from litellm.repositories.performance_alert_repository import PerformanceAlertRepository
from tests.integration._support.database import read_rows, write_rows

DAY: Final = date(2026, 1, 8)
NOW: Final = datetime(2026, 1, 9, 2, 0, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class Day:
    """One day of usage for a resource: spend, requests, failed requests, total response ms and timed requests."""

    spend: float = 1.0
    requests: int = 1000
    failed: int = 10
    response_ms: int = 100_000
    timed: int = 1000


STEADY: Final = Day()


_SPEND_TABLE_INSERT: Final = sql.SQL(
    "INSERT INTO {table} (id, {column}, date, spend, api_requests, successful_requests, failed_requests,"
    " total_response_time_ms, timed_requests, api_key, model, custom_llm_provider, mcp_namespaced_tool_name,"
    " endpoint, updated_at) VALUES (gen_random_uuid()::text, %s, %s, %s, %s, %s, %s, %s, %s, 'k', 'm', 'p', '',"
    " '', now())"
)
_KEY_INSERT: Final = (
    'INSERT INTO "LiteLLM_DailyUserSpend" (id, api_key, date, spend, api_requests, successful_requests,'
    " failed_requests, total_response_time_ms, timed_requests, user_id, model, custom_llm_provider,"
    " mcp_namespaced_tool_name, endpoint, updated_at) VALUES (gen_random_uuid()::text, %s, %s, %s, %s, %s, %s, %s,"
    " %s, '', 'm', 'p', '', '', now())"
)
_MODEL_INSERT: Final = (
    'INSERT INTO "LiteLLM_DailyModelUsage" (model_group, date, spend, request_count, successful_requests,'
    " failed_requests, total_response_time_ms, timed_requests, model, custom_llm_provider, task_type, updated_at)"
    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'm', 'p', 'uncategorized', now())"
)


def _seed(scope: str, resource: str, baseline: Day, evaluated: Day | None) -> None:
    days: Final = tuple((DAY - timedelta(days=offset), baseline) for offset in range(1, 8)) + (
        ((DAY, evaluated),) if evaluated is not None else ()
    )
    rows: Final = tuple(
        (resource, day.isoformat(), usage.spend, usage.requests, usage.requests - usage.failed, usage.failed)
        + (usage.response_ms, usage.timed)
        for day, usage in days
    )
    query: Final = {
        "team": _SPEND_TABLE_INSERT.format(
            table=sql.Identifier("LiteLLM_DailyTeamSpend"),
            column=sql.Identifier("team_id"),
        ),
        "user": _SPEND_TABLE_INSERT.format(
            table=sql.Identifier("LiteLLM_DailyUserSpend"),
            column=sql.Identifier("user_id"),
        ),
        "key": sql.SQL(_KEY_INSERT),
        "model": sql.SQL(_MODEL_INSERT),
    }[scope]
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection, connection.cursor() as cursor:
        cursor.executemany(query, rows)


@dataclass(frozen=True, slots=True)
class Fixture:
    prefix: str
    repository: PerformanceAlertRepository
    alerts: AlertRepository

    def key(self, name: str) -> str:
        return hashlib.sha256(f"{self.prefix}-{name}".encode()).hexdigest()

    async def sweep(self) -> None:
        for _ in range(20):
            await run_performance_alerts(self.repository, interval=timedelta(0), now=lambda: NOW)
            state = read_rows(
                'SELECT completed_at IS NOT NULL AS completed FROM "LiteLLM_PerformanceAlertState" WHERE id = 1', ()
            )
            if state and state[0]["completed"] is True:
                return
        pytest.fail("the sweep did not complete")

    def saved(self) -> dict[tuple[str, str], tuple[str, str]]:
        rows: Final = read_rows(
            'SELECT resource_type, resource_id, type, status FROM "LiteLLM_Alert" WHERE resource_id LIKE %s'
            " OR resource_id = %s",
            (f"{self.prefix}%", self.key("slow")),
        )
        return {
            (str(row["resource_id"]), str(row["type"])): (str(row["resource_type"]), str(row["status"])) for row in rows
        }


@pytest_asyncio.fixture(loop_scope="function")
async def fixture() -> AsyncIterator[Fixture]:
    prefix: Final = f"perf-{uuid.uuid4().hex[:12]}"
    client: Final = PrismaClient(os.environ["DATABASE_URL"], ProxyLogging(UserApiKeyCache()))
    await client.connect()
    write_rows('DELETE FROM "LiteLLM_PerformanceAlertState"', ())
    built: Final = Fixture(prefix, PerformanceAlertRepository(client), AlertRepository(client))
    try:
        yield built
    finally:
        try:
            pattern = f"{prefix}%"
            write_rows('DELETE FROM "LiteLLM_DailyTeamSpend" WHERE team_id LIKE %s', (pattern,))
            write_rows(
                'DELETE FROM "LiteLLM_DailyUserSpend" WHERE user_id LIKE %s OR api_key = %s',
                (pattern, built.key("slow")),
            )
            write_rows('DELETE FROM "LiteLLM_DailyModelUsage" WHERE model_group LIKE %s', (pattern,))
            write_rows('DELETE FROM "LiteLLM_UserTable" WHERE user_id LIKE %s', (pattern,))
            write_rows(
                'DELETE FROM "LiteLLM_Alert" WHERE resource_id LIKE %s OR resource_id = %s'
                " OR (resource_type = 'gateway' AND window_start = %s::timestamp)",
                (pattern, built.key("slow"), DAY.isoformat()),
            )
            write_rows('DELETE FROM "LiteLLM_PerformanceAlertState"', ())
        finally:
            await client.disconnect()


@pytest.mark.asyncio
async def test_a_full_sweep_saves_one_alert_per_real_change_and_nothing_for_normal_or_new_usage(
    fixture: Fixture,
) -> None:
    p: Final = fixture.prefix
    _seed("team", f"{p}-team-spike", Day(spend=10.0), Day(spend=100.0))
    _seed("team", f"{p}-team-steady", STEADY, STEADY)
    _seed("team", f"{p}-team-new", Day(requests=0, failed=0, timed=0, spend=0), Day(spend=900.0))
    write_rows('INSERT INTO "LiteLLM_UserTable" (user_id) VALUES (%s)', (f"{p}-user-stopped",))
    _seed("user", f"{p}-user-stopped", STEADY, None)
    _seed("user", f"{p}-user-deleted", STEADY, None)
    _seed("key", fixture.key("slow"), STEADY, Day(response_ms=1_000_000))
    _seed("model", f"{p}-model", STEADY, Day(failed=100))

    await fixture.sweep()

    (age,) = read_rows(
        "SELECT max(abs(extract(epoch FROM created_at - (now() AT TIME ZONE 'UTC'))))::int AS seconds"
        ' FROM "LiteLLM_Alert" WHERE resource_id LIKE %s',
        (f"{p}%",),
    )
    assert int(str(age["seconds"])) < 300
    assert fixture.saved() == {
        (f"{p}-team-spike", "spend"): ("team", "open"),
        (f"{p}-user-stopped", "traffic"): ("user", "open"),
        (fixture.key("slow"), "latency"): ("key", "open"),
        (f"{p}-model", "reliability"): ("model", "open"),
    }


@pytest.mark.asyncio
async def test_a_replayed_day_keeps_one_alert_and_its_dismissal(fixture: Fixture) -> None:
    team: Final = f"{fixture.prefix}-team-spike"
    _seed("team", team, Day(spend=10.0), Day(spend=100.0))
    await fixture.sweep()
    (saved,) = read_rows('SELECT id FROM "LiteLLM_Alert" WHERE resource_id = %s', (team,))
    await fixture.alerts.set_status(int(str(saved["id"])), "dismissed", "admin")

    write_rows('DELETE FROM "LiteLLM_PerformanceAlertState"', ())
    await fixture.sweep()

    assert fixture.saved() == {(team, "spend"): ("team", "dismissed")}


@pytest.mark.asyncio
async def test_a_resource_that_cannot_be_read_is_skipped_without_blocking_the_rest(fixture: Fixture) -> None:
    p: Final = fixture.prefix
    _seed("team", f"{p}-team-a-broken", STEADY, Day(requests=-5, failed=0))
    _seed("team", f"{p}-team-b-spike", Day(spend=10.0), Day(spend=100.0))

    await fixture.sweep()

    (state,) = read_rows('SELECT skipped FROM "LiteLLM_PerformanceAlertState" WHERE id = 1', ())
    assert int(str(state["skipped"])) >= 1
    assert fixture.saved() == {(f"{p}-team-b-spike", "spend"): ("team", "open")}


@pytest.mark.asyncio
async def test_only_one_of_many_concurrent_workers_gets_each_run(fixture: Fixture) -> None:
    claims: Final = await asyncio.gather(*(fixture.repository.claim(timedelta(minutes=5)) for _ in range(8)))
    assert sum(claim is not None for claim in claims) == 1


@pytest.mark.parametrize(("seconds_until_due", "claimed"), ((20, True), (60, False)))
@pytest.mark.asyncio
async def test_a_timer_firing_just_before_the_run_is_due_still_takes_it(
    fixture: Fixture, seconds_until_due: int, claimed: bool
) -> None:
    write_rows(
        'INSERT INTO "LiteLLM_PerformanceAlertState" (id, owner, next_run_at) VALUES (1, %s, now() + %s::interval)',
        ("earlier-worker", f"{seconds_until_due} seconds"),
    )
    assert (await fixture.repository.claim(timedelta(minutes=5)) is not None) is claimed
