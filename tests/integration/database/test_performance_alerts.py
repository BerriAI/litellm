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
from litellm_proxy_extras.request_log_indexes import ensure_request_log_indexes
from psycopg import sql

from litellm.models.alert import AlertDetection, DeprecationEvidence
from litellm.models.performance_alert import FREE_ALERT_LIMIT, SweepOwnershipLostError, SweepState
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.spend_tracking.performance_alerts import run_performance_alerts
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.repositories.alert_repository import AlertRepository
from litellm.repositories.performance_alert_repository import PerformanceAlertRepository
from tests.integration._support.client import Gateway
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
    client: PrismaClient

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
    built: Final = Fixture(prefix, PerformanceAlertRepository(client, enterprise=True), AlertRepository(client), client)
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
    assert ensure_request_log_indexes(os.environ["DATABASE_URL"], "public")
    p: Final = fixture.prefix
    _seed("team", f"{p}-team-spike", Day(spend=10.0), Day(spend=100.0))
    _seed("team", f"{p}-team-steady", STEADY, STEADY)
    _seed("team", f"{p}-team-new", Day(requests=0, failed=0, timed=0, spend=0), Day(spend=900.0))
    write_rows('INSERT INTO "LiteLLM_UserTable" (user_id) VALUES (%s)', (f"{p}-user-stopped",))
    _seed("user", f"{p}-user-stopped", STEADY, None)
    _seed("user", f"{p}-user-deleted", STEADY, None)
    _seed("user", f"{p}-user-busy", STEADY, Day(requests=20_000, timed=20_000, response_ms=2_000_000))
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
        (f"{p}-user-busy", "traffic"): ("user", "open"),
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


@pytest.mark.asyncio
async def test_free_allowance_is_atomic_and_enterprise_resumes_the_unfinished_page(fixture: Fixture) -> None:
    free: Final = PerformanceAlertRepository(fixture.client)
    for index in range(FREE_ALERT_LIMIT + 3):
        _seed("team", f"{fixture.prefix}-{index}", Day(spend=10), Day(spend=100))
    claim: Final = await free.claim(timedelta(0))
    assert claim is not None
    state: Final = SweepState(day=DAY, scope="team", cursor=fixture.prefix)
    await asyncio.gather(
        free.evaluate_page(claim.owner, state, 4),
        free.evaluate_page(claim.owner, state.model_copy(update={"cursor": f"{fixture.prefix}-3"}), 4),
    )
    assert len(fixture.saved()) == FREE_ALERT_LIMIT
    assert (await free.monitoring()).alerts_used == FREE_ALERT_LIMIT
    assert await free.claim(timedelta(0)) is None
    assert await free.evaluate_page(claim.owner, state, 25) is None
    upgraded: Final = await fixture.repository.claim(timedelta(0))
    assert upgraded is not None and upgraded.state is not None
    assert not upgraded.state.completed
    await fixture.repository.evaluate_page(upgraded.owner, upgraded.state, 25)
    assert len(fixture.saved()) == FREE_ALERT_LIMIT + 3
    assert (await fixture.repository.monitoring()).alert_limit is None
    assert await free.claim(timedelta(0)) is None


@pytest.mark.asyncio
async def test_retries_status_changes_deletion_and_reconnect_do_not_restore_allowance(fixture: Fixture) -> None:
    free: Final = PerformanceAlertRepository(fixture.client)
    _seed("team", f"{fixture.prefix}-a", Day(spend=10), Day(spend=100))
    claim: Final = await free.claim(timedelta(0))
    assert claim is not None
    state: Final = SweepState(day=DAY, scope="team", cursor=fixture.prefix)
    for _ in range(2):
        await free.evaluate_page(claim.owner, state, 25)
    assert (await free.monitoring()).alerts_used == 1
    (saved,) = read_rows('SELECT id FROM "LiteLLM_Alert" WHERE resource_id = %s', (f"{fixture.prefix}-a",))
    for status in ("resolved", "dismissed", "open"):
        response: Final = await fixture.alerts.set_status(int(str(saved["id"])), status, "admin")
        assert response is not None and response.status == status
    write_rows('DELETE FROM "LiteLLM_Alert" WHERE id = %s', (saved["id"],))
    await free.set_enabled(False)
    await free.set_enabled(True)
    client: Final = PrismaClient(os.environ["DATABASE_URL"], ProxyLogging(UserApiKeyCache()))
    await client.connect()
    try:
        restarted: Final = PerformanceAlertRepository(client)
        assert (await restarted.monitoring()).alerts_used == 1
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_opt_out_is_shared_and_config_false_cannot_be_overridden(fixture: Fixture, gateway: Gateway) -> None:
    response: Final = gateway.request("GET", "/alerts/monitoring")
    assert response.status_code == 200, response.text
    assert response.json() == {"enabled": True, "alerts_used": 0, "alert_limit": 5, "disabled_by_config": False}
    claim: Final = await fixture.repository.claim(timedelta(0))
    assert claim is not None
    disabled: Final = gateway.request("PATCH", "/alerts/monitoring", {"enabled": False})
    assert disabled.status_code == 200, disabled.text
    assert disabled.json()["enabled"] is False
    assert await fixture.repository.claim(timedelta(0)) is None
    assert await fixture.repository.evaluate_page(claim.owner, SweepState(day=DAY), 25) is None
    enabled: Final = gateway.request("PATCH", "/alerts/monitoring", {"enabled": True})
    assert enabled.status_code == 200, enabled.text
    assert (await fixture.repository.monitoring()).enabled
    configured_off: Final = PerformanceAlertRepository(fixture.client, configured=False)
    assert not (await configured_off.set_enabled(True)).enabled
    assert await configured_off.claim(timedelta(0)) is None


@pytest.mark.asyncio
async def test_admin_can_opt_out_while_a_detection_batch_holds_the_state_lock(
    fixture: Fixture,
    gateway: Gateway,
) -> None:
    await fixture.repository.monitoring()
    with psycopg.connect(os.environ["DATABASE_URL"]) as detector:
        detector.execute('SELECT id FROM "LiteLLM_PerformanceAlertState" WHERE id = 1 FOR UPDATE')
        request: Final = asyncio.create_task(
            asyncio.to_thread(gateway.request, "PATCH", "/alerts/monitoring", {"enabled": False})
        )
        try:
            async with asyncio.timeout(3):
                while not request.done():
                    waiting: Final = read_rows(
                        "SELECT pid FROM pg_stat_activity WHERE %s::int = ANY(pg_blocking_pids(pid)) "
                        "AND clock_timestamp() - query_start > interval '200 milliseconds'",
                        (str(detector.info.backend_pid),),
                    )
                    if waiting:
                        break
                    await asyncio.sleep(0.01)
        finally:
            detector.rollback()
        response: Final = await asyncio.wait_for(request, timeout=5)
    assert response.status_code == 200, response.text
    assert response.json()["enabled"] is False
    assert not (await fixture.repository.monitoring()).enabled
    assert await fixture.repository.claim(timedelta(0)) is None


@pytest.mark.asyncio
async def test_existing_performance_alerts_consume_allowance_but_deprecations_do_not(fixture: Fixture) -> None:
    start: Final = datetime(2026, 1, 8, tzinfo=timezone.utc)
    await fixture.alerts.record(
        AlertDetection(
            type="model_deprecation",
            resource_type="model",
            resource_id=fixture.prefix,
            window_start=start,
            window_end=start + timedelta(days=1),
            evidence=DeprecationEvidence(deprecation_date=DAY + timedelta(days=90)),
        )
    )
    assert (await fixture.repository.monitoring()).alerts_used == 0
    _seed("team", f"{fixture.prefix}-a", Day(spend=10), Day(spend=100))
    await fixture.sweep()
    write_rows('UPDATE "LiteLLM_PerformanceAlertState" SET alerts_used = NULL', ())
    free: Final = PerformanceAlertRepository(fixture.client)
    assert (await free.monitoring()).alerts_used == 1


@pytest.mark.asyncio
async def test_losing_ownership_rolls_back_alerts_and_allowance(fixture: Fixture) -> None:
    free: Final = PerformanceAlertRepository(fixture.client)
    for index in range(FREE_ALERT_LIMIT):
        _seed("team", f"{fixture.prefix}-{index}", Day(spend=10), Day(spend=100))
    assert await free.claim(timedelta(0)) is not None
    with pytest.raises(SweepOwnershipLostError):
        await free.evaluate_page("old-owner", SweepState(day=DAY, scope="team", cursor=fixture.prefix), 25)
    assert fixture.saved() == {}
    assert (await free.monitoring()).alerts_used == 0
