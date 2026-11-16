import asyncio
import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

import psycopg
import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import OwnedState, owned_proxy
from prisma import Prisma
from psycopg import sql
from redis import Redis

from litellm.repositories.unit_of_work import spend_reset_unit_of_work

pytestmark: Final = pytest.mark.shares_worker_state

RESET_LEASE_KEY: Final = "cronjob_lock:reset_budget_job"
PEER_POD_LEASE: Final = json.dumps("integration-peer-pod-holding-the-reset-lease")
FAST_RESET_TICK: Final = MappingProxyType(
    {"PROXY_BUDGET_RESCHEDULER_MIN_TIME": "2", "PROXY_BUDGET_RESCHEDULER_MAX_TIME": "2"}
)


@contextmanager
def _peer_reset_lease() -> Generator["Redis[bytes]", None, None]:
    cache: Final["Redis[bytes]"] = Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))
    with cache:
        eventually(
            lambda: cache.set(RESET_LEASE_KEY, PEER_POD_LEASE, ex=300, nx=True),
            lambda claimed: claimed is True,
            seconds=60,
        )
        try:
            yield cache
        finally:
            cache.delete(RESET_LEASE_KEY)


def _team_spend(team: str) -> float:
    rows: Final = read_rows('SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,))
    assert len(rows) == 1, rows
    spend: Final = rows[0]["spend"]
    assert isinstance(spend, (int, float)), rows
    return float(spend)


def _make_team_budget_due(team: str, spend: float) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        connection.execute(
            "UPDATE \"LiteLLM_TeamTable\" SET spend = %s, budget_reset_at = now() - interval '1 day' "
            "WHERE team_id = %s",
            (spend, team),
        )


@pytest.mark.covers("spend.budget_reset.one_pod_sweeps_per_tick")
@pytest.mark.parametrize("starting_spend", (0.5, 40.080000000000005, 1009.9442791267694))
def test_reset_sweep_skips_ticks_while_another_pod_holds_the_lease_and_resumes_after_release(
    gateway: Gateway, tmp_path: Path, starting_spend: float
) -> None:
    with (
        gateway.scenario() as scenario,
        Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as cache,
    ):
        model: Final = scenario.model(input_cost_per_token=0.0, output_cost_per_token=0.0)
        team: Final = scenario.team(max_budget=2000000.0, budget_duration="30d")
        key: Final = scenario.key(team_id=team)
        _make_team_budget_due(team, spend=starting_spend)
        assert _team_spend(team) == starting_spend
        eventually(
            lambda: cache.set(RESET_LEASE_KEY, PEER_POD_LEASE, ex=120, nx=True),
            lambda claimed: claimed is True,
            seconds=60,
        )
        try:
            with owned_proxy(gateway, tmp_path, FAST_RESET_TICK, workers=2) as replica:
                response: Final = replica.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": "classification request"}]},
                    key=key,
                )
                assert response.status_code == 200, response.text
                held: Final = eventually(
                    lambda: _team_spend(team),
                    lambda spend: spend != starting_spend,
                    seconds=8,
                    return_last_on_timeout=True,
                )
                assert held == starting_spend, (
                    f"team {team} was swept while another pod held the reset lease: spend={held}"
                )
                assert cache.get(RESET_LEASE_KEY) == PEER_POD_LEASE.encode()
                cache.delete(RESET_LEASE_KEY)
                swept: Final = eventually(
                    lambda: _team_spend(team), lambda spend: spend == 0.0, seconds=15, return_last_on_timeout=True
                )
                info: Final = replica.request("GET", f"/team/info?team_id={team}")
                assert info.status_code == 200, info.text
                assert info.json()["team_info"]["spend"] == 0.0
                assert swept == 0.0
                eventually(lambda: cache.get(RESET_LEASE_KEY), lambda value: value is None, seconds=15)
        finally:
            cache.delete(RESET_LEASE_KEY)


def _blocked_reset_updates(table: str) -> int:
    rows: Final = read_rows(
        "SELECT count(*) AS blocked FROM pg_stat_activity WHERE datname = current_database() "
        "AND wait_event_type = 'Lock' AND query LIKE %s AND query LIKE '%%UPDATE%%'",
        (f'%"{table}"%',),
    )
    return int(str(rows[0]["blocked"]))


@pytest.mark.parametrize("entity", ("key", "user", "team"))
@pytest.mark.parametrize(
    ("workers", "spend", "increment", "rollover", "reset_timezone"),
    (
        (3, 40.0, 0.25, False, "UTC"),
        (1, 40.080000000000005, 0.0, False, "UTC"),
        (1, 1234567.8901234567, 0.0, False, "UTC"),
        (1, 1009.9442791267694, 0.0, False, "UTC"),
        (3, 40.000000000000014, 0.0, True, "UTC"),
        (1, 0.30000000000000004, 0.0, False, "America/Los_Angeles"),
    ),
    ids=(
        "overlap-preserves-new-spend",
        "fractional-roundoff",
        "large-roundoff",
        "four-ulp-roundoff",
        "rollover-keeps-small-carry",
        "non-utc-roundoff",
    ),
)
def test_scheduled_reset_preserves_new_spend_and_never_leaves_roundoff(
    gateway: Gateway,
    tmp_path: Path,
    entity: Literal["key", "user", "team"],
    workers: int,
    spend: float,
    increment: float,
    rollover: bool,
    reset_timezone: str,
) -> None:
    max_budget: Final = 40.0 if rollover else 2000000.0
    expected_spend: Final = spend - max_budget + increment if rollover else increment
    config: Final = tmp_path / "without-redis.yaml"
    config.write_text(
        "model_list: []\ngeneral_settings:\n"
        "  master_key: os.environ/LITELLM_MASTER_KEY\n"
        "  database_url: os.environ/DATABASE_URL\n"
        "  store_model_in_db: true\n"
        "litellm_settings:\n  cache: false\n"
        f"  budget_rollover: {str(rollover).lower()}\n"
        f"  timezone: {reset_timezone}\n"
    )
    with gateway.scenario() as scenario, _peer_reset_lease() as peer:
        identity: Final = (
            sha256(scenario.key(max_budget=max_budget, budget_duration="1d").encode()).hexdigest()
            if entity == "key"
            else scenario.user(max_budget=max_budget, budget_duration="1d")
            if entity == "user"
            else scenario.team(max_budget=max_budget, budget_duration="1d")
        )
        table, column = {
            "key": ("LiteLLM_VerificationToken", "token"),
            "user": ("LiteLLM_UserTable", "user_id"),
            "team": ("LiteLLM_TeamTable", "team_id"),
        }[entity]
        with psycopg.connect(os.environ["DATABASE_URL"]) as gate:
            gate.execute(
                sql.SQL(
                    "UPDATE {} SET spend = %s, budget_reset_at = now() - interval '1 day',"
                    " updated_at = now() - interval '1 day' WHERE {} = %s"
                ).format(sql.Identifier(table), sql.Identifier(column)),
                (spend, identity),
            )
            gate.commit()
            gate.execute(
                sql.SQL("SELECT 1 FROM {} WHERE {} = %s FOR UPDATE").format(
                    sql.Identifier(table), sql.Identifier(column)
                ),
                (identity,),
            )
            with owned_proxy(
                gateway,
                tmp_path,
                FAST_RESET_TICK,
                config=config,
                remove_environment=("REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD", "REDIS_URL"),
                workers=workers,
                state=OwnedState.SHARED_WITH_WORKER,
            ) as replica:
                try:
                    blocked: Final = eventually(
                        lambda: _blocked_reset_updates(table), lambda count: count == workers, seconds=30
                    )
                    assert blocked == workers
                    gate.execute(
                        sql.SQL("UPDATE {} SET spend = spend + %s WHERE {} = %s").format(
                            sql.Identifier(table), sql.Identifier(column)
                        ),
                        (increment, identity),
                    )
                    gate.commit()
                    eventually(lambda: _blocked_reset_updates(table), lambda count: count == 0, seconds=30)
                    with psycopg.connect(os.environ["DATABASE_URL"]) as observer:
                        result: Final = observer.execute(
                            sql.SQL(
                                "SELECT spend, budget_reset_at > now(), updated_at > now() - interval '1 hour'"
                                " FROM {} WHERE {} = %s"
                            ).format(sql.Identifier(table), sql.Identifier(column)),
                            (identity,),
                        ).fetchall()
                    parameter, info = {
                        "key": ("key", "info"),
                        "user": ("user_id", "user_info"),
                        "team": ("team_id", "team_info"),
                    }[entity]
                    response: Final = replica.request("GET", f"/{entity}/info?{parameter}={identity}")
                    assert response.status_code == 200, response.text
                    assert response.json()[info]["spend"] == pytest.approx(expected_spend, rel=1e-14, abs=0.0)
                    assert result == [(expected_spend, True, True)], result
                    assert peer.get(RESET_LEASE_KEY) == PEER_POD_LEASE.encode()
                finally:
                    gate.rollback()


@pytest.mark.parametrize("entity", ("key", "user", "team"))
def test_stale_zero_reset_cannot_erase_another_workers_rollover(gateway: Gateway, entity: str) -> None:
    carry: Final = 1.4210854715202004e-14
    previous_reset: Final = datetime(2020, 1, 1, tzinfo=timezone.utc)
    next_reset: Final = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(days=1)
    with gateway.scenario() as scenario:
        identity: Final = (
            sha256(scenario.key(max_budget=40.0, budget_duration="1d").encode()).hexdigest()
            if entity == "key"
            else scenario.user(max_budget=40.0, budget_duration="1d")
            if entity == "user"
            else scenario.team(max_budget=40.0, budget_duration="1d")
        )
        table, column = {
            "key": ("LiteLLM_VerificationToken", "token"),
            "user": ("LiteLLM_UserTable", "user_id"),
            "team": ("LiteLLM_TeamTable", "team_id"),
        }[entity]
        with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
            connection.execute(
                sql.SQL("UPDATE {} SET spend = %s, budget_reset_at = %s WHERE {} = %s").format(
                    sql.Identifier(table), sql.Identifier(column)
                ),
                (40.0 + carry, previous_reset, identity),
            )

        async def commit_winner_then_stale_reset() -> None:
            async with Prisma() as client:
                for spend_after_reset in (carry, 0.0):
                    async with spend_reset_unit_of_work(client.batch_) as writes:
                        writer: Final = {"key": writes.keys, "user": writes.users, "team": writes.teams}[entity]
                        writer.queue_spend_reset(identity, next_reset, 40.0, previous_reset, spend_after_reset)
                    assert read_rows(
                        sql.SQL("SELECT spend FROM {} WHERE {} = %s")
                        .format(sql.Identifier(table), sql.Identifier(column))
                        .as_string(),
                        (identity,),
                    ) == [{"spend": carry}]

        asyncio.run(commit_winner_then_stale_reset())
        parameter, info = {
            "key": ("key", "info"),
            "user": ("user_id", "user_info"),
            "team": ("team_id", "team_info"),
        }[entity]
        response: Final = gateway.request("GET", f"/{entity}/info?{parameter}={identity}")
        assert response.status_code == 200, response.text
        assert response.json()[info]["spend"] == pytest.approx(carry, rel=1e-14, abs=0.0)


@pytest.mark.parametrize("entity", ("user", "team"))
def test_reset_starts_the_window_of_a_row_that_never_had_a_reset_time(gateway: Gateway, entity: str) -> None:
    next_reset: Final = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(days=1)
    with gateway.scenario() as scenario:
        identity: Final = (
            scenario.user(max_budget=100.0, budget_duration="1d")
            if entity == "user"
            else scenario.team(max_budget=100.0, budget_duration="1d")
        )
        table, column, info = {
            "user": ("LiteLLM_UserTable", "user_id", "user_info"),
            "team": ("LiteLLM_TeamTable", "team_id", "team_info"),
        }[entity]
        with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
            connection.execute(
                sql.SQL("UPDATE {} SET spend = 40, budget_reset_at = NULL WHERE {} = %s").format(
                    sql.Identifier(table), sql.Identifier(column)
                ),
                (identity,),
            )

        async def reset() -> None:
            async with Prisma() as client:
                async with spend_reset_unit_of_work(client.batch_) as writes:
                    (writes.users if entity == "user" else writes.teams).queue_spend_reset(
                        identity, next_reset, 40.0, None
                    )

        asyncio.run(reset())
        response: Final = gateway.request("GET", f"/{entity}/info?{column}={identity}")
        assert response.status_code == 200, response.text
        assert response.json()[info]["spend"] == 0.0
        assert response.json()[info]["budget_reset_at"] is not None


@pytest.mark.parametrize("entity", ("key", "user", "team"))
@pytest.mark.parametrize(("spend_decrement", "spend_after_reset"), ((40.0, 0.0), (30.0, 10.0)))
def test_reset_after_a_manual_spend_reset_never_goes_negative(
    gateway: Gateway, entity: str, spend_decrement: float, spend_after_reset: float
) -> None:
    previous_reset: Final = datetime(2020, 1, 1, tzinfo=timezone.utc)
    next_reset: Final = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(days=1)
    with gateway.scenario() as scenario:
        identity: Final = (
            sha256(scenario.key(max_budget=100.0, budget_duration="1d").encode()).hexdigest()
            if entity == "key"
            else scenario.user(max_budget=100.0, budget_duration="1d")
            if entity == "user"
            else scenario.team(max_budget=100.0, budget_duration="1d")
        )
        table, column, parameter, info = {
            "key": ("LiteLLM_VerificationToken", "token", "key", "info"),
            "user": ("LiteLLM_UserTable", "user_id", "user_id", "user_info"),
            "team": ("LiteLLM_TeamTable", "team_id", "team_id", "team_info"),
        }[entity]
        with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
            connection.execute(
                sql.SQL("UPDATE {} SET spend = 0, budget_reset_at = %s WHERE {} = %s").format(
                    sql.Identifier(table), sql.Identifier(column)
                ),
                (previous_reset, identity),
            )

        async def reset_with_the_spend_read_before_the_manual_reset() -> None:
            async with Prisma() as client:
                async with spend_reset_unit_of_work(client.batch_) as writes:
                    {"key": writes.keys, "user": writes.users, "team": writes.teams}[entity].queue_spend_reset(
                        identity, next_reset, spend_decrement, previous_reset, spend_after_reset
                    )

        asyncio.run(reset_with_the_spend_read_before_the_manual_reset())
        response: Final = gateway.request("GET", f"/{entity}/info?{parameter}={identity}")
        assert response.status_code == 200, response.text
        assert response.json()[info]["spend"] == 0.0
