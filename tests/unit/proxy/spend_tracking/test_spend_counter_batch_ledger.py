from typing import Final
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import litellm.proxy.proxy_server as ps
from tests.unit.proxy.spend_counters_ledger_twin import (
    RedisCalls,
    force_ledger_windows,
    ledger_scenarios,
    rust_redis_store,
)
from tests.unit.proxy.spend_tracking import test_spend_counter_batch as scenarios
from tests.unit.proxy.spend_tracking.native_budgets_replica import RespRedis

globals().update(ledger_scenarios(scenarios))


@pytest.fixture(autouse=True)
def ledger_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    force_ledger_windows(monkeypatch)


# The two post-call scenarios below drove FakeSpendCounters. They are redefined under the same name on the Rust store
# the ledger's counter step runs on, over a RESP server whose commands the test reads


_POST_CALL_KEYS: Final[frozenset[str]] = frozenset(scenarios.POST_CALL_KEYS)


def _round_trip_commands(redis: RespRedis) -> list[tuple[str, ...]]:
    return [command for command in redis.commands if command[0].upper() != "CLIENT"]


@pytest.mark.asyncio
async def test_post_call_increment_for_every_entity_costs_one_mget_and_one_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis: Final = RespRedis(values=dict.fromkeys(_POST_CALL_KEYS, 1.0))
    calls: Final = RedisCalls()
    monkeypatch.setattr(ps, "prisma_client", None)

    with rust_redis_store(monkeypatch, redis, calls):
        await ps.increment_spend_counters(
            token="hashed",
            team_id="team",
            user_id="user",
            org_id="org",
            end_user_id="eu",
            tags=["prod"],
            model_access_groups=["premium"],
            response_cost=0.5,
        )
        reported: Final = await calls.reported(2)

    commands: Final = _round_trip_commands(redis)
    assert reported == ("MGET", "pipeline")
    assert set(commands[0][1:]) == _POST_CALL_KEYS
    assert {command[1] for command in commands if command[0] == "INCRBYFLOAT"} == _POST_CALL_KEYS
    assert {key: redis.values[key] for key in _POST_CALL_KEYS} == dict.fromkeys(_POST_CALL_KEYS, 1.5)


@pytest.mark.asyncio
async def test_post_call_cold_counters_seed_from_the_mget_miss_without_a_second_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis: Final = RespRedis(values={"spend:key:hashed": 1.0})
    calls: Final = RedisCalls()
    prisma: Final = SimpleNamespace(
        db=SimpleNamespace(
            litellm_teamtable=SimpleNamespace(find_unique=AsyncMock(return_value=SimpleNamespace(spend=4.0)))
        )
    )
    monkeypatch.setattr(ps, "prisma_client", prisma)

    with rust_redis_store(monkeypatch, redis, calls):
        await ps.increment_spend_counters(token="hashed", team_id="team", user_id=None, response_cost=0.5)
        reported: Final = await calls.reported(3)

    seeds: Final = [command[:3] for command in _round_trip_commands(redis) if command[0] == "SET"]
    assert reported == ("MGET", "SET", "pipeline")
    assert seeds == [("SET", "spend:team:team", "4.0")]
    assert redis.values["spend:key:hashed"] == 1.5
    assert redis.values["spend:team:team"] == 4.5
