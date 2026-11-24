from collections import Counter
from typing import Final
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import litellm.proxy.proxy_server as ps
from litellm.redis.stores.auth_cache import AuthObject
from tests.unit.proxy.spend_counters_ledger_twin import (
    RedisCalls,
    force_ledger_windows,
    ledger_scenarios,
    rust_redis_store,
)
from tests.unit.proxy.spend_tracking import test_spend_counter_batch as scenarios
from tests.unit.proxy.spend_tracking.native_budgets_replica import LostSeedReplyRedis, RespRedis

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
    assert Counter[str](reported) == Counter[str]({"MGET": 1, "pipeline": 1})
    assert commands[0][0] == "MGET"
    assert Counter[str](command[0] for command in commands)["MGET"] == 1
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

    commands: Final = _round_trip_commands(redis)
    seeds: Final = [command[:3] for command in commands if command[0] == "SET"]
    assert Counter[str](reported) == Counter[str]({"MGET": 1, "SET": 1, "pipeline": 1})
    assert [command[0] for command in commands[:2]] == ["MGET", "SET"]
    assert Counter[str](command[0] for command in commands)["MGET"] == 1
    assert seeds == [("SET", "spend:team:team", "4.0")]
    assert redis.values["spend:key:hashed"] == 1.5
    assert redis.values["spend:team:team"] == 4.5


class _CachedSpend:
    def __init__(self, spend: float) -> None:
        self.spend: Final = spend

    async def async_get_cache(self, key: object, **_: object) -> object:
        return {"spend": self.spend}


@pytest.mark.asyncio
async def test_cold_seed_whose_reply_is_lost_leaves_no_counter_in_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    redis: Final = LostSeedReplyRedis(values={}, counter="spend:end_user:eu")
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(ps, "user_api_key_cache", _CachedSpend(6.0))

    with rust_redis_store(monkeypatch, redis, socket_timeout=0.5):
        with pytest.raises(Exception, match="timeout"):
            await ps._ensure_spend_counter_initialized(
                counter_key="spend:end_user:eu", source_cache_key=AuthObject.end_user("eu")
            )

    assert redis.count("EVAL") == 1
    assert "spend:end_user:eu" not in redis.values


@pytest.mark.asyncio
async def test_post_call_cold_seed_whose_reply_is_lost_leaves_no_counter_in_redis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis: Final = LostSeedReplyRedis(values={}, counter="spend:end_user:eu")
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(ps, "user_api_key_cache", _CachedSpend(6.0))

    with rust_redis_store(monkeypatch, redis, socket_timeout=0.5):
        with pytest.raises(Exception, match="timeout"):
            await ps.increment_spend_counters(
                token=None, team_id=None, user_id=None, end_user_id="eu", response_cost=0.5
            )

    assert redis.count("EVAL") == 1
    assert "spend:end_user:eu" not in redis.values


def _refuses_the_seed_increment(command: tuple[str, ...]) -> bool:
    """The seed's own INCRBYFLOAT, of nothing, as a Redis Cluster slot mid-migration refuses a pipelined command."""
    return command[0] == "INCRBYFLOAT" and command[1] == "spend:end_user:eu" and float(command[2]) == 0.0


@pytest.mark.asyncio
async def test_cold_seed_the_increment_cannot_follow_drops_that_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    redis: Final = RespRedis(values={}, fail_once=[_refuses_the_seed_increment])
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(ps, "user_api_key_cache", _CachedSpend(6.0))

    with rust_redis_store(monkeypatch, redis):
        with pytest.raises(Exception, match="rejected"):
            await ps._ensure_spend_counter_initialized(
                counter_key="spend:end_user:eu", source_cache_key=AuthObject.end_user("eu")
            )

    assert redis.count("EVAL") == 1
    assert "spend:end_user:eu" not in redis.values


@pytest.mark.asyncio
async def test_post_call_cold_seed_the_increment_cannot_follow_still_charges_the_other_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis: Final = RespRedis(values={"spend:key:hashed": 1.0}, fail_once=[_refuses_the_seed_increment])
    monkeypatch.setattr(ps, "prisma_client", None)
    monkeypatch.setattr(ps, "user_api_key_cache", _CachedSpend(6.0))

    with rust_redis_store(monkeypatch, redis):
        with pytest.raises(Exception, match="rejected"):
            await ps.increment_spend_counters(
                token="hashed", team_id=None, user_id=None, end_user_id="eu", response_cost=0.5
            )

    assert redis.values == {"spend:key:hashed": 1.5}
