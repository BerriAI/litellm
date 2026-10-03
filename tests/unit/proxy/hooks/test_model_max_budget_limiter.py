import asyncio
import time
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

import pytest

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy._types import Litellm_EntityType
from litellm.proxy.hooks.model_max_budget_limiter import (
    _PROXY_VirtualKeyModelMaxBudgetLimiter,
    model_budget_spend_cache_key,
    team_member_budget_entity_id,
)
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.utils import LiteLLMBatch, Usage

KEY_HASH: Final = "key-hash-batch"
USER_ID: Final = "user-batch"
TEAM_ID: Final = "team-batch"
MODEL_GROUP: Final = "batch-qa-primary"
BATCH_COST: Final = 2.925e-05
CHAT_COST: Final = 0.001
KEY_SPEND_KEY: Final = f"virtual_key_spend:{KEY_HASH}:{MODEL_GROUP}:1d"
USER_SPEND_KEY: Final = f"user_model_spend:{USER_ID}:{MODEL_GROUP}:1d"


def _batch(batch_id: str, status: str) -> LiteLLMBatch:
    return LiteLLMBatch(
        id=batch_id,
        completion_window="24h",
        created_at=1,
        endpoint="/v1/chat/completions",
        input_file_id="file-batch",
        object="batch",
        status=status,
        usage=Usage(prompt_tokens=20, completion_tokens=18, total_tokens=38),
    )


def _event(
    call_type: str,
    response_cost: float,
    team_member_model_max_budget: Mapping[str, object] | None = None,
    user_id: str = USER_ID,
    team_id: str = TEAM_ID,
) -> dict[str, object]:
    return {
        "call_type": call_type,
        "standard_logging_object": {
            "call_type": call_type,
            "response_cost": response_cost,
            "model": "openai/gpt-5.4-mini",
            "model_group": MODEL_GROUP,
            "metadata": {
                "user_api_key_hash": KEY_HASH,
                "user_api_key_user_id": user_id,
                "user_api_key_team_id": team_id,
            },
        },
        "litellm_params": {
            "metadata": {
                "user_api_key_model_max_budget": {MODEL_GROUP: {"budget_limit": 0.0001, "time_period": "1d"}},
                "user_api_key_user_model_max_budget": {MODEL_GROUP: {"budget_limit": 0.0001, "time_period": "1d"}},
                **(
                    {"user_api_key_team_member_model_max_budget": team_member_model_max_budget}
                    if team_member_model_max_budget is not None
                    else {}
                ),
            }
        },
    }


async def _poll(limiter: _PROXY_VirtualKeyModelMaxBudgetLimiter, batch: LiteLLMBatch, response_cost: float) -> None:
    await limiter.async_log_success_event(
        _event("aretrieve_batch", response_cost), response_obj=batch, start_time=None, end_time=None
    )


async def _chat(
    limiter: _PROXY_VirtualKeyModelMaxBudgetLimiter,
    team_member_model_max_budget: Mapping[str, object] | None = None,
    user_id: str = USER_ID,
    team_id: str = TEAM_ID,
) -> None:
    await limiter.async_log_success_event(
        _event(
            "acompletion",
            CHAT_COST,
            team_member_model_max_budget=team_member_model_max_budget,
            user_id=user_id,
            team_id=team_id,
        ),
        response_obj=None,
        start_time=None,
        end_time=None,
    )


async def _spend(limiter: _PROXY_VirtualKeyModelMaxBudgetLimiter, spend_key: str) -> float:
    return await limiter.dual_cache.async_get_cache(key=spend_key) or 0.0


def _local_spend(limiter: _PROXY_VirtualKeyModelMaxBudgetLimiter, spend_key: str) -> float:
    return limiter.dual_cache.in_memory_cache.get_cache(key=spend_key) or 0.0


@pytest.mark.asyncio
async def test_team_member_budget_is_charged_and_enforced_after_success():
    budget: Final = {MODEL_GROUP: {"max_budget": 0.0005, "budget_duration": "1d"}}
    limiter: Final = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())

    await _chat(limiter, team_member_model_max_budget=budget)

    entity_id: Final = team_member_budget_entity_id(user_id=USER_ID, team_id=TEAM_ID)
    spend_key: Final = model_budget_spend_cache_key(
        entity_type=Litellm_EntityType.TEAM_MEMBER,
        entity_id=entity_id,
        budget_model=MODEL_GROUP,
        budget_duration="1d",
    )
    assert await _spend(limiter, spend_key) == CHAT_COST

    with pytest.raises(litellm.BudgetExceededError, match="LiteLLM Team Member"):
        await limiter.is_team_member_within_model_budget(
            user_id=USER_ID,
            team_id=TEAM_ID,
            team_member_model_max_budget=budget,
            model=MODEL_GROUP,
        )


@pytest.mark.asyncio
async def test_team_member_budget_spend_isolated_by_user_and_team():
    budget: Final = {MODEL_GROUP: {"max_budget": CHAT_COST * 2, "budget_duration": "1d"}}
    limiter: Final = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())

    await _chat(limiter, team_member_model_max_budget=budget)

    assert await limiter.is_team_member_within_model_budget(
        user_id="different-user",
        team_id=TEAM_ID,
        team_member_model_max_budget=budget,
        model=MODEL_GROUP,
    )
    assert await limiter.is_team_member_within_model_budget(
        user_id=USER_ID,
        team_id="different-team",
        team_member_model_max_budget=budget,
        model=MODEL_GROUP,
    )


@pytest.mark.asyncio
async def test_zero_team_member_model_budget_rejects_first_request():
    limiter: Final = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())

    with pytest.raises(litellm.BudgetExceededError, match="LiteLLM Team Member"):
        await limiter.is_team_member_within_model_budget(
            user_id=USER_ID,
            team_id=TEAM_ID,
            team_member_model_max_budget={MODEL_GROUP: {"max_budget": 0, "budget_duration": "1d"}},
            model=MODEL_GROUP,
        )


class _Clock:
    def __init__(self) -> None:
        self.seconds = 0.0

    def now(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        self.seconds = self.seconds + seconds


class _SharedRedisDouble:
    def __init__(self, now: Callable[[], float] = time.time) -> None:
        self.now = now
        self.entries: Mapping[str, tuple[float, float | None]] = MappingProxyType({})

    def _live(self, key: str) -> tuple[float, float | None] | None:
        entry: Final = self.entries.get(key)
        if entry is None:
            return None
        expires_at: Final = entry[1]
        if expires_at is not None and expires_at <= self.now():
            return None
        return entry

    def _store(self, key: str, value: float, expires_at: float | None) -> None:
        self.entries = MappingProxyType({**self.entries, key: (value, expires_at)})

    async def async_get_cache(self, key: str, **kwargs: object) -> float | None:
        await asyncio.sleep(0)
        entry: Final = self._live(key)
        return None if entry is None else entry[0]

    async def async_set_cache(self, key: str, value: float, ttl: int | None = None, **kwargs: object) -> None:
        await asyncio.sleep(0)
        self._store(key, value, None if ttl is None else self.now() + ttl)

    async def async_increment(
        self,
        key: str,
        value: float,
        ttl: int | None = None,
        parent_otel_span: object = None,
        refresh_ttl: bool = False,
    ) -> float:
        await asyncio.sleep(0)
        live: Final = self._live(key)
        total: Final = value if live is None else live[0] + value
        kept_expiry: Final = None if live is None else live[1]
        expires_at: Final = (
            kept_expiry if ttl is None or (kept_expiry is not None and not refresh_ttl) else self.now() + ttl
        )
        self._store(key, total, expires_at)
        return total

    async def async_increment_pipeline(self, increment_list: list[RedisPipelineIncrementOperation]) -> list[float]:
        return [await self.async_increment(op["key"], op["increment_value"], ttl=op["ttl"]) for op in increment_list]


def _worker(redis: _SharedRedisDouble) -> _PROXY_VirtualKeyModelMaxBudgetLimiter:
    return _PROXY_VirtualKeyModelMaxBudgetLimiter(
        dual_cache=DualCache(redis_cache=redis)  # pyright: ignore[reportArgumentType]  # duck-typed Redis double
    )


async def _drain_redis_pushes() -> None:
    await asyncio.gather(*(task for task in asyncio.all_tasks() if task is not asyncio.current_task()))


@pytest.mark.asyncio
async def test_polls_of_a_finished_batch_charge_each_per_model_budget_once():
    limiter: Final = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())
    first: Final = _batch("batch_first", "completed")

    await _poll(limiter, _batch("batch_first", "in_progress"), response_cost=0)
    for _ in range(3):
        await _poll(limiter, first, response_cost=BATCH_COST)

    assert await _spend(limiter, KEY_SPEND_KEY) == pytest.approx(BATCH_COST)
    assert await _spend(limiter, USER_SPEND_KEY) == pytest.approx(BATCH_COST)


@pytest.mark.asyncio
async def test_a_second_batch_and_chat_requests_still_charge_the_budget():
    limiter: Final = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())

    await _poll(limiter, _batch("batch_first", "completed"), response_cost=BATCH_COST)
    await _poll(limiter, _batch("batch_first", "completed"), response_cost=BATCH_COST)
    await _poll(limiter, _batch("batch_second", "completed"), response_cost=BATCH_COST)
    await _chat(limiter)
    await _chat(limiter)

    assert await _spend(limiter, KEY_SPEND_KEY) == pytest.approx(2 * BATCH_COST + 2 * CHAT_COST)


@pytest.mark.asyncio
async def test_two_workers_polling_the_same_finished_batch_at_once_charge_it_once():
    redis: Final = _SharedRedisDouble()
    worker_a: Final = _worker(redis)
    worker_b: Final = _worker(redis)
    finished: Final = _batch("batch_first", "completed")

    await asyncio.gather(_poll(worker_a, finished, BATCH_COST), _poll(worker_b, finished, BATCH_COST))
    await _drain_redis_pushes()

    assert _local_spend(worker_a, KEY_SPEND_KEY) + _local_spend(worker_b, KEY_SPEND_KEY) == pytest.approx(BATCH_COST)
    assert await redis.async_get_cache(KEY_SPEND_KEY) == pytest.approx(BATCH_COST)


@pytest.mark.asyncio
async def test_a_batch_polled_within_every_budget_window_is_never_charged_again():
    clock: Final = _Clock()
    limiter: Final = _worker(_SharedRedisDouble(now=clock.now))
    finished: Final = _batch("batch_first", "completed")

    await _poll(limiter, finished, BATCH_COST)
    clock.advance(12 * 3600)
    await _poll(limiter, finished, BATCH_COST)
    clock.advance(18 * 3600)
    await _poll(limiter, finished, BATCH_COST)

    assert _local_spend(limiter, KEY_SPEND_KEY) == pytest.approx(BATCH_COST)
