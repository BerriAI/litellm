import asyncio
import time
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

import pytest

from litellm.caching.caching import DualCache
from litellm.litellm_core_utils import internal_call_metadata as billing
from litellm.proxy.hooks.model_max_budget_limiter import (
    _PROXY_VirtualKeyModelMaxBudgetLimiter,
)
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.utils import LiteLLMBatch, Usage

KEY_HASH: Final = "key-hash-batch"
USER_ID: Final = "user-batch"
MODEL_GROUP: Final = "batch-qa-primary"
BATCH_COST: Final = 2.925e-05
CHAT_COST: Final = 0.001
KEY_SPEND_KEY: Final = f"virtual_key_spend:{KEY_HASH}:{MODEL_GROUP}:1d"
USER_SPEND_KEY: Final = f"user_model_spend:{USER_ID}:{MODEL_GROUP}:1d"
TEAM_SPEND_KEY: Final = f"team_model_spend:sampled-team:{MODEL_GROUP}:1d"


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


def _event(call_type: str, response_cost: float, team_budget: bool = False) -> dict[str, object]:
    budget: Final = {MODEL_GROUP: {"budget_limit": 0.0001, "time_period": "1d"}}
    return {
        billing.EVALUATION_BILLING_OWNER_KEY: billing.get_evaluation_billing_owner(),
        "call_type": call_type,
        "standard_logging_object": {
            "call_type": call_type,
            "response_cost": response_cost,
            "model": "openai/gpt-5.4-mini",
            "model_group": MODEL_GROUP,
            "metadata": {
                "user_api_key_hash": KEY_HASH, "user_api_key_user_id": USER_ID, "user_api_key_team_id": "sampled-team",
            },
        },
        "litellm_params": {
            "metadata": {
                "user_api_key_model_max_budget": None if team_budget else budget,
                "user_api_key_user_model_max_budget": budget,
                "user_api_key_team_model_max_budget": budget,
            }
        },
    }


async def _poll(limiter: _PROXY_VirtualKeyModelMaxBudgetLimiter, batch: LiteLLMBatch, response_cost: float) -> None:
    await limiter.async_log_success_event(
        _event("aretrieve_batch", response_cost), response_obj=batch, start_time=None, end_time=None
    )


async def _chat(limiter: _PROXY_VirtualKeyModelMaxBudgetLimiter, team_budget: bool = False) -> None:
    await limiter.async_log_success_event(
        _event("acompletion", CHAT_COST, team_budget), response_obj=None, start_time=None, end_time=None
    )


async def _spend(limiter: _PROXY_VirtualKeyModelMaxBudgetLimiter, spend_key: str) -> float:
    return await limiter.dual_cache.async_get_cache(key=spend_key) or 0.0


def _local_spend(limiter: _PROXY_VirtualKeyModelMaxBudgetLimiter, spend_key: str) -> float:
    return limiter.dual_cache.in_memory_cache.get_cache(key=spend_key) or 0.0


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
@pytest.mark.parametrize("evaluation", (False, True))
@pytest.mark.parametrize("team_budget", (False, True))
async def test_a_second_batch_and_chat_requests_still_charge_the_budget(evaluation: bool, team_budget: bool) -> None:
    limiter: Final = _PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache())

    await _poll(limiter, _batch("batch_first", "completed"), response_cost=BATCH_COST)
    await _poll(limiter, _batch("batch_first", "completed"), response_cost=BATCH_COST)
    await _poll(limiter, _batch("batch_second", "completed"), response_cost=BATCH_COST)
    owner: Final = billing.EvaluationBillingOwner("evaluation-admin") if evaluation else None
    with billing.evaluation_billing_context(owner):
        await _chat(limiter, team_budget)
        await _chat(limiter, team_budget)

    chat_cost: Final = 0 if evaluation else 2 * CHAT_COST
    assert await _spend(limiter, KEY_SPEND_KEY) == pytest.approx(2 * BATCH_COST + (0 if team_budget else chat_cost))
    assert await _spend(limiter, USER_SPEND_KEY) == pytest.approx(2 * BATCH_COST + chat_cost)
    assert await _spend(limiter, TEAM_SPEND_KEY) == pytest.approx(chat_cost if team_budget else 0)


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
