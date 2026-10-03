from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Iterator, Sequence
from types import SimpleNamespace
from typing import Final

import pytest
from redis import Redis

import litellm
from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.litellm_core_utils.internal_call_metadata import EvaluationBillingOwner
from litellm.proxy import proxy_server
from litellm.proxy.hooks.model_max_budget_limiter import _PROXY_VirtualKeyModelMaxBudgetLimiter
from litellm.proxy.spend_tracking.budget_reservation import estimate_request_max_cost
from litellm.proxy.spend_tracking.evaluation_budget import (
    _SCRIPT,
    _model_hold_key,
    model_budget_spend,
    release_evaluation_budget,
    reserve_evaluation_budget,
)

MODEL: Final = "openai/evaluation-budget-integration"
REQUEST: Final = {"model": MODEL, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 10}


@pytest.fixture
def budget(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[DualCache, EvaluationBillingOwner, float]]:
    namespace: Final = f"evaluation-budget-{uuid.uuid4().hex}"
    cache: Final = DualCache(
        redis_cache=RedisCache(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), namespace=namespace)
    )
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(proxy_server, "model_max_budget_limiter", _PROXY_VirtualKeyModelMaxBudgetLimiter(cache))
    monkeypatch.setitem(
        litellm.model_cost,
        MODEL,
        {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002, "max_output_tokens": 1000},
    )
    estimate: Final = estimate_request_max_cost(REQUEST, "/chat/completions", None)
    assert estimate is not None and estimate > 0
    owner: Final = EvaluationBillingOwner("creator", {MODEL: {"max_budget": 2 * estimate, "budget_duration": "1d"}})
    try:
        yield cache, owner, estimate
    finally:
        with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as raw:
            keys: Final = tuple(raw.scan_iter(match=f"{namespace}:*"))
            if keys:
                raw.delete(*keys)


@pytest.mark.asyncio
async def test_settlement_before_reserve_reply_does_not_count_a_request_twice(
    budget: tuple[DualCache, EvaluationBillingOwner, float],
) -> None:
    cache, owner, estimate = budget
    earlier: Final = await reserve_evaluation_budget(owner, REQUEST, "acompletion")
    assert earlier is not None and earlier.model is not None
    backend: Final = cache.redis_cache
    assert backend is not None

    def delayed_register(source: str) -> object:
        execute: Final = _SCRIPT.validate_python(backend.async_register_script(source))

        async def execute_then_settle(*, keys: Sequence[str], args: Sequence[str | int | float]) -> object:
            result: Final = await execute(keys=keys, args=args)
            if args and args[0] == "reserve":
                await release_evaluation_budget(earlier, actual_cost=estimate / 4)
            return result

        return execute_then_settle

    cache.redis_cache = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]  # injected transport boundary forwards every script to real Redis
        check_and_fix_namespace=backend.check_and_fix_namespace,
        async_register_script=delayed_register,
    )
    later: Final = await reserve_evaluation_budget(owner, REQUEST, "acompletion")
    assert later is not None
    assert await model_budget_spend(cache, earlier.model.spend_key) == pytest.approx(estimate * 1.25)
    await release_evaluation_budget(later)
    assert await model_budget_spend(cache, earlier.model.spend_key) == pytest.approx(estimate / 4)


@pytest.mark.asyncio
@pytest.mark.parametrize("marker_age", (30, 86401))
async def test_persistent_budget_marker_preserves_the_remaining_spend_window(
    budget: tuple[DualCache, EvaluationBillingOwner, float], marker_age: int
) -> None:
    cache, owner, estimate = budget
    reservation: Final = await reserve_evaluation_budget(owner, REQUEST, "acompletion")
    assert reservation is not None and reservation.model is not None
    backend: Final = cache.redis_cache
    assert backend is not None
    model: Final = reservation.model
    start_key: Final = backend.check_and_fix_namespace(model.start_key)
    spend_key: Final = backend.check_and_fix_namespace(model.spend_key)
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as raw:
        raw.set(start_key, raw.time()[0] - marker_age)
        assert raw.ttl(start_key) == -1
        await release_evaluation_budget(reservation, actual_cost=estimate / 4)
        expected: Final = model.duration - marker_age if marker_age < model.duration else model.duration
        assert expected - 2 <= raw.ttl(spend_key) <= expected
        assert expected - 2 <= raw.ttl(start_key) <= expected
        assert float(raw.get(spend_key)) == pytest.approx(estimate / 4)
        assert raw.zcard(_model_hold_key(spend_key)) == 0


@pytest.mark.asyncio
async def test_cancelled_settlement_drains_the_atomic_redis_write(
    budget: tuple[DualCache, EvaluationBillingOwner, float],
) -> None:
    cache, owner, estimate = budget
    reservation: Final = await reserve_evaluation_budget(owner, REQUEST, "acompletion")
    assert reservation is not None and reservation.model is not None
    backend: Final = cache.redis_cache
    assert backend is not None
    written: Final = asyncio.Event()
    deliver: Final = asyncio.Event()

    def delayed_register(source: str) -> object:
        execute: Final = _SCRIPT.validate_python(backend.async_register_script(source))

        async def execute_then_wait(*, keys: Sequence[str], args: Sequence[str | int | float]) -> object:
            result: Final = await execute(keys=keys, args=args)
            if args and args[0] == "settle":
                written.set()
                await deliver.wait()
            return result

        return execute_then_wait

    cache.redis_cache = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]  # injected transport boundary delays only the real Redis reply
        check_and_fix_namespace=backend.check_and_fix_namespace,
        async_register_script=delayed_register,
    )
    pending: Final = asyncio.create_task(release_evaluation_budget(reservation, actual_cost=estimate / 4))
    await asyncio.wait_for(written.wait(), 5)
    pending.cancel()
    await asyncio.sleep(0)
    pending.cancel()
    assert not pending.done()
    deliver.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await release_evaluation_budget(reservation, actual_cost=estimate / 4)
    assert await model_budget_spend(cache, reservation.model.spend_key) == pytest.approx(estimate / 4)
