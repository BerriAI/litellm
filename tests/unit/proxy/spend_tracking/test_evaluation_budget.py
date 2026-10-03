import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final
from unittest.mock import patch

import pytest
import respx

import litellm
from litellm.caching.caching import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.litellm_core_utils.internal_call_metadata import EvaluationBillingOwner, evaluation_billing_context
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.proxy import proxy_server
from litellm.proxy.spend_tracking.budget_reservation import estimate_request_max_cost
from litellm.proxy.spend_tracking.evaluation_budget import (
    EvaluationAttempt,
    _complete,
    model_budget_spend,
    reserve_evaluation_budget,
)
from litellm.types.utils import ModelResponse, Usage


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("total", "model", "both"))
async def test_concurrent_evaluations_settle_once_without_spending_another_attempt(
    evaluation_spend_cache: DualCache, monkeypatch: pytest.MonkeyPatch, scope: str
) -> None:
    monkeypatch.setattr(evaluation_spend_cache, "in_memory_cache", InMemoryCache(clock=lambda: 0.0))
    model: Final = "openai/evaluation-budget-test"
    monkeypatch.setitem(litellm.model_cost, model, {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002})
    request: Final = {"model": model, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 10}
    estimate: Final = estimate_request_max_cost(request, "/chat/completions", None)
    assert estimate is not None and estimate > 0
    owner: Final = EvaluationBillingOwner(
        "creator",
        {model: {"max_budget": estimate * 1.5, "budget_duration": "1d"}} if scope != "total" else None,
        max_budget=estimate * 1.5 if scope != "model" else None,
    )
    attempts: Final = await asyncio.gather(
        *(reserve_evaluation_budget(owner, request, "acompletion") for _ in range(2)), return_exceptions=True
    )
    assert sum(isinstance(result, litellm.BudgetExceededError) for result in attempts) == 1
    admitted: Final = next(result for result in attempts if not isinstance(result, BaseException))
    assert admitted is not None
    await admitted.settle(estimate / 4)
    later: Final = await reserve_evaluation_budget(owner, request, "acompletion")
    assert later is not None
    for cost in (estimate / 4, 0, estimate / 2):
        await admitted.settle(cost)
        await proxy_server.increment_spend_counters(
            token=None, team_id=None, user_id="creator", response_cost=cost, budget_reservation=admitted.total
        )
    await later.settle(0)
    await proxy_server.model_max_budget_limiter.async_log_success_event(
        {
            "litellm_params": {"metadata": {"user_api_key_user_model_max_budget": owner.user_model_max_budget}},
            "standard_logging_object": {
                "model": model,
                "response_cost": 0.01,
                "metadata": {"user_api_key_user_id": owner.user_id},
            },
        },
        None,
        None,
        None,
    )
    keys: Final = {"total": "spend:user:creator", "model": f"user_model_spend:creator:{model}:1d"}
    for kind in keys if scope == "both" else (scope,):
        assert await evaluation_spend_cache.async_get_cache(keys[kind]) == pytest.approx(
            estimate / 2 + (0.01 if kind == "model" else 0)
        )


@pytest.mark.asyncio
async def test_auto_router_prices_its_selected_deployment_and_bills_the_creator(
    evaluation_spend_cache: DualCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    rates: Final = {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002}
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "leaf",
                "litellm_params": {"model": "openai/private-evaluation", "api_key": "test", **rates},
                "model_info": {"id": "selected", "max_input_tokens": 1000, "max_output_tokens": 10},
            },
            {
                "model_name": "router",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {
                        "classifier_type": "heuristic",
                        "tiers": dict.fromkeys(("SIMPLE", "MEDIUM", "COMPLEX", "REASONING"), "leaf"),
                    },
                },
            },
        ],
        num_retries=0,
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    owner: Final = EvaluationBillingOwner("creator", {"router": {"max_budget": 1, "budget_duration": "1d"}}, 1)
    receipts: Final[asyncio.Queue[Mapping[str, object]]] = asyncio.Queue()
    source: Final = {"user_api_key_user_id": "source", "user_api_key_team_id": "team", "agent_id": "agent"}

    async def capture(kwargs: Mapping[str, object], response: object, start: datetime, end: datetime) -> None:
        receipts.put_nowait(kwargs)

    with respx.mock(assert_all_called=True) as transport:
        transport.post("https://api.openai.com/v1/chat/completions").respond(
            200,
            json=ModelResponse(
                model="private-evaluation",
                choices=[{"message": {"role": "assistant", "content": "ok"}}],
                usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            ).model_dump(),
        )

        with evaluation_billing_context(owner):
            await router.acompletion(
                model="router",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
                metadata=source,
                fallbacks=[],
                success_callback=[capture],
            )
        receipt: Final = await asyncio.wait_for(receipts.get(), 10)
        await GLOBAL_LOGGING_WORKER.flush()
    assert (receipt["user"], receipt["agent_id"], receipt["request_tags"]) == ("creator", None, [])
    assert source["user_api_key_user_id"] == "source" and source["agent_id"] == "agent"
    actual: Final = 10 * rates["input_cost_per_token"] + 2 * rates["output_cost_per_token"]
    assert receipt["response_cost"] == pytest.approx(actual)
    for key in ("spend:user:creator", "user_model_spend:creator:router:1d"):
        assert await evaluation_spend_cache.async_get_cache(key) == pytest.approx(actual)


@dataclass(slots=True)
class _Clock:
    seconds: float = 0.0


@pytest.mark.asyncio
async def test_expired_model_hold_cannot_release_or_renew_another_requests_budget() -> None:
    clock: Final = _Clock()
    cache: Final = DualCache(in_memory_cache=InMemoryCache(clock=lambda: clock.seconds))
    await model_budget_spend(cache, "budget", operation="reserve", member="old:0.2")
    clock.seconds = 30
    assert await model_budget_spend(cache, "budget", operation="reserve", member="new:0.3") == pytest.approx(0.5)
    clock.seconds = 61
    assert await model_budget_spend(cache, "budget") == pytest.approx(0.3)
    assert await model_budget_spend(cache, "budget", operation="settle", member="old:0.2") == pytest.approx(0.3)
    with pytest.raises(RuntimeError, match="Evaluation reservation expired"):
        await model_budget_spend(cache, "budget", operation="renew", member="old:0.2")
    await model_budget_spend(cache, "budget", operation="renew", member="new:0.3")
    clock.seconds = 92
    assert await model_budget_spend(cache, "budget") == pytest.approx(0.3)
    assert await model_budget_spend(cache, "budget", operation="settle", member="new:0.3") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("admission", "settlement"))
async def test_repeated_cancellation_drains_budget_operations_before_returning(
    evaluation_spend_cache: DualCache, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    model: Final = "openai/cancelled-evaluation"
    monkeypatch.setitem(litellm.model_cost, model, {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002})
    request: Final = {"model": model, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 10}
    owner: Final = EvaluationBillingOwner("creator", {model: {"max_budget": 1, "budget_duration": "1d"}}, 1)
    existing: Final = await reserve_evaluation_budget(owner, request, "acompletion") if phase == "settlement" else None
    entered: Final = asyncio.Event()
    proceed: Final = asyncio.Event()

    async def delayed_write() -> EvaluationAttempt | None:
        attempt: Final = existing or await reserve_evaluation_budget(owner, request, "acompletion")
        assert attempt is not None
        entered.set()
        await proceed.wait()
        if existing is None:
            return attempt
        await attempt.settle(0.005)
        return None

    pending: Final = asyncio.create_task(_complete(delayed_write()))
    await asyncio.wait_for(entered.wait(), 5)
    pending.cancel()
    await asyncio.sleep(0)
    pending.cancel()
    assert not pending.done()
    proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    expected: Final = 0.005 if phase == "settlement" else 0
    assert await evaluation_spend_cache.async_get_cache("spend:user:creator") == pytest.approx(expected)
    assert await model_budget_spend(evaluation_spend_cache, f"user_model_spend:creator:{model}:1d") == pytest.approx(
        expected
    )


@pytest.mark.asyncio
async def test_evaluation_settlement_preserves_spend_until_the_shared_budget_window_ends(
    evaluation_spend_cache: DualCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock: Final = _Clock()
    epoch: Final = 1000.0
    monkeypatch.setattr(time, "time", lambda: epoch + 0.2 + clock.seconds)
    monkeypatch.setattr(evaluation_spend_cache, "in_memory_cache", InMemoryCache(clock=lambda: clock.seconds))
    model: Final = "openai/shared-window"
    monkeypatch.setitem(litellm.model_cost, model, {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002})
    owner: Final = EvaluationBillingOwner("creator", {model: {"max_budget": 1, "budget_duration": "10s"}})
    key: Final = f"user_model_spend:creator:{model}:10s"
    evaluation_spend_cache.in_memory_cache.set_cache(key, 0.1, ttl=1)
    evaluation_spend_cache.in_memory_cache.set_cache(f"user_model_budget_start_time:creator:{model}:10s", epoch, ttl=10)
    attempt: Final = await reserve_evaluation_budget(
        owner, {"model": model, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 10}, "acompletion"
    )
    assert attempt is not None
    await attempt.settle(0.02)
    with patch("litellm.router_strategy.budget_limiter.datetime") as wall_clock:
        wall_clock.now.return_value = datetime.fromtimestamp(epoch + 0.2, timezone.utc)
        await proxy_server.model_max_budget_limiter.async_log_success_event(
            {
                "litellm_params": {"metadata": {"user_api_key_user_model_max_budget": owner.user_model_max_budget}},
                "standard_logging_object": {
                    "model": model,
                    "response_cost": 0.01,
                    "metadata": {"user_api_key_user_id": owner.user_id},
                },
            },
            None,
            None,
            None,
        )
    clock.seconds = 2
    assert await model_budget_spend(evaluation_spend_cache, key) == pytest.approx(0.13)
    clock.seconds = 9.9
    assert await model_budget_spend(evaluation_spend_cache, key) == pytest.approx(0.13)
    clock.seconds = 10.1
    assert await model_budget_spend(evaluation_spend_cache, key) == 0
