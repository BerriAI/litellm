from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from datetime import datetime
from types import SimpleNamespace
from typing import Final

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import TypeAdapter
from redis.crc import key_slot

import litellm
from litellm.caching.caching import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.litellm_core_utils.internal_call_metadata import (
    EVALUATION_BILLING_OWNER_KEY,
    EVALUATION_BUDGET_RESERVATION_KEY,
    EvaluationBillingOwner,
    evaluation_billing_context,
)
from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLM_BudgetTable, Litellm_EntityType, LiteLLM_TagTable, LiteLLM_UserTable
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, tag_cache_key
from litellm.proxy.hooks.model_max_budget_limiter import (
    _PROXY_VirtualKeyModelMaxBudgetLimiter,
    model_budget_spend_cache_key,
    model_budget_start_time_cache_key,
    resolve_model_budget,
)
from litellm.proxy.spend_tracking.budget_reservation import estimate_request_input_cost, estimate_request_max_cost
from litellm.proxy.spend_tracking.evaluation_budget import (
    EvaluationBudgetReservation,
    EvaluationModelReservation,
    _model_hold_key,
    model_budget_spend,
    release_evaluation_budget,
    reserve_evaluation_budget,
)
from litellm.types.router import RetryPolicy

MODEL: Final = "openai/evaluation-budget-fixture"
REQUEST: Final = {"model": MODEL, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 10}


@pytest.mark.parametrize("namespace", ("", "gateway:", "{gateway}:", "gateway{}:", "gateway{:", "{}later{tag}:"))
@pytest.mark.parametrize("model", ("model", "model{}", "model{tag}", "model{", "模型"))
def test_model_holds_share_the_unchanged_numeric_counters_redis_slot(namespace: str, model: str) -> None:
    effective: Final = f"{namespace}user_model_spend:creator:{model}:1d"
    hold_key: Final = _model_hold_key(effective)
    assert hold_key != effective
    assert hold_key.startswith(namespace)
    assert key_slot(hold_key.encode()) == key_slot(effective.encode())


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch) -> DualCache:
    cache: Final = DualCache()
    monkeypatch.setattr(proxy_server, "spend_counter_cache", cache)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", UserApiKeyCache())
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.setattr(proxy_server, "model_max_budget_limiter", _PROXY_VirtualKeyModelMaxBudgetLimiter(cache))
    monkeypatch.setitem(
        litellm.model_cost,
        MODEL,
        {
            "input_cost_per_token": 0.001,
            "output_cost_per_token": 0.002,
            "max_input_tokens": 1000,
            "max_output_tokens": 1000,
            "litellm_provider": "openai",
            "mode": "chat",
        },
    )
    return cache


async def _owner(scope: str, limit: float) -> EvaluationBillingOwner:
    owner: Final = EvaluationBillingOwner(
        "evaluation-admin",
        {MODEL: {"max_budget": limit, "budget_duration": "1d"}} if scope in ("model", "both") else None,
        max_budget=limit if scope in ("total", "both") else None,
    )
    await proxy_server.user_api_key_cache.async_set_cache(
        key=owner.user_id, value=LiteLLM_UserTable(user_id=owner.user_id, max_budget=owner.max_budget, spend=0.0)
    )
    return owner


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("total", "model", "both"))
async def test_concurrent_evaluations_reserve_the_creators_remaining_budget(cache: DualCache, scope: str) -> None:
    estimate: Final = estimate_request_max_cost(REQUEST, "/chat/completions", None)
    assert estimate is not None and estimate > 0
    owner: Final = await _owner(scope, estimate * 1.5)
    attempts: Final = await asyncio.gather(
        reserve_evaluation_budget(owner, REQUEST, "acompletion"),
        reserve_evaluation_budget(owner, REQUEST, "acompletion"),
        return_exceptions=True,
    )
    admitted: Final = tuple(item for item in attempts if isinstance(item, EvaluationBudgetReservation))
    assert len(admitted) == 1
    assert sum(isinstance(item, litellm.BudgetExceededError) for item in attempts) == 1
    reservation: Final = admitted[0]
    assert (await cache.async_get_cache("spend:user:evaluation-admin") or 0.0) == pytest.approx(
        estimate if scope in ("total", "both") else 0.0
    )
    await release_evaluation_budget(reservation)
    retried: Final = await reserve_evaluation_budget(owner, REQUEST, "acompletion")
    assert retried is not None
    await release_evaluation_budget(retried)
    if reservation.model is not None:
        assert (await cache.async_get_cache(reservation.model.spend_key) or 0.0) == pytest.approx(0.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("total", "model"))
async def test_zero_creator_budget_blocks_evaluation(cache: DualCache, scope: str) -> None:
    owner: Final = await _owner(scope, 0.0)
    with pytest.raises(litellm.BudgetExceededError):
        await reserve_evaluation_budget(owner, REQUEST, "acompletion")


@pytest.mark.asyncio
async def test_rejected_model_reservation_cannot_block_a_smaller_request(cache: DualCache) -> None:
    assert await model_budget_spend(cache, "budget", operation="reserve", member="large:2", limit=1) == 2
    assert await model_budget_spend(cache, "budget", operation="reserve", member="small:0.5", limit=1) == 0.5
    assert await model_budget_spend(cache, "budget") == 0.5


@pytest.mark.asyncio
async def test_evaluation_reserves_and_charges_only_creator_without_sampled_tag_budget(
    cache: DualCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner: Final = await _owner("total", 1.0)
    tag: Final = LiteLLM_TagTable(
        tag_name="sampled-tag", spend=0.3, litellm_budget_table=LiteLLM_BudgetTable(max_budget=1.0)
    )
    monkeypatch.setattr(proxy_server, "prisma_client", object())
    await proxy_server.user_api_key_cache.async_set_cache(
        key=tag_cache_key(tag.tag_name), value=tag, model_type=LiteLLM_TagTable
    )
    await cache.async_set_cache("spend:user:evaluation-admin", 0.0)
    await cache.async_set_cache("spend:tag:sampled-tag", tag.spend)
    reservation: Final = await reserve_evaluation_budget(owner, {**REQUEST, "tags": [tag.tag_name]}, "acompletion")
    assert reservation is not None
    estimate: Final = estimate_request_max_cost(REQUEST, "/chat/completions", None)
    assert await cache.async_get_cache("spend:user:evaluation-admin") == pytest.approx(estimate)
    assert await cache.async_get_cache("spend:tag:sampled-tag") == pytest.approx(tag.spend)
    await release_evaluation_budget(reservation, actual_cost=0.01)
    assert await cache.async_get_cache("spend:user:evaluation-admin") == pytest.approx(0.01)
    assert await cache.async_get_cache("spend:tag:sampled-tag") == pytest.approx(tag.spend)


def _receipt(
    owner: EvaluationBillingOwner, reservation: EvaluationBudgetReservation, cost: float
) -> Mapping[str, object]:
    return {
        EVALUATION_BILLING_OWNER_KEY: owner,
        EVALUATION_BUDGET_RESERVATION_KEY: reservation,
        "litellm_params": {"metadata": {"user_api_key_user_id": "sampled-user"}},
        "standard_logging_object": {
            "model": MODEL,
            "response_cost": cost,
            "metadata": {"user_api_key_user_id": "sampled-user"},
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("model", "both"))
async def test_model_callback_settles_once_and_preserves_recovered_failure_spend(cache: DualCache, scope: str) -> None:
    owner: Final = await _owner(scope, 1.0)
    reservation: Final = await reserve_evaluation_budget(owner, REQUEST, "acompletion")
    assert reservation is not None and reservation.model is not None
    limiter: Final = proxy_server.model_max_budget_limiter
    await release_evaluation_budget(reservation)
    receipt: Final = _receipt(owner, reservation, 0.003)
    await limiter.async_log_failure_event(receipt, None, None, None)
    await limiter.async_log_success_event(receipt, None, None, None)
    assert await cache.async_get_cache(reservation.model.spend_key) == pytest.approx(0.003)
    assert (await cache.async_get_cache("spend:user:evaluation-admin") or 0.0) == pytest.approx(
        0.003 if scope == "both" else 0.0
    )


@pytest.mark.asyncio
async def test_failed_evaluation_keeps_incurred_cost_and_frees_unused_reservation(cache: DualCache) -> None:
    owner: Final = await _owner("both", 1.0)
    reservation: Final = await reserve_evaluation_budget(owner, REQUEST, "acompletion")
    assert reservation is not None and reservation.model is not None
    await release_evaluation_budget(reservation, actual_cost=0.005)
    await release_evaluation_budget(reservation, actual_cost=0.005)
    assert await cache.async_get_cache("spend:user:evaluation-admin") == pytest.approx(0.005)
    assert await cache.async_get_cache(reservation.model.spend_key) == pytest.approx(0.005)


@pytest.mark.asyncio
async def test_sdk_blocks_concurrent_evaluation_before_dispatch_and_settles_actual_usage(cache: DualCache) -> None:
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    estimate: Final = estimate_request_max_cost(REQUEST, "/chat/completions", None)
    assert estimate is not None
    owner: Final = await _owner("both", estimate * 1.5)
    litellm.logging_callback_manager.add_litellm_async_success_callback(proxy_server.model_max_budget_limiter)
    entered: Final = asyncio.Event()
    complete: Final = asyncio.Event()
    requests: Final[asyncio.Queue[httpx.Request]] = asyncio.Queue()
    receipts: Final[asyncio.Queue[Mapping[str, object]]] = asyncio.Queue()

    async def upstream(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(request)
        entered.set()
        await complete.wait()
        return httpx.Response(
            200,
            json={
                "id": "evaluation-response",
                "object": "chat.completion",
                "created": 1,
                "model": MODEL,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

    async def capture(kwargs: Mapping[str, object], response: object, start: datetime, end: datetime) -> None:
        receipts.put_nowait(kwargs)

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as http_client:
        client: Final = AsyncOpenAI(api_key="transport-only", http_client=http_client)

        async def completion() -> object:
            return await litellm.acompletion(
                model=MODEL,
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
                client=client,
                api_key="transport-only",
                num_retries=0,
                success_callback=[capture],
            )

        with evaluation_billing_context(owner):
            pending: Final = asyncio.create_task(completion())
            try:
                await asyncio.wait_for(entered.wait(), 30)
                with pytest.raises(litellm.BudgetExceededError):
                    await asyncio.wait_for(completion(), 5)
            finally:
                complete.set()
                await pending
        receipt: Final = await asyncio.wait_for(receipts.get(), 30)
        await GLOBAL_LOGGING_WORKER.flush()
    reservation: Final = receipt[EVALUATION_BUDGET_RESERVATION_KEY]
    assert isinstance(reservation, EvaluationBudgetReservation) and reservation.model is not None
    actual: Final = 10 * 0.001 + 2 * 0.002
    assert receipt.get("response_cost") == pytest.approx(actual)
    assert reservation.model.settled_cost == pytest.approx(actual)
    await proxy_server.increment_spend_counters(
        token=None, team_id=None, user_id=owner.user_id, response_cost=actual, budget_reservation=reservation.total
    )
    assert requests.qsize() == 1
    assert await cache.async_get_cache("spend:user:evaluation-admin") == pytest.approx(actual)
    assert await cache.async_get_cache(reservation.model.spend_key) == pytest.approx(actual)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", (False, True))
async def test_sdk_evaluation_failure_releases_budget_without_unreserved_retries(
    cache: DualCache, cancelled: bool
) -> None:
    owner: Final = await _owner("both", 1.0)
    entered: Final = asyncio.Event()
    blocked: Final = asyncio.Event()
    requests: Final[asyncio.Queue[httpx.Request]] = asyncio.Queue()

    async def upstream(request: httpx.Request) -> httpx.Response:
        requests.put_nowait(request)
        entered.set()
        if cancelled:
            await blocked.wait()
        return httpx.Response(500, json={"error": {"message": "upstream failed", "type": "server_error"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as http_client:
        client: Final = AsyncOpenAI(api_key="transport-only", http_client=http_client, max_retries=0)
        with evaluation_billing_context(owner):
            pending: Final = asyncio.create_task(
                litellm.acompletion(
                    model=MODEL,
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                    client=client,
                    api_key="transport-only",
                    num_retries=0,
                    retry_policy=RetryPolicy(InternalServerErrorRetries=2),
                )
            )
            await asyncio.wait_for(entered.wait(), 30)
            if cancelled:
                pending.cancel()
            with pytest.raises(asyncio.CancelledError if cancelled else litellm.InternalServerError):
                await asyncio.wait_for(pending, 5)
    incurred: Final = estimate_request_input_cost(REQUEST, "/chat/completions", None) if cancelled else 0.0
    assert incurred is not None
    model_key: Final = model_budget_spend_cache_key(Litellm_EntityType.USER, owner.user_id, MODEL, "1d")
    assert requests.qsize() == 1
    assert await cache.async_get_cache("spend:user:evaluation-admin") == pytest.approx(incurred)
    assert (await cache.async_get_cache(model_key) or 0.0) == pytest.approx(incurred)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", (MODEL, "openai/private-evaluation-deployment"))
@pytest.mark.parametrize("routing", ("group", "hidden-alias", "deployment-id", "routing-group"))
async def test_evaluation_uses_configured_prices_for_the_selected_model_group(
    cache: DualCache, monkeypatch: pytest.MonkeyPatch, backend: str, routing: str
) -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "configured-judge",
                "litellm_params": {"model": backend, "api_key": "transport-only"},
                "model_info": {
                    "id": "judge-deployment",
                    "input_cost_per_token": 0.004,
                    "output_cost_per_token": 0.007,
                    "max_input_tokens": 1000,
                    "max_output_tokens": 1000,
                },
            }
        ],
        model_group_alias={"hidden-judge": {"model": "configured-judge", "hidden": True}},
        routing_groups=[
            {"group_name": "evaluation-group", "models": ["configured-judge"], "routing_strategy": "simple-shuffle"}
        ],
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    group: Final = {
        "group": "configured-judge",
        "hidden-alias": "hidden-judge",
        "deployment-id": "judge-deployment",
        "routing-group": "evaluation-group",
    }[routing]
    request: Final = {
        **REQUEST,
        "model": backend,
        "metadata": {"model_group": group},
        "model_info": {"id": "judge-deployment"} if routing in ("deployment-id", "routing-group") else {},
    }
    owner: Final = EvaluationBillingOwner(
        "configured-price-owner", {group: {"max_budget": 1.0, "budget_duration": "1d"}}, max_budget=1.0
    )
    estimate: Final = estimate_request_max_cost({**REQUEST, "model": "configured-judge"}, "/chat/completions", router)
    assert estimate is not None
    reservation: Final = await reserve_evaluation_budget(owner, request, "acompletion")
    assert reservation is not None and reservation.model is not None and reservation.total is not None
    assert reservation.total["reserved_cost"] == pytest.approx(estimate)
    assert reservation.model.reserved_cost == pytest.approx(estimate)
    assert reservation.model.spend_key == model_budget_spend_cache_key(
        Litellm_EntityType.USER, owner.user_id, group, "1d"
    )
    await release_evaluation_budget(reservation)


@pytest.mark.asyncio
@pytest.mark.parametrize("configured_prices", (False, True))
async def test_auto_router_prices_the_selected_leaf_and_preserves_its_logical_budget(
    cache: DualCache, monkeypatch: pytest.MonkeyPatch, configured_prices: bool
) -> None:
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    prices: Final = {"input_cost_per_token": 0.004, "output_cost_per_token": 0.007} if configured_prices else {}
    backend: Final = "openai/private-auto-router-leaf" if configured_prices else MODEL
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "priced-leaf",
                "litellm_params": {"model": backend, "api_key": "transport-only", "max_tokens": 10},
                "model_info": {"id": "selected-leaf", "max_input_tokens": 1000, "max_output_tokens": 10, **prices},
            },
            {
                "model_name": "evaluation-router",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {
                        "classifier_type": "heuristic",
                        "tiers": dict.fromkeys(("SIMPLE", "MEDIUM", "COMPLEX", "REASONING"), "priced-leaf"),
                    },
                },
            },
        ],
        num_retries=0,
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    owner: Final = EvaluationBillingOwner(
        "auto-router-owner", {"evaluation-router": {"max_budget": 1.0, "budget_duration": "1d"}}, max_budget=1.0
    )
    estimate: Final = estimate_request_max_cost({**REQUEST, "model": "priced-leaf"}, "/chat/completions", router)
    assert estimate is not None and estimate > 0
    model_key: Final = model_budget_spend_cache_key(Litellm_EntityType.USER, owner.user_id, "evaluation-router", "1d")
    receipts: Final[asyncio.Queue[Mapping[str, object]]] = asyncio.Queue()
    litellm.logging_callback_manager.add_litellm_async_success_callback(proxy_server.model_max_budget_limiter)

    async def upstream(request: httpx.Request) -> httpx.Response:
        payload: Final = TypeAdapter(Mapping[str, object]).validate_json(request.content)
        assert payload.get("model") == backend.removeprefix("openai/")
        assert payload.get("max_tokens") == 10
        assert await cache.async_get_cache(f"spend:user:{owner.user_id}") == pytest.approx(estimate)
        assert await model_budget_spend(cache, model_key) == pytest.approx(estimate)
        return httpx.Response(
            200,
            json={
                "id": "auto-router-evaluation-response",
                "object": "chat.completion",
                "created": 1,
                "model": backend,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

    async def capture(kwargs: Mapping[str, object], response: object, start: datetime, end: datetime) -> None:
        receipts.put_nowait(kwargs)

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as http_client:
        client: Final = AsyncOpenAI(api_key="transport-only", http_client=http_client)
        with evaluation_billing_context(owner):
            await router.acompletion(
                model="evaluation-router",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
                client=client,
                fallbacks=[],
                success_callback=[capture],
            )
        receipt: Final = await asyncio.wait_for(receipts.get(), 30)
        await GLOBAL_LOGGING_WORKER.flush()
    reservation: Final = receipt[EVALUATION_BUDGET_RESERVATION_KEY]
    assert isinstance(reservation, EvaluationBudgetReservation) and reservation.model is not None
    assert reservation.model.spend_key == model_key
    assert reservation.model.reserved_cost == pytest.approx(estimate)
    actual: Final = 10 * prices.get("input_cost_per_token", 0.001) + 2 * prices.get("output_cost_per_token", 0.002)
    assert await cache.async_get_cache(model_key) == pytest.approx(actual)
    await proxy_server.increment_spend_counters(
        token=None, team_id=None, user_id=owner.user_id, response_cost=actual, budget_reservation=reservation.total
    )
    assert await cache.async_get_cache(f"spend:user:{owner.user_id}") == pytest.approx(actual)


@pytest.mark.asyncio
@pytest.mark.parametrize("window_change", ("missing-start", "replaced-start", "rollover"))
async def test_model_settlement_never_refunds_another_window_or_request(cache: DualCache, window_change: str) -> None:
    owner: Final = await _owner("model", 1.0)
    spend_key: Final = model_budget_spend_cache_key(Litellm_EntityType.USER, owner.user_id, MODEL, "1d")
    start_key: Final = model_budget_start_time_cache_key(Litellm_EntityType.USER, owner.user_id, MODEL, "1d")
    await cache.async_set_cache(spend_key, 0.2)
    await cache.async_set_cache(start_key, time.time())
    earlier: Final = await reserve_evaluation_budget(owner, REQUEST, "acompletion")
    assert earlier is not None and earlier.model is not None
    cache.in_memory_cache.delete_cache(start_key)
    if window_change == "replaced-start":
        await cache.async_set_cache(start_key, time.time() + 1)
    if window_change == "rollover":
        cache.in_memory_cache.delete_cache(spend_key)
        await cache.async_set_cache(spend_key, 0.3)
    later: Final = await reserve_evaluation_budget(owner, REQUEST, "acompletion")
    assert later is not None and later.model is not None
    await release_evaluation_budget(earlier, actual_cost=0.01)
    assert await model_budget_spend(cache, spend_key) == pytest.approx(
        (0.3 if window_change == "rollover" else 0.2) + 0.01 + later.model.reserved_cost
    )
    assert await cache.async_get_cache(spend_key) == pytest.approx((0.3 if window_change == "rollover" else 0.2) + 0.01)
    await release_evaluation_budget(later, actual_cost=0.02)
    assert await model_budget_spend(cache, spend_key) == pytest.approx(
        (0.3 if window_change == "rollover" else 0.2) + 0.03
    )
    assert await cache.async_get_cache(spend_key) == pytest.approx((0.3 if window_change == "rollover" else 0.2) + 0.03)


class _Clock:
    seconds: float = 0.0

    def now(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


@pytest.mark.asyncio
async def test_expired_model_hold_cannot_release_or_renew_a_new_requests_budget() -> None:
    clock: Final = _Clock()
    cache: Final = DualCache(in_memory_cache=InMemoryCache(clock=clock.now))
    await model_budget_spend(cache, "budget", operation="reserve", member="old:0.2")
    clock.advance(30)
    assert await model_budget_spend(cache, "budget", operation="reserve", member="new:0.3") == pytest.approx(0.5)
    clock.advance(31)
    assert await model_budget_spend(cache, "budget") == pytest.approx(0.3)
    assert await model_budget_spend(cache, "budget", operation="settle", member="old:0.2") == pytest.approx(0.3)
    with pytest.raises(RuntimeError, match="Evaluation budget reservation expired"):
        await model_budget_spend(cache, "budget", operation="renew", member="old:0.2")
    assert await model_budget_spend(cache, "budget") == pytest.approx(0.3)
    await model_budget_spend(cache, "budget", operation="renew", member="new:0.3")
    clock.advance(31)
    assert await model_budget_spend(cache, "budget") == pytest.approx(0.3)
    await model_budget_spend(cache, "budget", operation="settle", member="new:0.3")
    assert await model_budget_spend(cache, "budget") == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_ordinary_user_model_gate_includes_outstanding_evaluations(cache: DualCache) -> None:
    estimate: Final = estimate_request_max_cost(REQUEST, "/chat/completions", None)
    assert estimate is not None
    owner: Final = await _owner("model", estimate)
    reservation: Final = await reserve_evaluation_budget(owner, REQUEST, "acompletion")
    assert reservation is not None
    resolved: Final = resolve_model_budget(MODEL, owner.user_model_max_budget or {})
    assert resolved is not None
    limiter: Final = proxy_server.model_max_budget_limiter
    with pytest.raises(litellm.BudgetExceededError):
        await limiter.is_user_within_model_budget(owner.user_id, owner.user_model_max_budget or {}, MODEL)
    await release_evaluation_budget(reservation)
    assert await limiter.is_user_within_model_budget(owner.user_id, owner.user_model_max_budget or {}, MODEL)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("total-acquire", "total-settle"))
async def test_repeated_cancellation_drains_budget_writes_before_returning(cache: DualCache, phase: str) -> None:
    owner: Final = await _owner("both", 1.0)
    model_key: Final = model_budget_spend_cache_key(Litellm_EntityType.USER, owner.user_id, MODEL, "1d")
    total_key: Final = f"spend:user:{owner.user_id}"
    reservation: Final = (
        await reserve_evaluation_budget(owner, REQUEST, "acompletion") if phase == "total-settle" else None
    )
    backend: Final = cache.in_memory_cache
    entered: Final = asyncio.Event()
    proceed: Final = asyncio.Event()
    pause_key: Final = total_key

    async def write_then_wait(key: str, value: float, **kwargs: object) -> float:
        result: Final = await backend.async_increment(key, value, **kwargs)
        if key == pause_key and value != 0 and not entered.is_set():
            entered.set()
            await proceed.wait()
        return result

    cache.in_memory_cache = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]  # injected storage boundary delegates every operation to the real cache
        get_cache=backend.get_cache,
        set_cache=backend.set_cache,
        delete_cache=backend.delete_cache,
        async_get_cache=backend.async_get_cache,
        async_set_cache=backend.async_set_cache,
        async_increment=write_then_wait,
        increment_cache=backend.increment_cache,
        _clock=backend._clock,
    )
    pending: Final = asyncio.create_task(
        reserve_evaluation_budget(owner, REQUEST, "acompletion")
        if reservation is None
        else release_evaluation_budget(reservation, actual_cost=0.005)
    )
    await asyncio.wait_for(entered.wait(), 5)
    pending.cancel()
    await asyncio.sleep(0)
    pending.cancel()
    assert not pending.done()
    proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    expected: Final = 0.0 if reservation is None else 0.005
    assert (await cache.async_get_cache(total_key) or 0.0) == pytest.approx(expected)
    assert (await cache.async_get_cache(model_key) or 0.0) == pytest.approx(expected)
    assert await model_budget_spend(cache, model_key) == pytest.approx(expected)
    await release_evaluation_budget(reservation, actual_cost=expected)
    assert (await cache.async_get_cache(model_key) or 0.0) == pytest.approx(expected)


@pytest.mark.asyncio
async def test_evaluation_and_ordinary_requests_share_epoch_budget_window_markers(cache: DualCache) -> None:
    clock: Final = _Clock()
    cache.in_memory_cache = InMemoryCache(clock=clock.now)
    owner: Final = await _owner("model", 1.0)
    reservation: Final = await reserve_evaluation_budget(owner, REQUEST, "acompletion")
    assert reservation is not None and reservation.model is not None
    await release_evaluation_budget(reservation, actual_cost=0.02)
    ordinary: Final = {
        "litellm_params": {"metadata": {"user_api_key_user_model_max_budget": owner.user_model_max_budget}},
        "standard_logging_object": {
            "model": MODEL,
            "response_cost": 0.01,
            "metadata": {"user_api_key_user_id": owner.user_id},
        },
    }
    await proxy_server.model_max_budget_limiter.async_log_success_event(ordinary, None, None, None)
    await release_evaluation_budget(reservation, actual_cost=0.03)
    await proxy_server.model_max_budget_limiter.async_log_success_event(ordinary, None, None, None)
    assert await cache.async_get_cache(reservation.model.spend_key) == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_total_storage_failure_does_not_strand_model_budget(
    cache: DualCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner: Final = await _owner("both", 1.0)
    reservation: Final = await reserve_evaluation_budget(owner, REQUEST, "acompletion")
    assert reservation is not None and reservation.model is not None and reservation.total is not None

    def unavailable(key: str) -> object:
        raise OSError("total spend storage unavailable")

    failed_cache: Final = DualCache(
        in_memory_cache=SimpleNamespace(get_cache=unavailable)  # pyright: ignore[reportArgumentType]  # injected failing storage boundary
    )
    monkeypatch.setattr(proxy_server, "spend_counter_cache", failed_cache)
    with pytest.raises(OSError, match="total spend storage unavailable"):
        await release_evaluation_budget(reservation, actual_cost=0.005)
    assert reservation.total["finalized"] is False
    assert await model_budget_spend(cache, reservation.model.spend_key) == pytest.approx(0.005)
    assert await cache.async_get_cache(reservation.model.spend_key) == pytest.approx(0.005)
    monkeypatch.setattr(proxy_server, "spend_counter_cache", cache)
    await release_evaluation_budget(reservation, actual_cost=0.0)
    assert reservation.total["finalized"] is True
    assert await cache.async_get_cache(f"spend:user:{owner.user_id}") == pytest.approx(0.005)
    assert await cache.async_get_cache(reservation.model.spend_key) == pytest.approx(0.005)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ("queued-receipt", "lost-lease"))
async def test_model_lease_survives_queued_logging_and_cancels_an_unreserved_call(
    cache: DualCache, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    clock: Final = _Clock()
    cache.in_memory_cache = InMemoryCache(clock=clock.now)
    model: Final = EvaluationModelReservation(cache, "model-spend", "model-start", 86400, 0.2)
    await model_budget_spend(cache, model.spend_key, operation="reserve", member=model.member)
    dispatched: Final = asyncio.Event()
    request: Final = asyncio.create_task(dispatched.wait())
    if case == "queued-receipt":
        dispatched.set()
        await request
    intervals: Final[asyncio.Queue[asyncio.Event]] = asyncio.Queue()
    sleep: Final = asyncio.sleep

    async def tick(delay: float) -> None:
        if asyncio.current_task() is not renewal:
            await sleep(delay)
            return
        interval: Final = asyncio.Event()
        intervals.put_nowait(interval)
        await interval.wait()

    with monkeypatch.context() as timers:
        timers.setattr(asyncio, "sleep", tick)
        renewal: Final = asyncio.create_task(model.renew(request))
        first: Final = await asyncio.wait_for(intervals.get(), 5)
        clock.advance(61 if case == "lost-lease" else 31)
        first.set()
        if case == "lost-lease":
            await asyncio.wait_for(renewal, 5)
            with pytest.raises(asyncio.CancelledError):
                await request
            assert await model_budget_spend(cache, model.spend_key) == pytest.approx(0.0)
        else:
            second: Final = await asyncio.wait_for(intervals.get(), 5)
            clock.advance(31)
            assert await model_budget_spend(cache, model.spend_key) == pytest.approx(0.2)
            await model.settle(0.0)
            second.set()
            await asyncio.wait_for(renewal, 5)
            assert await model_budget_spend(cache, model.spend_key) == pytest.approx(0.0)
