from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Final, Literal

import pytest

import litellm
from litellm.caching import DualCache
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.proxy import proxy_server
from litellm.proxy._types import (
    LiteLLM_OrganizationTable,
    LiteLLM_ProjectTableCachedObj,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    project_cache_key,
    team_membership_reservation_cache_key,
)
from litellm.proxy.spend_tracking.budget_reservation import (
    _get_team_member_budget_counter,
    estimate_request_max_cost,
    get_budget_window_start,
    release_unbound_budget_reservation,
    reserve_budget_for_request,
)
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

TOKEN_COUNTING_ROUTES: Final = (
    "/responses/input_tokens",
    "/v1/responses/input_tokens",
    "/openai/v1/responses/input_tokens",
    "/utils/token_counter",
    "/v1/messages/count_tokens",
    "/v1beta/models/gemini-3.8-flash:countTokens",
    "/models/gemini-3.8-flash:countTokens",
    "/bedrock/v1/messages/count-tokens",
    "/bedrock/model/us.anthropic.claude-sonnet-4-6/count-tokens",
    "/vertex_ai/v1/projects/p/locations/us-east5/publishers/anthropic/models/count-tokens:rawPredict",
    "/vertex-ai/v1/projects/p/locations/us-east5/publishers/anthropic/models/count-tokens:rawPredict",
)


def _budgeted_token() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test", token="hashed-token", max_budget=100.0, spend=0.0)


async def _reserve(route: str) -> dict | None:
    return await reserve_budget_for_request(
        request_body={"model": "gpt-4o", "input": "hello"},
        route=route,
        llm_router=None,
        valid_token=_budgeted_token(),
        team_object=None,
        user_object=None,
        prisma_client=None,
        user_api_key_cache=UserApiKeyCache(),
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("route", TOKEN_COUNTING_ROUTES)
async def test_token_counting_routes_are_exempt_from_budget_reservation(route):
    assert await _reserve(route) is None


@pytest.mark.asyncio
async def test_non_exempt_llm_route_still_reserves_budget():
    reservation: Final = await _reserve("/v1/responses")

    assert reservation is not None
    assert reservation["reserved_cost"] > 0


@pytest.mark.asyncio
async def test_reservation_carries_the_admission_input_token_count():
    reservation: Final = await _reserve("/v1/responses")
    expected: Final = litellm.token_counter(model="gpt-4o", text="hello")

    assert reservation is not None
    assert expected > 0
    assert reservation["input_tokens"] == expected


ANTHROPIC_MESSAGES: Final = [{"role": "user", "content": "hello!!!"}]
COUNT_TOKENS_REQUESTS: Final[tuple[tuple[str, dict[str, object]], ...]] = (
    ("/v1/messages/count_tokens", {"model": "claude-sonnet-5", "messages": ANTHROPIC_MESSAGES}),
    ("/v1beta/models/gemini-3.8-flash:countTokens", {"contents": [{"role": "user", "parts": [{"text": "hello!!!"}]}]}),
    (
        "/vertex_ai/v1/projects/p/locations/us-east5/publishers/anthropic/models/count-tokens:rawPredict",
        {"model": "claude-sonnet-5", "messages": ANTHROPIC_MESSAGES},
    ),
    ("/bedrock/v1/messages/count-tokens", {"model": "claude-sonnet-5", "messages": ANTHROPIC_MESSAGES}),
)
TINY_BUDGET_KEY_TOKEN: Final = "hashed-count-tokens-key"


@pytest.fixture
def spend_counter_cache(monkeypatch: pytest.MonkeyPatch) -> DualCache:
    cache: Final = DualCache()
    monkeypatch.setattr(proxy_server, "spend_counter_cache", cache)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    return cache


async def _reserve_for_tiny_budget_key(route: str, request_body: dict[str, object]) -> dict[str, object] | None:
    return await reserve_budget_for_request(
        request_body=request_body,
        route=route,
        llm_router=None,
        valid_token=UserAPIKeyAuth(token=TINY_BUDGET_KEY_TOKEN, max_budget=0.01, spend=0.0),
        team_object=None,
        user_object=None,
        prisma_client=None,
        user_api_key_cache=UserApiKeyCache(),
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("route", "request_body"), COUNT_TOKENS_REQUESTS)
async def test_repeated_token_counting_never_touches_a_tiny_budget(
    spend_counter_cache: DualCache, route: str, request_body: dict[str, object]
):
    counter_key: Final = f"spend:key:{TINY_BUDGET_KEY_TOKEN}"

    assert await _reserve_for_tiny_budget_key(route, request_body) is None
    assert await _reserve_for_tiny_budget_key(route, request_body) is None
    assert spend_counter_cache.in_memory_cache.get_cache(key=counter_key) is None

    completion: Final = await _reserve_for_tiny_budget_key(
        "/v1/messages", {"model": "claude-sonnet-5", "max_tokens": 16, "messages": ANTHROPIC_MESSAGES}
    )
    assert completion is not None
    reserved_cost: Final = completion["reserved_cost"]
    assert isinstance(reserved_cost, float)
    assert reserved_cost > 0
    assert spend_counter_cache.in_memory_cache.get_cache(key=counter_key) == pytest.approx(reserved_cost)


BEDROCK_SONNET: Final = "us.anthropic.claude-sonnet-4-6"
CONVERSE_BODY: Final = {
    "messages": [{"role": "user", "content": [{"text": "Reply with one word: pong"}]}],
    "inferenceConfig": {"maxTokens": 5},
}
INVOKE_BODY: Final = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 5,
    "messages": [{"role": "user", "content": "Reply with one word: pong"}],
}


def test_bedrock_converse_body_reserves_the_prompt_not_the_context_window():
    converse_cost: Final = estimate_request_max_cost(
        request_body=CONVERSE_BODY,
        route=f"/bedrock/model/{BEDROCK_SONNET}/converse",
        llm_router=None,
        input_token_counts={},
    )
    invoke_cost: Final = estimate_request_max_cost(
        request_body=INVOKE_BODY,
        route=f"/bedrock/model/{BEDROCK_SONNET}/invoke",
        llm_router=None,
        input_token_counts={},
    )
    assert converse_cost is not None and invoke_cost is not None
    assert invoke_cost < converse_cost < 2 * invoke_cost


def _tiered_deployment(input_cost_per_token: float) -> Deployment:
    return Deployment(
        model_name="tiered-group",
        litellm_params=LiteLLM_Params(model="dashscope/qwen3-max", api_key="sk-fake"),
        model_info=ModelInfo(
            id="tiered-deployment",
            max_output_tokens=1000,
            tiered_pricing=[
                {
                    "input_cost_per_token": input_cost_per_token,
                    "output_cost_per_token": input_cost_per_token,
                    "range": [0, 128000],
                }
            ],
        ),
    )


TIERED_BODY: Final = {"model": "tiered-group", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 10}


def test_repeated_estimates_reuse_cached_model_cost_info() -> None:
    router: Final = Router(model_list=[_tiered_deployment(1e-06).model_dump()])
    first: Final = estimate_request_max_cost(request_body=TIERED_BODY, route="/chat/completions", llm_router=router)
    hits_before: Final = router.cached_deployment_model_info.cache_info().hits

    second: Final = estimate_request_max_cost(request_body=TIERED_BODY, route="/chat/completions", llm_router=router)

    assert second == first
    assert router.cached_deployment_model_info.cache_info().hits == hits_before + 1


def test_deployment_pricing_update_invalidates_cached_estimate() -> None:
    router: Final = Router(model_list=[_tiered_deployment(1e-06).model_dump()])
    before: Final = estimate_request_max_cost(request_body=TIERED_BODY, route="/chat/completions", llm_router=router)
    assert before is not None

    router.upsert_deployment(_tiered_deployment(1e-03))

    after: Final = estimate_request_max_cost(request_body=TIERED_BODY, route="/chat/completions", llm_router=router)
    assert after is not None
    assert math.isclose(after, before * 1000)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expiry_offset, expected_max_budget",
    [
        (timedelta(days=1), 3.0),
        (timedelta(days=-1), 2.0),
    ],
)
async def test_team_member_reservation_counter_honours_temp_budget_increase(
    expiry_offset: timedelta, expected_max_budget: float
) -> None:
    user_id: Final = "member-temp"
    team_id: Final = "team-temp"
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(
        key=team_membership_reservation_cache_key(user_id=user_id, team_id=team_id),
        value=LiteLLM_TeamMembership(
            user_id=user_id,
            team_id=team_id,
            spend=0.5,
            budget_id="budget-temp",
            litellm_budget_table=LiteLLM_BudgetTable(
                max_budget=2.0,
                temp_budget_increase=1.0,
                temp_budget_expiry=datetime.now(timezone.utc) + expiry_offset,
            ),
        ),
    )

    counter: Final = await _get_team_member_budget_counter(
        valid_token=UserAPIKeyAuth(token="hashed", user_id=user_id, team_id=team_id),
        team_object=LiteLLM_TeamTable(team_id=team_id),
        user_object=LiteLLM_UserTable(user_id=user_id),
        user_api_key_cache=cache,
    )

    assert counter is not None
    assert counter.max_budget == expected_max_budget
    assert counter.fallback_spend == 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "default_cap, expiry_offset, expected_max_budget",
    [
        (2.0, timedelta(days=1), 3.0),
        (2.0, timedelta(days=-1), 2.0),
        (0.0, timedelta(days=1), None),
    ],
)
async def test_team_member_reservation_counter_adds_temp_increase_to_live_team_default(
    default_cap: float, expiry_offset: timedelta, expected_max_budget: float | None
) -> None:
    user_id: Final = "member-bare"
    team_id: Final = "team-bare"
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(
        key="team_member_default_budget:default-bare",
        value=LiteLLM_BudgetTable(budget_id="default-bare", max_budget=default_cap),
    )
    await cache.async_set_cache(
        key=team_membership_reservation_cache_key(user_id=user_id, team_id=team_id),
        value=LiteLLM_TeamMembership(
            user_id=user_id,
            team_id=team_id,
            spend=0.5,
            budget_id="budget-bare",
            litellm_budget_table=LiteLLM_BudgetTable(
                max_budget=None,
                temp_budget_increase=1.0,
                temp_budget_expiry=datetime.now(timezone.utc) + expiry_offset,
            ),
        ),
    )

    counter: Final = await _get_team_member_budget_counter(
        valid_token=UserAPIKeyAuth(token="hashed", user_id=user_id, team_id=team_id),
        team_object=LiteLLM_TeamTable(team_id=team_id, metadata={"team_member_budget_id": "default-bare"}),
        user_object=LiteLLM_UserTable(user_id=user_id),
        user_api_key_cache=cache,
    )

    if expected_max_budget is None:
        assert counter is None
        return
    assert counter is not None
    assert counter.max_budget == expected_max_budget
    assert counter.fallback_spend == 0.5


@pytest.mark.asyncio
async def test_reservation_starts_unbound_to_any_callback():
    reservation: Final = await _reserve("/v1/responses")

    assert reservation is not None
    assert reservation["callback_bound"] is False


@pytest.mark.asyncio
async def test_release_unbound_budget_reservation_frees_the_counter(spend_counter_cache: DualCache):
    counter_key: Final = f"spend:key:{TINY_BUDGET_KEY_TOKEN}"
    reservation: Final = await _reserve_for_tiny_budget_key(
        "/v1/chat/completions", {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
    )
    assert reservation is not None
    assert spend_counter_cache.in_memory_cache.get_cache(key=counter_key) == pytest.approx(reservation["reserved_cost"])

    await release_unbound_budget_reservation(reservation)

    assert spend_counter_cache.in_memory_cache.get_cache(key=counter_key) == pytest.approx(0.0)
    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_release_unbound_budget_reservation_leaves_a_bound_one_to_its_callback(spend_counter_cache: DualCache):
    counter_key: Final = f"spend:key:{TINY_BUDGET_KEY_TOKEN}"
    reservation: Final = await _reserve_for_tiny_budget_key(
        "/v1/chat/completions", {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
    )
    assert reservation is not None
    reservation["callback_bound"] = True

    await release_unbound_budget_reservation(reservation)

    assert spend_counter_cache.in_memory_cache.get_cache(key=counter_key) == pytest.approx(reservation["reserved_cost"])
    assert reservation["finalized"] is False


@pytest.mark.asyncio
async def test_reservation_holds_cost_against_cached_org_and_project_budgets(spend_counter_cache: DualCache):
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(
        key="org_id:org-budgeted:with_budget",
        value=LiteLLM_OrganizationTable(
            organization_id="org-budgeted",
            budget_id="org-budget",
            spend=1.5,
            models=[],
            created_by="admin",
            updated_by="admin",
            litellm_budget_table=LiteLLM_BudgetTable(max_budget=10.0),
        ),
        model_type=LiteLLM_OrganizationTable,
    )
    await cache.async_set_cache(
        key=project_cache_key("project-budgeted"),
        value=LiteLLM_ProjectTableCachedObj(
            project_id="project-budgeted", spend=2.5, litellm_budget_table=LiteLLM_BudgetTable(max_budget=20.0)
        ),
        model_type=LiteLLM_ProjectTableCachedObj,
    )

    reservation: Final = await reserve_budget_for_request(
        request_body={"model": "gpt-4o", "input": "hello"},
        route="/v1/responses",
        llm_router=None,
        valid_token=UserAPIKeyAuth(token="hashed-org-project", org_id="org-budgeted", project_id="project-budgeted"),
        team_object=None,
        user_object=None,
        prisma_client=None,
        user_api_key_cache=cache,
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
    )

    assert reservation is not None
    reserved_cost: Final = reservation["reserved_cost"]
    assert reserved_cost > 0
    assert reservation["entries"] == [
        {
            "counter_key": "spend:org:org-budgeted",
            "entity_type": "Organization",
            "entity_id": "org-budgeted",
            "reserved_cost": reserved_cost,
            "applied_adjustment": 0.0,
        },
        {
            "counter_key": "spend:project:project-budgeted",
            "entity_type": "Project",
            "entity_id": "project-budgeted",
            "reserved_cost": reserved_cost,
            "applied_adjustment": 0.0,
        },
    ]
    assert spend_counter_cache.in_memory_cache.get_cache(key="spend:org:org-budgeted") == pytest.approx(reserved_cost)
    assert spend_counter_cache.in_memory_cache.get_cache(key="spend:project:project-budgeted") == pytest.approx(
        reserved_cost
    )


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        (
            '{"budget_duration": "1h", "reset_at": "2030-01-01T01:00:00Z"}',
            datetime(2030, 1, 1, 0, 0, tzinfo=timezone.utc),
        ),
        ('{"reset_at": "2030-01-01T01:00:00Z"}', None),
        ("{}", None),
        ('["budget_duration", "1h"]', None),
        ('"1h"', None),
        ("null", None),
        ("not json", None),
    ],
)
def test_budget_window_start_reads_json_encoded_windows(window: str, expected: datetime | None) -> None:
    assert get_budget_window_start(window) == expected


def _decision_rerank_model(strategy: str) -> str:
    return "perplexity/pplx-decider-flash-v1.2" if strategy == "multiple_choice" else "typesafe/jev-1.14.0"


@pytest.fixture
def decision_rerank_catalog(monkeypatch):
    for strategy in ("pointwise", "multiple_choice"):
        model = _decision_rerank_model(strategy)
        monkeypatch.setitem(
            litellm.model_cost,
            model,
            {**litellm.model_cost[model], "rerank_strategy": strategy},
        )


def _decision_rerank_router(strategy: str = "pointwise", *, model: str | None = None) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "decision-reranker",
                "litellm_params": {
                    "model": model or _decision_rerank_model(strategy),
                    "api_key": "test-key",
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.01,
                },
                "model_info": {"max_output_tokens": 7},
            }
        ]
    )


def _decision_expected_costs(body: dict, strategy: Literal["pointwise", "multiple_choice"]) -> tuple[float, float]:
    from litellm.rerank_api.litellm_decisions_transformation.transformation import (
        LiteLLMDecisionsRerankConfig,
    )

    request: Final = LiteLLMDecisionsRerankConfig.validate_request(
        body,
        _decision_rerank_model(strategy),
        "perplexity" if strategy == "multiple_choice" else "typesafe",
        strategy=strategy,
    )
    count: Final = 1 if strategy == "multiple_choice" else len(body["documents"])
    input_cost: Final = sum(
        litellm.token_counter(
            model=_decision_rerank_model(strategy),
            text=LiteLLMDecisionsRerankConfig.transform_rerank_request(request, index).model_dump_json(),
        )
        * 0.001
        for index in range(count)
    )
    return input_cost, input_cost + count * 7 * 0.01


@pytest.mark.parametrize("strategy", ["pointwise", "multiple_choice"])
def test_decision_rerank_alias_reserves_transformed_inputs_and_each_output_bound(strategy, decision_rerank_catalog):
    from litellm.proxy.spend_tracking.budget_reservation import (
        estimate_request_input_cost,
    )

    router: Final = _decision_rerank_router(strategy)
    body: Final = {
        "model": "decision-reranker",
        "query": "repeat this query " * 30,
        "documents": ["one", "two", "three"],
        "max_tokens": 1,
    }
    input_cost, maximum = _decision_expected_costs(body, strategy)
    # Cached raw-body token counts must not override the transformed call estimate.
    cached: Final = {"decision-reranker": 1}
    assert estimate_request_max_cost(body, "/rerank", router, cached) == pytest.approx(maximum)
    assert estimate_request_input_cost(body, "/rerank", router, cached) == pytest.approx(input_cost)


def test_decision_rerank_request_cannot_override_catalog_strategy(
    decision_rerank_catalog,
):
    router: Final = _decision_rerank_router("multiple_choice")
    body: Final = {
        "model": "decision-reranker",
        "query": "query",
        "documents": ["one", "two", "three"],
        "decision_rerank": {"strategy": "pointwise"},
    }
    with pytest.raises(litellm.UnsupportedParamsError):
        estimate_request_max_cost(body, "/rerank", router)


def test_native_rerank_estimate_remains_input_only():
    router: Final = _decision_rerank_router(model="cohere/rerank-v3.5")
    body: Final = {
        "model": "decision-reranker",
        "query": "query",
        "documents": ["one", "two"],
    }
    assert estimate_request_max_cost(body, "/rerank", router, {"decision-reranker": 10}) == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_fail_closed_budget_rejects_aggregate_decision_fanout_before_admission(
    spend_counter_cache, decision_rerank_catalog
):
    router: Final = _decision_rerank_router()
    body: Final = {
        "model": "decision-reranker",
        "query": "repeat this query " * 30,
        "documents": ["one", "two", "three"],
    }
    _, required = _decision_expected_costs(body, "pointwise")
    with pytest.raises(litellm.BudgetExceededError):
        await reserve_budget_for_request(
            request_body=body,
            route="/rerank",
            llm_router=router,
            valid_token=UserAPIKeyAuth(token="decision-budget-test", max_budget=required / 2, spend=0.0),
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=UserApiKeyCache(),
            proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
            fail_closed_budget_enforcement=True,
        )


@pytest.mark.asyncio
async def test_decision_rerank_cancellation_settles_only_dispatched_inputs(
    spend_counter_cache, respx_mock, monkeypatch, decision_rerank_catalog
):
    import asyncio

    from litellm.proxy.spend_tracking.budget_reservation import (
        release_budget_reservation_on_cancel,
    )

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    router: Final = _decision_rerank_router()
    body: Final = {
        "model": "decision-reranker",
        "query": "query",
        "documents": [f"document {index}" for index in range(12)],
    }
    reservation: Final = await reserve_budget_for_request(
        request_body=body,
        route="/rerank",
        llm_router=router,
        valid_token=UserAPIKeyAuth(token="cancel-decision-budget", max_budget=100, spend=0.0),
        team_object=None,
        user_object=None,
        prisma_client=None,
        user_api_key_cache=UserApiKeyCache(),
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
        fail_closed_budget_enforcement=True,
    )
    assert reservation is not None
    assert reservation["input_cost"] == 0
    started = []
    eight_started = asyncio.Event()

    async def score(request):
        started.append(request)
        if len(started) == 8:
            eight_started.set()
        await asyncio.Event().wait()

    respx_mock.post("https://api.typesafe.ai/v1/systemone").mock(side_effect=score)
    task = asyncio.create_task(router.arerank(**body, metadata={"user_api_key_budget_reservation": reservation}))
    await asyncio.wait_for(eight_started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    expected: Final = sum(reservation["decision_rerank_input_costs"]["pointwise"][:8])
    assert len(started) == 8
    assert reservation["input_cost"] == pytest.approx(expected)
    await release_budget_reservation_on_cancel(reservation)
    assert spend_counter_cache.in_memory_cache.get_cache(key="spend:key:cancel-decision-budget") == pytest.approx(
        expected
    )


@pytest.mark.asyncio
async def test_unknown_decision_output_pricing_cannot_fall_back_to_raw_rerank_estimate(
    spend_counter_cache, monkeypatch, decision_rerank_catalog
):
    from fastapi import HTTPException

    from litellm.proxy.spend_tracking import budget_reservation as budgets

    monkeypatch.setattr(
        budgets,
        "_get_model_cost_infos",
        lambda **kwargs: [{"input_cost_per_token": 0.001}],
    )
    body: Final = {
        "model": "decision-reranker",
        "query": "query",
        "documents": ["one", "two"],
    }
    with pytest.raises(HTTPException) as error:
        await reserve_budget_for_request(
            request_body=body,
            route="/rerank",
            llm_router=_decision_rerank_router(),
            valid_token=UserAPIKeyAuth(token="unknown-decision-budget", max_budget=100, spend=0.0),
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=UserApiKeyCache(),
            proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
            fail_closed_budget_enforcement=True,
        )
    assert error.value.status_code == 503
    assert spend_counter_cache.in_memory_cache.get_cache(key="spend:key:unknown-decision-budget") is None


@pytest.mark.asyncio
async def test_cancellation_during_decision_estimation_does_not_reserve_counters(
    spend_counter_cache, monkeypatch, decision_rerank_catalog
):
    import asyncio
    import threading

    from litellm.proxy.spend_tracking import budget_reservation as budgets

    started = threading.Event()
    release = threading.Event()
    original = budgets._decision_rerank_costs

    def estimate(*args, **kwargs):
        started.set()
        release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(budgets, "_decision_rerank_costs", estimate)
    task = asyncio.create_task(
        reserve_budget_for_request(
            request_body={
                "model": "decision-reranker",
                "query": "query",
                "documents": ["one", "two"],
            },
            route="/rerank",
            llm_router=_decision_rerank_router(),
            valid_token=UserAPIKeyAuth(token="estimate-cancel-budget", max_budget=100, spend=0.0),
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=UserApiKeyCache(),
            proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
            fail_closed_budget_enforcement=True,
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert spend_counter_cache.in_memory_cache.get_cache(key="spend:key:estimate-cancel-budget") is None
    finally:
        release.set()


def test_explicit_provider_deployment_reserves_each_decision_call(
    decision_rerank_catalog,
):
    router = Router(
        model_list=[
            {
                "model_name": "decision-reranker",
                "litellm_params": {
                    "model": "jev-1.14.0",
                    "custom_llm_provider": "typesafe",
                    "api_key": "test-key",
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.01,
                },
                "model_info": {"max_output_tokens": 7},
            }
        ]
    )
    body = {
        "model": "decision-reranker",
        "query": "query",
        "documents": ["one", "two", "three"],
    }
    _, expected = _decision_expected_costs(body, "pointwise")
    assert estimate_request_max_cost(body, "/rerank", router, {"decision-reranker": 1}) == pytest.approx(expected)
