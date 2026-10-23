from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Final

import httpx
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
    count_request_input_tokens,
    estimate_request_input_cost,
    estimate_request_max_cost,
    get_budget_window_start,
    release_unbound_budget_reservation,
    reserve_budget_for_request,
)
from litellm.proxy.utils import ProxyLogging
from litellm.rerank_api.litellm_decisions_transformation.transformation import (
    LiteLLMDecisionsRerankConfig,
)
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


@pytest.fixture
def decision_rerank_models(monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    models: Final = ("typesafe/budget-pointwise", "typesafe/budget-choice")
    for model, strategy in zip(models, ("pointwise", "multiple_choice"), strict=True):
        monkeypatch.setitem(
            litellm.model_cost,
            model,
            {
                "mode": "evaluation",
                "litellm_provider": "typesafe",
                "rerank_strategy": strategy,
                "max_input_tokens": 65536,
                "max_output_tokens": 32,
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
            },
        )
    return models


def _decision_rerank_body(model: str, count: int) -> dict[str, object]:
    return {
        "model": model,
        "query": "Which passage explains the solar cycle?",
        "documents": ["A solar passage."] * count,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/rerank", "/v1/rerank", "/v2/rerank"])
async def test_pointwise_reserves_every_transformed_input_and_output(
    decision_rerank_models: tuple[str, str], route: str
) -> None:
    model: Final = decision_rerank_models[0]
    single: Final = _decision_rerank_body(model, 1)
    multiple: Final = {**_decision_rerank_body(model, 3), "top_n": 1, "max_tokens": 1}
    single_counts: Final = await count_request_input_tokens(single, route, None, json.dumps(single).encode())
    counts: Final = await count_request_input_tokens(multiple, route, None, json.dumps(multiple).encode())
    raw_prompt_tokens: Final = litellm.token_counter(model=model, text=str(single["query"])) + litellm.token_counter(
        model=model, text="A solar passage."
    )
    expected_input_cost: Final = counts[model] * 0.001
    expected_output_cost: Final = 3 * 32 * 0.002

    request: Final = LiteLLMDecisionsRerankConfig.validate_request(multiple, model, "typesafe", "pointwise")
    sent_inputs: Final = tuple(
        LiteLLMDecisionsRerankConfig.transform_rerank_request(request, index) for index in range(request.request_count)
    )
    assert single_counts[model] > raw_prompt_tokens
    assert counts[model] == sum(
        litellm.token_counter(model=model, text=sent.model_dump_json(exclude_none=True)) for sent in sent_inputs
    )
    assert counts[model] == 3 * single_counts[model]
    assert estimate_request_max_cost(multiple, route, None, counts) == pytest.approx(
        expected_input_cost + expected_output_cost
    )
    assert estimate_request_max_cost(multiple, route, None) == pytest.approx(expected_input_cost + expected_output_cost)
    assert estimate_request_input_cost(multiple, route, None, counts) == pytest.approx(expected_input_cost)


@pytest.mark.asyncio
async def test_multiple_choice_reserves_one_decision_output(decision_rerank_models: tuple[str, str]) -> None:
    model: Final = decision_rerank_models[1]
    body: Final = _decision_rerank_body(model, 3)
    counts: Final = await count_request_input_tokens(body, "/rerank", None)
    input_cost: Final = estimate_request_input_cost(body, "/rerank", None, counts)
    total_cost: Final = estimate_request_max_cost(body, "/rerank", None, counts)

    assert input_cost is not None
    assert total_cost == pytest.approx(input_cost + 32 * 0.002)


@pytest.mark.asyncio
async def test_alias_reserves_decision_fanout_at_deployment_prices(decision_rerank_models: tuple[str, str]) -> None:
    backend: Final = decision_rerank_models[0]
    alias: Final = "rerank-budget-group"
    router: Final = Router(
        model_list=[
            {
                "model_name": alias,
                "litellm_params": {
                    "model": backend,
                    "api_key": "test-key",
                    "input_cost_per_token": 0.003,
                    "output_cost_per_token": 0.004,
                },
            }
        ]
    )
    body: Final = _decision_rerank_body(alias, 3)
    counts: Final = await count_request_input_tokens(body, "/rerank", router, json.dumps(body).encode())
    direct_counts: Final = await count_request_input_tokens(_decision_rerank_body(backend, 3), "/rerank", None)

    assert counts[alias] == direct_counts[backend]
    assert estimate_request_max_cost(body, "/rerank", router, counts) == pytest.approx(
        counts[alias] * 0.003 + 3 * 32 * 0.004
    )


@pytest.mark.asyncio
async def test_budget_rejects_fanout_that_only_fits_one_decision(
    decision_rerank_models: tuple[str, str], spend_counter_cache: DualCache
) -> None:
    model: Final = decision_rerank_models[0]
    single_cost: Final = estimate_request_max_cost(_decision_rerank_body(model, 1), "/rerank", None)
    assert single_cost is not None
    with pytest.raises(litellm.BudgetExceededError):
        await reserve_budget_for_request(
            request_body=_decision_rerank_body(model, 3),
            route="/rerank",
            llm_router=None,
            valid_token=UserAPIKeyAuth(token="decision-fanout-budget", max_budget=single_cost * 2, spend=0.0),
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=UserApiKeyCache(),
            proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
            fail_closed_budget_enforcement=True,
        )
    assert spend_counter_cache.in_memory_cache.get_cache(key="spend:key:decision-fanout-budget") == pytest.approx(0.0)


def test_native_rerank_still_reserves_only_query_and_document_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    model: Final = "cohere/budget-native-rerank"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {"mode": "rerank", "litellm_provider": "cohere", "input_cost_per_token": 0.001, "output_cost_per_token": 0.002},
    )
    body: Final = _decision_rerank_body(model, 3)
    input_cost: Final = estimate_request_input_cost(body, "/rerank", None)
    assert input_cost is not None
    assert estimate_request_max_cost(body, "/rerank", None) == input_cost


@pytest.mark.asyncio
async def test_decision_fanout_does_not_apply_aggregate_tier_discount(
    decision_rerank_models: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    model: Final = decision_rerank_models[0]
    single: Final = _decision_rerank_body(model, 1)
    single_counts: Final = await count_request_input_tokens(single, "/rerank", None)
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {
            **litellm.model_cost[model],
            "tiered_pricing": [
                {"range": [0, single_counts[model]], "input_cost_per_token": 0.01, "output_cost_per_token": 0.02},
                {
                    "range": [single_counts[model], 100000],
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                },
            ],
        },
    )
    single_cost: Final = estimate_request_max_cost(single, "/rerank", None)
    single_input: Final = estimate_request_input_cost(single, "/rerank", None)
    multiple: Final = _decision_rerank_body(model, 3)

    assert single_cost is not None
    assert single_input is not None
    assert estimate_request_max_cost(multiple, "/rerank", None) == pytest.approx(single_cost * 3)
    assert estimate_request_input_cost(multiple, "/rerank", None) == pytest.approx(single_input * 3)


@pytest.mark.asyncio
async def test_budget_honors_deployment_drop_params_and_timeout(decision_rerank_models: tuple[str, str]) -> None:
    backend: Final = decision_rerank_models[0]
    alias: Final = "rerank-drop-params-group"
    router: Final = Router(
        model_list=[
            {
                "model_name": alias,
                "litellm_params": {
                    "model": backend,
                    "api_key": "test-key",
                    "drop_params": True,
                    "timeout": httpx.Timeout(30.0),
                },
            }
        ]
    )
    plain: Final = _decision_rerank_body(alias, 3)
    with_dropped: Final = {**plain, "rank_fields": ["text"]}
    plain_counts: Final = await count_request_input_tokens(plain, "/rerank", router)
    dropped_counts: Final = await count_request_input_tokens(with_dropped, "/rerank", router)

    direct_counts: Final = await count_request_input_tokens(_decision_rerank_body(backend, 3), "/rerank", None)
    assert plain_counts[alias] == direct_counts[backend]
    assert plain_counts == dropped_counts
    assert estimate_request_max_cost(plain, "/rerank", router) == estimate_request_max_cost(
        with_dropped, "/rerank", router
    )


def test_invalid_decision_candidate_does_not_reject_native_rerank_group(
    decision_rerank_models: tuple[str, str],
) -> None:
    alias: Final = "rerank-mixed-group"
    router: Final = Router(
        model_list=[
            {"model_name": alias, "litellm_params": {"model": decision_rerank_models[0], "api_key": "test-key"}},
            {"model_name": alias, "litellm_params": {"model": "cohere/rerank-v3.5", "api_key": "test-key"}},
        ]
    )
    body: Final = {**_decision_rerank_body(alias, 3), "rank_fields": ["text"]}

    assert estimate_request_max_cost(body, "/rerank", router) is not None


def test_pointwise_candidate_reserves_when_choice_candidate_is_ineligible(
    decision_rerank_models: tuple[str, str],
) -> None:
    alias: Final = "rerank-mixed-strategy-group"
    router: Final = Router(
        model_list=[
            {"model_name": alias, "litellm_params": {"model": model, "api_key": "test-key"}}
            for model in decision_rerank_models
        ]
    )
    body: Final = _decision_rerank_body(alias, 1)
    direct: Final = _decision_rerank_body(decision_rerank_models[0], 1)

    assert estimate_request_max_cost(body, "/rerank", router) == estimate_request_max_cost(direct, "/rerank", None)
