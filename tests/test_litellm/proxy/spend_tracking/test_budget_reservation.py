from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

import litellm
from litellm.caching import DualCache
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.proxy import proxy_server
from litellm.proxy._types import (
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    team_membership_reservation_cache_key,
)
from litellm.proxy.spend_tracking.budget_reservation import (
    _get_team_member_budget_counter,
    estimate_request_max_cost,
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
@pytest.mark.parametrize("enabled", [False, True])
async def test_overflow_reserves_crossing_request_only_against_team_cap(
    spend_counter_cache: DualCache, enabled: bool,
) -> None:
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(
        key=team_membership_reservation_cache_key(user_id="overflow-member", team_id="overflow-team"),
        value=LiteLLM_TeamMembership(
            user_id="overflow-member", team_id="overflow-team", spend=0.0,
            litellm_budget_table=LiteLLM_BudgetTable(max_budget=1e-9),
        ),
    )
    reserve: Final = reserve_budget_for_request(
        request_body={"model": "gpt-6-astra", "input": "hello", "max_output_tokens": 10},
        route="/v1/responses", llm_router=None,
        valid_token=UserAPIKeyAuth(token="overflow-key", user_id="overflow-member", team_id="overflow-team"),
        team_object=LiteLLM_TeamTable(
            team_id="overflow-team", max_budget=10.0,
            metadata={"allow_team_member_budget_overflow": enabled},
        ),
        user_object=LiteLLM_UserTable(user_id="overflow-member"), prisma_client=None,
        user_api_key_cache=cache, proxy_logging_obj=ProxyLogging(user_api_key_cache=cache),
        fail_closed_budget_enforcement=True,
    )
    if not enabled:
        with pytest.raises(litellm.BudgetExceededError, match="TeamMember"):
            await reserve
        return
    reservation: Final = await reserve
    assert reservation is not None
    assert reservation["reserved_cost"] > 1e-9
    assert [entry["counter_key"] for entry in reservation["entries"]] == ["spend:team:overflow-team"]
    assert spend_counter_cache.in_memory_cache.get_cache(key="spend:team:overflow-team") == reservation["reserved_cost"]
    await release_unbound_budget_reservation(reservation)


@pytest.mark.asyncio
async def test_overflow_keeps_every_other_budget_reservation_guard() -> None:
    from litellm.models.project import LiteLLM_ProjectTable
    from litellm.proxy._types import LiteLLM_OrganizationTable
    from litellm.proxy.common_utils.user_api_key_cache import project_cache_key
    from litellm.proxy.spend_tracking.budget_reservation import _get_budget_counters

    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(
        key=team_membership_reservation_cache_key(user_id="overflow-member", team_id="overflow-team"),
        value=LiteLLM_TeamMembership(
            user_id="overflow-member", team_id="overflow-team", spend=1.0,
            litellm_budget_table=LiteLLM_BudgetTable(max_budget=1.0),
        ),
    )
    await cache.async_set_cache(
        key="org_id:overflow-org:with_budget",
        value=LiteLLM_OrganizationTable(
            organization_id="overflow-org", budget_id="org-budget", created_by="admin", updated_by="admin",
            litellm_budget_table=LiteLLM_BudgetTable(max_budget=20.0),
        ),
    )
    await cache.async_set_cache(
        key=project_cache_key("overflow-project"),
        value=LiteLLM_ProjectTable(project_id="overflow-project", team_id="overflow-team", litellm_budget_table=LiteLLM_BudgetTable(max_budget=5.0)),
    )
    counters: Final = await _get_budget_counters(
        request_body={},
        valid_token=UserAPIKeyAuth(
            token="overflow-key", user_id="overflow-member", team_id="overflow-team",
            project_id="overflow-project", max_budget=2.0,
        ),
        team_object=LiteLLM_TeamTable(
            team_id="overflow-team", organization_id="overflow-org", max_budget=10.0,
            budget_limits=[{"budget_duration": "1d", "max_budget": 3.0}],
            metadata={"allow_team_member_budget_overflow": True},
        ),
        user_object=LiteLLM_UserTable(user_id="overflow-member", max_budget=4.0),
        prisma_client=None, user_api_key_cache=cache,
        proxy_logging_obj=ProxyLogging(user_api_key_cache=cache), apply_user_budget_to_team_keys=True,
    )
    assert [(counter.entity_type, counter.max_budget) for counter in counters] == [
        ("Key", 2.0), ("Team", 10.0), ("Team", 3.0), ("User", 4.0), ("Organization", 20.0), ("Project", 5.0),
    ]


@pytest.mark.asyncio
async def test_overflow_requests_compete_for_remaining_team_reservation_capacity(spend_counter_cache: DualCache) -> None:
    import asyncio

    body: Final = {"model": "gpt-6-astra", "input": "hello", "max_output_tokens": 10}
    estimated: Final = estimate_request_max_cost(request_body=body, route="/v1/responses", llm_router=None)
    assert estimated > 0
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(
        key=team_membership_reservation_cache_key(user_id="concurrent-member", team_id="concurrent-team"),
        value=LiteLLM_TeamMembership(
            user_id="concurrent-member", team_id="concurrent-team", spend=1.0,
            litellm_budget_table=LiteLLM_BudgetTable(max_budget=1.0),
        ),
    )

    async def reserve() -> object:
        return await reserve_budget_for_request(
            request_body=body, route="/v1/responses", llm_router=None,
            valid_token=UserAPIKeyAuth(token="concurrent-key", user_id="concurrent-member", team_id="concurrent-team"),
            team_object=LiteLLM_TeamTable(
                team_id="concurrent-team", max_budget=estimated * 1.5,
                metadata={"allow_team_member_budget_overflow": True},
            ),
            user_object=LiteLLM_UserTable(user_id="concurrent-member"), prisma_client=None,
            user_api_key_cache=cache, proxy_logging_obj=ProxyLogging(user_api_key_cache=cache),
            fail_closed_budget_enforcement=True,
        )

    results: Final = await asyncio.gather(reserve(), reserve(), return_exceptions=True)
    admitted: Final = tuple(result for result in results if isinstance(result, dict))
    rejected: Final = tuple(result for result in results if isinstance(result, litellm.BudgetExceededError))
    assert len(admitted) == 1
    assert len(rejected) == 1
    assert "Team=concurrent-team" in str(rejected[0])
    assert spend_counter_cache.in_memory_cache.get_cache(key="spend:team:concurrent-team") == pytest.approx(estimated)
    await release_unbound_budget_reservation(admitted[0])
    assert spend_counter_cache.in_memory_cache.get_cache(key="spend:team:concurrent-team") == pytest.approx(0.0)
