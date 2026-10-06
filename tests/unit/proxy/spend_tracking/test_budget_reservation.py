from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
from fastapi import HTTPException

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
@pytest.mark.parametrize("charged_agent", ("caller-agent", "target-agent"))
@pytest.mark.parametrize("window", [None, "2026-01-02T00:00:00Z"])
@pytest.mark.parametrize("outcome", ["success", "cancelled"])
async def test_agent_invocation_reserves_exact_fee_and_reconciles_without_child_cost(
    monkeypatch: pytest.MonkeyPatch, charged_agent: str, window: str | None, outcome: str,
) -> None:
    from litellm.proxy.spend_tracking.budget_reservation import reconcile_budget_reservation
    from litellm.types.agents import AgentResponse

    cache: Final = DualCache()
    counter_key: Final = f"spend:agent_window:20260102T000000.000000Z:{charged_agent}" if window else f"spend:agent_lifetime:agent-budget:{charged_agent}"
    cache.set_cache(counter_key, 0.1)
    monkeypatch.setattr(proxy_server, "spend_counter_cache", cache)
    auth: Final = UserAPIKeyAuth(agent_id="caller-agent" if charged_agent == "caller-agent" else None)
    auth.billing_agent_policy = AgentResponse(
        agent_id=charged_agent, agent_name="Charged agent", agent_card_params={}, spend=0.1, lifetime_budget_spend=0.1,
        litellm_budget_table={"budget_id": "agent-budget", "max_budget": 0.5, "budget_reset_at": window, "budget_duration": "1d" if window else None},
    )
    auth.invoked_agent_id = "target-agent"
    auth.agent_invocation_cost = 0.2
    reservation: Final = await reserve_budget_for_request(
        request_body={"jsonrpc": "2.0", "method": "message/send"}, route="/a2a/target-agent",
        llm_router=None, valid_token=auth, team_object=None, user_object=None, prisma_client=None,
        user_api_key_cache=UserApiKeyCache(), proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
        fail_closed_budget_enforcement=True,
    )
    assert reservation is not None
    assert reservation["reserved_cost"] == pytest.approx(0.2)
    assert [entry["counter_key"] for entry in reservation["entries"]] == [counter_key]
    assert await cache.async_get_cache(counter_key) == pytest.approx(0.3)
    if outcome == "cancelled":
        from litellm.proxy.spend_tracking.budget_reservation import release_budget_reservation_on_cancel

        await release_budget_reservation_on_cancel(reservation)
        assert await cache.async_get_cache(counter_key) == pytest.approx(0.1)
        assert reservation["finalized"] is True
        return
    await proxy_server.increment_spend_counters(
        token=None, team_id=None, user_id=None, response_cost=0.2,
        billing_agent_id=charged_agent, billing_agent_counter_key=counter_key, budget_reservation=reservation,
    )
    await reconcile_budget_reservation(reservation, actual_cost=4.0)
    assert await cache.async_get_cache(counter_key) == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_agent_invocation_over_budget_is_rejected_and_reservation_is_refunded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.types.agents import AgentResponse

    cache: Final = DualCache()
    cache.set_cache("spend:agent_lifetime:agent-budget:agent", 0.4)
    monkeypatch.setattr(proxy_server, "spend_counter_cache", cache)
    auth: Final = UserAPIKeyAuth(agent_id="agent")
    auth.billing_agent_policy = AgentResponse(
        agent_id="agent", agent_name="Charged agent", agent_card_params={}, spend=0.4, lifetime_budget_spend=0.4,
        litellm_budget_table={"budget_id": "agent-budget", "max_budget": 0.5},
    )
    auth.agent_invocation_cost = 0.2
    with pytest.raises(litellm.BudgetExceededError):
        await reserve_budget_for_request(
            request_body={"method": "message/send"}, route="/a2a/target-agent", llm_router=None,
            valid_token=auth, team_object=None, user_object=None, prisma_client=None,
            user_api_key_cache=UserApiKeyCache(), proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
            fail_closed_budget_enforcement=True,
        )
    assert await cache.async_get_cache("spend:agent_lifetime:agent-budget:agent") == pytest.approx(0.4)


@pytest.mark.asyncio
@pytest.mark.parametrize("priced", (True, False))
async def test_native_a2a_invoked_agent_model_requires_fixed_fee_for_token_pricing(
    monkeypatch: pytest.MonkeyPatch,
    priced: bool,
) -> None:
    from litellm.types.agents import AgentResponse

    model: Final = "native-agent-budget-pricing-test"
    model_cost: Final = {
        **litellm.model_cost,
        **(
            {
                model: {
                    "input_cost_per_token": 0.001,
                    "litellm_provider": "openai",
                    "mode": "chat",
                }
            }
            if priced
            else {}
        ),
    }
    monkeypatch.setattr(litellm, "model_cost", model_cost)
    litellm.get_model_info.cache_clear()

    agent: Final = AgentResponse(
        agent_id="target-agent",
        agent_name="Target agent",
        agent_card_params={},
        litellm_params={"model": model},
        litellm_budget_table={"budget_id": "target-budget", "max_budget": 1.0},
    )
    counter_key: Final = agent.budget_counter_key
    cache: Final = DualCache()
    cache.set_cache(counter_key, 0.0)
    monkeypatch.setattr(proxy_server, "spend_counter_cache", cache)
    auth: Final = UserAPIKeyAuth()
    auth.invoked_agent_policy = agent
    auth.target_agent_budget_policy = agent
    auth.agent_invocation_cost = None
    request_body: Final = {"jsonrpc": "2.0", "method": "message/send"}
    route: Final = "/a2a/target-agent"

    input_token_counts: Final = await count_request_input_tokens(
        request_body=request_body,
        route=route,
        llm_router=None,
    )
    assert input_token_counts == {}
    assert estimate_request_max_cost(
        request_body=request_body,
        route=route,
        llm_router=None,
        input_token_counts=input_token_counts,
    ) is None

    if priced:
        with pytest.raises(HTTPException) as error:
            await reserve_budget_for_request(
                request_body=request_body,
                route=route,
                llm_router=None,
                valid_token=auth,
                team_object=None,
                user_object=None,
                prisma_client=None,
                user_api_key_cache=UserApiKeyCache(),
                proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
            )
        assert error.value.status_code == 503
        assert error.value.detail == (
            "Budgeted token-priced agent invocations require a fixed cost_per_query before execution"
        )
    else:
        reservation: Final = await reserve_budget_for_request(
            request_body=request_body,
            route=route,
            llm_router=None,
            valid_token=auth,
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=UserApiKeyCache(),
            proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
        )
        assert reservation is None
    assert await cache.async_get_cache(counter_key) == pytest.approx(0.0)


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
@pytest.mark.parametrize("outcome", ("failed", "cancelled", "completed"))
@pytest.mark.parametrize("budget_owner", ("agent", "key", "team"))
@pytest.mark.parametrize(("route", "pricing"), (("/a2a/target", "fixed"), ("/v1/chat/completions", "fixed"), ("/v1/chat/completions", "tokens")))
async def test_budgeted_caller_reserves_unmanaged_agent_fees_before_concurrent_admission(
    spend_counter_cache: DualCache, monkeypatch: pytest.MonkeyPatch, outcome: str, budget_owner: str, route: str, pricing: str
) -> None:
    import asyncio

    from litellm.proxy.agent_endpoints import agent_registry
    from litellm.proxy.agent_endpoints.auth.managed_authorization import invocation_target, prepare_agent_invocation
    from litellm.proxy.spend_tracking.budget_reservation import (
        reconcile_budget_reservation,
        release_budget_reservation,
        release_budget_reservation_on_cancel,
    )
    from litellm.types.agents import AgentResponse

    caller: Final = AgentResponse(
        agent_id="caller", agent_name="Caller", agent_card_params={}, spend=0.0,
        litellm_budget_table={"budget_id": "caller-budget", "max_budget": 0.5 if budget_owner == "agent" else None},
    )
    target: Final = AgentResponse(
        agent_id="target", agent_name="Target", agent_card_params={},
        litellm_params={"cost_per_query": 0.25} if pricing == "fixed" else {},
    )
    registry: Final = agent_registry.AgentRegistry()
    registry.register_agent(target)
    monkeypatch.setattr(agent_registry, "global_agent_registry", registry)

    if pricing == "tokens":
        target_cost_info: Final = (
            {"output_cost_per_token": 0.125}
            if budget_owner == "agent"
            else {"input_cost_per_token": 0.0, "output_cost_per_token": 0.125}
        )
        monkeypatch.setattr(litellm, "model_cost", {
            **litellm.model_cost,
            "a2a/target": {
                **target_cost_info,
                "litellm_provider": "a2a",
                "mode": "chat",
            },
        })
        litellm.get_model_info.cache_clear()

    token: Final = "fee-key" if budget_owner == "key" else None
    team: Final = LiteLLM_TeamTable(team_id="fee-team", max_budget=0.5, spend=0.0) if budget_owner == "team" else None
    counter_key: Final = (
        caller.budget_counter_key if budget_owner == "agent"
        else f"spend:key:{token}" if budget_owner == "key" else "spend:team:fee-team"
    )
    await spend_counter_cache.async_set_cache(key=counter_key, value=0.0)

    request_body: Final = {
        "method": "message/send",
        "model": "a2a/target",
        "max_tokens": 0 if budget_owner == "agent" and pricing == "tokens" else 2,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    if budget_owner == "agent" and pricing == "tokens":
        input_token_counts: Final = await count_request_input_tokens(
            request_body=request_body,
            route=route,
            llm_router=None,
        )
        assert input_token_counts["a2a/target"] > 0
        assert estimate_request_max_cost(
            request_body=request_body,
            route=route,
            llm_router=None,
            input_token_counts=input_token_counts,
        ) == pytest.approx(0.0)

    async def admit() -> dict[str, object] | None:
        auth: Final = UserAPIKeyAuth(
            agent_id="caller" if budget_owner == "agent" else None, user_role="proxy_admin",
            token=token, max_budget=0.5 if budget_owner == "key" else None,
            team_id=team.team_id if team is not None else None,
        )
        if budget_owner == "agent" or pricing == "tokens":
            auth.billing_agent_policy = caller
        target_name: Final = invocation_target(route, request_body)
        assert target_name == "target"
        await prepare_agent_invocation(auth, target_name, None)
        assert auth.invoked_agent_id == "target"
        assert auth.invoked_agent_policy is not None
        return await reserve_budget_for_request(
            request_body=request_body, route=route, llm_router=None,
            valid_token=auth, team_object=team, user_object=None, prisma_client=None,
            user_api_key_cache=UserApiKeyCache(), proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
            fail_closed_budget_enforcement=True,
        )

    results: Final = await asyncio.gather(*(admit() for _ in range(8)), return_exceptions=True)
    if budget_owner == "agent" and pricing == "tokens":
        policy_unavailable: Final = tuple(
            result
            for result in results
            if isinstance(result, HTTPException)
            and result.status_code == 503
            and result.detail
            == "Budgeted token-priced agent invocations require a fixed cost_per_query before execution"
        )
        assert len(policy_unavailable) == 8, results
        assert await spend_counter_cache.async_get_cache(counter_key) == pytest.approx(0.0)
        monkeypatch.setattr(litellm, "model_cost", {
            **litellm.model_cost,
            "a2a/target": {"litellm_provider": "a2a", "mode": "chat"},
        })
        litellm.get_model_info.cache_clear()
        unpriced_reservation: Final = await admit()
        assert unpriced_reservation is None
        assert await spend_counter_cache.async_get_cache(counter_key) == pytest.approx(0.0)
        monkeypatch.setattr(litellm, "model_cost", {
            **litellm.model_cost,
            "a2a/target": {
                "litellm_provider": "a2a",
                "mode": "chat",
                "tiered_pricing": [
                    {
                        "range": [0, 128000],
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                ],
            },
        })
        litellm.get_model_info.cache_clear()
        zero_rate_tier_reservation: Final = await admit()
        assert zero_rate_tier_reservation is None
        assert await spend_counter_cache.async_get_cache(counter_key) == pytest.approx(0.0)
        return
    accepted: Final = tuple(result for result in results if isinstance(result, dict))
    rejected: Final = tuple(result for result in results if isinstance(result, litellm.BudgetExceededError))
    assert all(result is None or isinstance(result, (dict, litellm.BudgetExceededError)) for result in results), results
    assert len(accepted) == 2, results
    assert len(rejected) == 6
    assert await spend_counter_cache.async_get_cache(counter_key) == pytest.approx(0.5)
    first: Final = accepted[0]
    if outcome == "completed":
        await proxy_server.increment_spend_counters(
            token=token, team_id=team.team_id if team is not None else None, user_id=None, response_cost=0.25,
            billing_agent_id=caller.agent_id if budget_owner == "agent" else None,
            billing_agent_counter_key=caller.budget_counter_key if budget_owner == "agent" else None,
            budget_reservation=first,
        )
        await reconcile_budget_reservation(first, actual_cost=0.25)
        assert await spend_counter_cache.async_get_cache(counter_key) == pytest.approx(0.5)
        with pytest.raises(litellm.BudgetExceededError):
            await admit()
    else:
        release: Final = release_budget_reservation_on_cancel if outcome == "cancelled" else release_budget_reservation
        await release(first)
        await release(first)
        assert await spend_counter_cache.async_get_cache(counter_key) == pytest.approx(0.25)
        assert await admit() is not None
        assert await spend_counter_cache.async_get_cache(counter_key) == pytest.approx(0.5)


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
