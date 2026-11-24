from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import wave
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from datetime import time as dt_time
from types import SimpleNamespace
from typing import Final
from unittest.mock import patch

import pytest
from fastapi import Request

import litellm
from litellm.caching import DualCache
from litellm.models.budget import BudgetLimit, LiteLLM_BudgetTable
from litellm.proxy import proxy_server
from litellm.llms.openai.responses.transformation import OPENAI_RESPONSES_API_MIN_MAX_OUTPUT_TOKENS
from litellm.proxy._types import (
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.timezone_utils import BudgetResetSettings, get_budget_window_reset_fields
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.spend_tracking.budget_reservation import (
    DEFAULT_MAX_OUTPUT_TOKENS_FALLBACK,
    _get_budget_counters,
    _get_team_member_budget_counter,
    input_cost_for_cost_info,
    max_cost_for_cost_info,
    count_request_input_tokens,
    customer_window_counters,
    estimate_request_input_cost,
    estimate_request_max_cost,
    get_budget_window_start,
    reconcile_budget_reservation,
    reserve_budget_for_request,
)
from litellm.proxy.spend_tracking.input_tokens import count_input_tokens_for_model
from litellm.proxy.utils import ProxyLogging
from litellm.redis.stores.auth_cache import AuthObject
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore, user_window_counter
from litellm.router import Router
from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth, CustomerScopeAuth
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


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/v1/messages", "/v1/chat/completions", "/anthropic/v1/messages"])
@pytest.mark.parametrize("oauth,operator,skip", [(True, False, True), (False, False, False), (True, True, False)])
@pytest.mark.parametrize(
    "fallback",
    [
        None,
        "fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
        "default_fallbacks",
        "request_fallbacks",
        "request_models",
        "router_settings_override",
        "user_config",
        "key_settings",
        "team_settings",
    ],
)
async def test_only_forwarded_subscription_oauth_skips_budget_hold(
    route: str,
    oauth: bool,
    operator: bool,
    skip: bool,
    spend_counter_store: SpendCounterStore,
    monkeypatch: pytest.MonkeyPatch,
    fallback: str | None,
):
    credential: Final = "sk-ant-oat-test" if oauth else "sk-ant-api-test"
    request: Final = Request(
        {
            "type": "http",
            "path": route,
            "headers": [(b"authorization", f"Bearer {credential}".encode()), (b"x-litellm-api-key", b"sk-test")],
        }
    )
    router: Final = Router(
        model_list=[
            {
                "model_name": "claude",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5",
                    "api_key": "operator-key",
                    "use_in_pass_through": True,
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                },
            }
        ],
        fallbacks=[{"claude": ["paid"]}] if fallback == "fallbacks" else None,
        context_window_fallbacks=[{"claude": ["paid"]}] if fallback == "context_window_fallbacks" else None,
        content_policy_fallbacks=[{"claude": ["paid"]}] if fallback == "content_policy_fallbacks" else None,
        default_fallbacks=["paid"] if fallback == "default_fallbacks" else None,
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(proxy_server, "llm_router", router if operator else None)
    identity: Final = _budgeted_token().model_copy(
        update={
            **({"router_settings": {"fallbacks": [{"claude": ["paid"]}]}} if fallback == "key_settings" else {}),
            **({"team_id": "oauth-budget-team"} if fallback == "team_settings" else {}),
        }
    )
    if fallback == "team_settings":
        cache: Final = UserApiKeyCache()
        cache.set_cache(
            key=AuthObject.team("oauth-budget-team"),
            value=LiteLLM_TeamTableCachedObj(
                team_id="oauth-budget-team",
                router_settings={"fallbacks": [{"claude": ["paid"]}]},
            ),
        )
        monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
        monkeypatch.setattr(proxy_server, "prisma_client", SimpleNamespace())
    result: Final = await reserve_budget_for_request(
        request_body={
            "model": "claude",
            "messages": ANTHROPIC_MESSAGES,
            "max_tokens": 1,
            **({"fallbacks": ["paid"]} if fallback == "request_fallbacks" else {}),
            **({"models": ["claude", "paid"]} if fallback == "request_models" else {}),
            **(
                {"router_settings_override": {"fallbacks": [{"claude": ["paid"]}]}}
                if fallback == "router_settings_override"
                else {}
            ),
            **({"user_config": {"fallbacks": [{"claude": ["paid"]}]}} if fallback == "user_config" else {}),
            "metadata": {"used_client_oauth_token": True, "claude_subscription_usage": {"is_overage": False}},
        },
        route=route,
        llm_router=router,
        valid_token=identity,
        team_object=None,
        user_object=None,
        prisma_client=None,
        user_api_key_cache=UserApiKeyCache(),
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
        request=request,
    )
    expected_skip: Final = (fallback is None or route.startswith("/anthropic/")) and (
        skip or (oauth and operator and not route.startswith("/anthropic/"))
    )
    if expected_skip:
        assert result is None
        return
    assert result is not None
    assert result["reserved_cost"] > 0
    assert len(result["entries"]) == 1
    assert result["entries"][0]["reserved_cost"] == pytest.approx(result["reserved_cost"])
    await reconcile_budget_reservation(result, actual_cost=0.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/v1/messages", "/v1/chat/completions"])
@pytest.mark.parametrize("source", ["bare", "bearer", "key", "team", "global", "group", "cli", "completion"])
async def test_subscription_hold_uses_consumed_credential_and_prepared_model(
    route: str,
    source: str,
    spend_counter_store: SpendCounterStore,
    monkeypatch: pytest.MonkeyPatch,
):
    router: Final = Router(
        model_list=[
            {
                "model_name": "claude",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5",
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                },
            },
            {
                "model_name": "paid",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                },
            },
        ]
    )
    identity: Final = _budgeted_token().model_copy(
        update={
            **({"aliases": {"claude": "paid"}} if source == "key" else {}),
            **({"team_model_aliases": {"claude": "paid"}} if source == "team" else {}),
            **({"router_settings": {"model_group_alias": {"claude": "paid"}}} if source == "group" else {}),
        }
    )
    monkeypatch.setattr(litellm, "model_alias_map", {"claude": "paid"} if source == "global" else {})
    monkeypatch.setattr(proxy_server, "user_model", "paid" if source == "cli" else None)
    monkeypatch.setattr(
        proxy_server, "general_settings", {"completion_model": "paid"} if source == "completion" else {}
    )
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "operator-bearer")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    credential: Final = "sk-ant-oat-test" if source == "bare" else "Bearer sk-ant-oat-test"
    result: Final = await reserve_budget_for_request(
        request_body={"model": "claude", "messages": ANTHROPIC_MESSAGES, "max_tokens": 1},
        route=route,
        llm_router=router,
        valid_token=identity,
        team_object=None,
        user_object=None,
        prisma_client=None,
        user_api_key_cache=UserApiKeyCache(),
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
        request=Request(
            {
                "type": "http",
                "path": route,
                "headers": [
                    (b"authorization", credential.encode()),
                    (b"x-litellm-api-key", b"sk-test"),
                ],
            }
        ),
    )
    if source == "bearer":
        assert result is None
        return
    assert result is not None
    assert result["reserved_cost"] > 0
    await reconcile_budget_reservation(result, actual_cost=0.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("team, apply, expected", [(False, False, 2), (True, False, 0), (True, True, 2)])
async def test_user_reservation_includes_each_window_only_when_user_budget_applies(
    team: bool, apply: bool, expected: int
) -> None:
    cache: Final = UserApiKeyCache()
    counters: Final = await _get_budget_counters(
        request_body={},
        valid_token=UserAPIKeyAuth(),
        team_object=LiteLLM_TeamTable(team_id="t") if team else None,
        user_object=LiteLLM_UserTable.model_validate(
            {
                "user_id": "u",
                "budget_limits": [
                    {"budget_duration": "1h", "max_budget": 5.0, "reset_at": "2026-08-01T01:00:00Z"},
                    {"budget_duration": "30d", "max_budget": 100.0, "reset_at": "2026-08-31T00:00:00Z"},
                ],
            }
        ),
        prisma_client=None,
        user_api_key_cache=cache,
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
        apply_user_budget_to_team_keys=apply,
    )
    assert len(counters) == expected
    assert {counter.counter_key for counter in counters} == (
        {user_window_counter("u", "1h"), user_window_counter("u", "30d")} if expected else set()
    )


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
def spend_counter_store(monkeypatch: pytest.MonkeyPatch) -> SpendCounterStore:
    store: Final = SpendCounterStore(connect(None).spend_counters())
    monkeypatch.setattr(proxy_server, "spend_counter_store", store)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    return store


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


CUSTOMER: Final = CustomerHierarchyAuth(
    customer_org_id="org-1",
    scopes=(CustomerScopeAuth(key="customer_org", entity_id="org-1", max_budget=10000),),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("margin", "customer", "factor"),
    [(None, CUSTOMER, 1.0), (0.02, CUSTOMER, 1.02), (10, CUSTOMER, 11.0), (0.02, None, 1.0)],
)
async def test_customer_margin_scales_the_reservation_only_for_customer_requests(
    spend_counter_store, margin: float | None, customer: CustomerHierarchyAuth | None, factor: float
) -> None:
    router: Final = Router(model_list=[_tiered_deployment(0.1).model_dump()])
    estimate: Final = estimate_request_max_cost(request_body=TIERED_BODY, route="/chat/completions", llm_router=router)
    input_estimate: Final = estimate_request_input_cost(
        request_body=TIERED_BODY, route="/chat/completions", llm_router=router
    )
    with patch.dict(litellm.cost_margin_config, {} if margin is None else {"customer": margin}, clear=True):
        reservation: Final = await reserve_budget_for_request(
            request_body=TIERED_BODY,
            route="/chat/completions",
            llm_router=router,
            valid_token=UserAPIKeyAuth(token="margin-allowed", max_budget=10000, customer_hierarchy=customer),
            team_object=None,
            user_object=None,
            prisma_client=None,
            user_api_key_cache=UserApiKeyCache(),
            proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
        )
    assert estimate is not None and input_estimate is not None and reservation is not None
    assert reservation["reserved_cost"] == pytest.approx(estimate * factor)
    assert reservation["input_cost"] == pytest.approx(input_estimate * factor)


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


def _priced_group(model_name: str, cost_per_token: float) -> dict[str, object]:
    return Deployment(
        model_name=model_name,
        litellm_params=LiteLLM_Params(model="dashscope/qwen3-max", api_key="sk-fake"),
        model_info=ModelInfo(
            id=model_name,
            max_output_tokens=1000,
            tiered_pricing=[
                {"input_cost_per_token": cost_per_token, "output_cost_per_token": cost_per_token, "range": [0, 128000]}
            ],
        ),
    ).model_dump()


def test_google_route_reserves_on_the_model_its_path_param_selects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "user_model", None)
    router: Final = Router(model_list=[_priced_group("cheap-group", 1e-06), _priced_group("premium-group", 1e-03)])
    route: Final = "/v1beta/models/premium-group:generateContent"
    body: Final = {"model": "cheap-group", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 10}

    url_model_cost: Final = estimate_request_max_cost(request_body=body, route=route, llm_router=router)
    path_param_cost: Final = estimate_request_max_cost(
        request_body=body, route=route, llm_router=router, path_params={"model_name": "cheap-group"}
    )
    cheap_cost: Final = estimate_request_max_cost(
        request_body=body, route="/v1beta/models/cheap-group:generateContent", llm_router=router
    )

    assert url_model_cost is not None and cheap_cost is not None
    assert path_param_cost == cheap_cost
    assert math.isclose(url_model_cost, cheap_cost * 1000)


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
        key=AuthObject.team_membership_reservation(user_id=user_id, team_id=team_id),
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
        key=AuthObject.team_member_default_budget("default-bare"),
        value=LiteLLM_BudgetTable(budget_id="default-bare", max_budget=default_cap),
    )
    await cache.async_set_cache(
        key=AuthObject.team_membership_reservation(user_id=user_id, team_id=team_id),
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


@pytest.mark.parametrize(
    "reset_at,now,expected",
    (
        ("2026-12-01T06:30:00Z", "2026-11-01T06:15:00Z", "2026-11-01T05:30:00Z"),
        ("2026-03-01T06:30:00Z", "2026-02-15T12:00:00Z", "2026-02-01T06:30:00Z"),
    ),
)
def test_customer_and_inline_monthly_windows_count_the_same_calendar_period(
    reset_at: str, now: str, expected: str
) -> None:
    settings: Final = BudgetResetSettings(timezone="America/New_York", reset_time_of_day=dt_time(1, 30))
    limit: Final = BudgetLimit(budget_duration="1mo", max_budget=10.0, reset_at=datetime.fromisoformat(reset_at))
    counters: Final = customer_window_counters(
        "customer_team", "dst-team", (limit,), now=datetime.fromisoformat(now), reset_settings=settings
    )

    assert counters[0].window_start == datetime.fromisoformat(expected)
    assert (
        get_budget_window_start(
            {**limit.model_dump(), **get_budget_window_reset_fields("1mo", reset_at=limit.reset_at, settings=settings)},
            settings,
        )
        == counters[0].window_start
    )
    assert counters[0].reset_at == datetime.fromisoformat(reset_at)


@pytest.mark.parametrize("today", ["2028-03-01", "2029-03-01"])
@pytest.mark.parametrize(
    ("duration", "reset_at", "expected"),
    [
        ("12mo", "2029-01-01T00:00:00Z", datetime(2028, 1, 1, tzinfo=timezone.utc)),
        ("12mo", "2028-01-01T00:00:00", datetime(2027, 1, 1, tzinfo=timezone.utc)),
    ],
)
def test_annual_window_keeps_all_spend_from_its_reset_boundary(
    today: str, duration: str, reset_at: str, expected: datetime
) -> None:
    with patch("time.time", return_value=datetime.fromisoformat(today).replace(tzinfo=timezone.utc).timestamp()):
        assert get_budget_window_start({"budget_duration": duration, "reset_at": reset_at}) == expected


@pytest.mark.parametrize("duration", ("1h", "1mo", "12mo"))
@pytest.mark.parametrize("reset_at", ("2026-11-15T05:48:02.123456Z", "2026-11-15T11:33:02.123456+05:45"))
def test_legacy_window_start_uses_database_millisecond_precision(duration: str, reset_at: str) -> None:
    window: Final = {"budget_duration": duration, "reset_at": reset_at}
    whole_second: Final = {"budget_duration": duration, "reset_at": "2026-11-15T05:48:02Z"}
    start: Final = get_budget_window_start(whole_second)
    assert start is not None
    assert get_budget_window_start(window) == start.replace(microsecond=123000)


class _AnnualWindowDatabase:
    def __init__(self) -> None:
        self.db = self
        self.litellm_budgetwindowspend = self
        self.litellm_spendlogs = self
        self.start: datetime | None = None
        self.spend = 0.0
        self.scans = 0

    async def execute_raw(self, sql: str, *parameters: object) -> int:
        start: Final = parameters[3]
        assert isinstance(start, datetime)
        self.start = start
        self.spend = 0.0
        return 1

    async def find_unique(self, **query: object) -> SimpleNamespace:
        return SimpleNamespace(window_start=self.start, spend=self.spend)

    async def group_by(self, **query: object) -> list[dict[str, object]]:
        self.scans += 1
        return [{"api_key": "calendar-key", "_sum": {"spend": 0.0}}]


@pytest.mark.asyncio
async def test_reservation_counts_a_pdf_with_the_deployment_behind_a_gpt_named_group() -> None:
    from litellm.proxy.spend_tracking.input_tokens import count_input_tokens_for_model
    from tests.unit.proxy.spend_tracking.test_input_tokens import _one_page_pdf_message

    request_body: Final = {"model": "gpt-4o", "messages": _one_page_pdf_message()}
    router: Final = Router(
        model_list=[{"model_name": "gpt-4o", "litellm_params": {"model": "anthropic/claude-haiku-5-5", "api_key": "x"}}]
    )

    counts: Final = await count_request_input_tokens(
        request_body=request_body, route="/chat/completions", llm_router=router, team_id=None
    )

    assert counts == {"gpt-4o": count_input_tokens_for_model(request_body=request_body, model="claude-haiku-5-5")}
    assert counts["gpt-4o"] > count_input_tokens_for_model(request_body=request_body, model="gpt-4o")


_MULTILINGUAL_MESSAGES: Final = [{"role": "user", "content": "日本語のテキストとहिन्दी पाठ और русский текст. " * 20}]


def _router_with_groups(*groups: tuple[str, str]) -> Router:
    return Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": model, "api_key": "sk-test"}} for name, model in groups
        ]
    )


async def _input_tokens(router: Router, model: str) -> dict[str, int]:
    return dict(
        await count_request_input_tokens(
            request_body={"model": model, "messages": _MULTILINGUAL_MESSAGES},
            route="/v1/chat/completions",
            llm_router=router,
            team_id=None,
        )
    )


async def _reserved_input_tokens(router: Router, model: str, route: str, messages: list[dict[str, object]]) -> int:
    counts: Final = await count_request_input_tokens(
        request_body={"model": model, "messages": messages}, route=route, llm_router=router, team_id=None
    )
    return counts[model]


@pytest.mark.asyncio
async def test_an_openai_chat_request_reserves_what_openai_chat_bills_on_top_of_each_pdf_files_pages() -> None:
    from litellm.constants import OPENAI_CHAT_PDF_FILE_TOKENS
    from tests.unit.proxy.spend_tracking.test_input_tokens import _one_page_pdf_message

    router: Final = _router_with_groups(("on-openai", "openai/gpt-5.6-sol"), ("on-azure", "azure/gpt-5.6-sol"))
    two_pdfs: Final = [*_one_page_pdf_message(), *_one_page_pdf_message()]

    assert await _reserved_input_tokens(router, "on-openai", "/v1/chat/completions", two_pdfs) == (
        await _reserved_input_tokens(router, "on-azure", "/v1/chat/completions", two_pdfs)
        + 2 * OPENAI_CHAT_PDF_FILE_TOKENS
    )
    assert await _reserved_input_tokens(router, "on-openai", "/v1/messages", two_pdfs) == (
        await _reserved_input_tokens(router, "on-azure", "/v1/messages", two_pdfs)
    )
    assert await _reserved_input_tokens(router, "on-openai", "/v1/chat/completions", _MULTILINGUAL_MESSAGES) == (
        await _reserved_input_tokens(router, "on-azure", "/v1/chat/completions", _MULTILINGUAL_MESSAGES)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("served", ["openai/gpt-5.6-sol", "azure/gpt-5.6-sol"])
async def test_a_gemini_pdf_body_for_an_openai_deployment_reserves_what_the_chat_request_it_becomes_reserves(
    served: str,
) -> None:
    from tests.unit.proxy.spend_tracking.test_input_tokens import _one_page_pdf_base64

    pdf: Final = _one_page_pdf_base64()
    question: Final = "Summarize the report."
    router: Final = _router_with_groups(("pdf-reports", served))
    chat_file: Final = {"type": "file", "file": {"file_data": f"data:application/pdf;base64,{pdf}"}}

    chat: Final = await count_request_input_tokens(
        request_body={
            "model": "pdf-reports",
            "messages": [{"role": "user", "content": [chat_file, {"type": "text", "text": question}]}],
        },
        route="/v1/chat/completions",
        llm_router=router,
        team_id=None,
    )
    gemini: Final = await count_request_input_tokens(
        request_body={
            "contents": [
                {"role": "user", "parts": [{"inlineData": {"mimeType": "application/pdf", "data": pdf}}, {"text": question}]}
            ]
        },
        route="/v1beta/models/pdf-reports:generateContent",
        llm_router=router,
        team_id=None,
    )

    assert chat == gemini
    assert "pdf-reports" in gemini


@pytest.mark.asyncio
async def test_alias_group_of_an_openai_model_reserves_on_that_models_token_count():
    router: Final = _router_with_groups(("gpt-6.1-sol", "openai/gpt-6.1-sol"), ("prod-gpt-6", "openai/gpt-6.1-sol"))

    by_name: Final = await _input_tokens(router, "gpt-6.1-sol")

    assert await _input_tokens(router, "prod-gpt-6") == {"prod-gpt-6": by_name["gpt-6.1-sol"]}


@pytest.mark.asyncio
async def test_model_group_alias_of_an_openai_group_reserves_on_that_groups_token_count():
    router: Final = Router(
        model_list=[
            {"model_name": "gpt-6.1-sol", "litellm_params": {"model": "openai/gpt-6.1-sol", "api_key": "sk-test"}}
        ],
        model_group_alias={"prod": "gpt-6.1-sol"},
    )

    by_name: Final = await _input_tokens(router, "gpt-6.1-sol")

    assert await _input_tokens(router, "prod") == {"prod": by_name["gpt-6.1-sol"]}


@pytest.mark.asyncio
async def test_model_group_alias_of_a_group_no_openai_model_serves_reserves_on_the_alias_as_written():
    router: Final = Router(
        model_list=[
            {
                "model_name": "claude-sonnet-5-5",
                "litellm_params": {"model": "anthropic/claude-sonnet-5-5", "api_key": "sk-test"},
            }
        ],
        model_group_alias={"claude-alias": "claude-sonnet-5-5"},
    )

    assert await _input_tokens(router, "claude-alias") == {
        "claude-alias": litellm.token_counter(model="claude-alias", messages=_MULTILINGUAL_MESSAGES)
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["gpt-6-claude", "o4-alias"])
async def test_release_named_group_or_alias_no_openai_model_serves_reserves_on_an_unknown_names_count(
    name: str,
) -> None:
    router: Final = Router(
        model_list=[
            {"model_name": group, "litellm_params": {"model": "anthropic/claude-sonnet-5-5", "api_key": "sk-test"}}
            for group in ("claude-sonnet-5-5", "gpt-6-claude")
        ],
        model_group_alias={"o4-alias": "claude-sonnet-5-5"},
    )

    assert await _input_tokens(router, name) == {
        name: litellm.token_counter(model="my-model", messages=_MULTILINGUAL_MESSAGES)
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("group", "counted_as"),
    [("gpt-6.1-sol", "my-model"), ("openai/gpt-6.1-sol", "my-model"), ("mixed", "my-model"), ("gpt-4o", "gpt-4o")],
)
async def test_group_mixing_in_a_non_openai_model_reserves_on_its_name_as_written_and_its_openai_models(
    group: str, counted_as: str
):
    served: Final = ("openai/gpt-6.1-sol", "cohere/command-r-plus")
    router: Final = _router_with_groups(*((group, model) for model in served))

    reserved: Final = (await _input_tokens(router, group))[group]

    assert reserved == max(
        litellm.token_counter(model=counted_as, messages=_MULTILINGUAL_MESSAGES),
        litellm.token_counter(model="openai/gpt-6.1-sol", messages=_MULTILINGUAL_MESSAGES),
    )


@pytest.mark.asyncio
async def test_group_reserves_on_its_current_deployments_after_another_vendors_deployment_joins():
    router: Final = _router_with_groups(("prod", "openai/gpt-6.1-sol"))
    openai_only: Final = (await _input_tokens(router, "prod"))["prod"]

    router.add_deployment(
        Deployment(
            model_name="prod",
            litellm_params=LiteLLM_Params(model="anthropic/claude-sonnet-5-5", api_key="sk-test"),
        )
    )

    by_own_name: Final = litellm.token_counter(model="prod", messages=_MULTILINGUAL_MESSAGES)
    assert openai_only != by_own_name
    assert (await _input_tokens(router, "prod"))["prod"] == by_own_name


@pytest.mark.parametrize("window", [{}, {"budget_duration": "1d"}, {"reset_at": "2026-11-03T00:00:00Z"}])
def test_fixed_budget_window_requires_duration_and_reset_boundary(window: object) -> None:
    from litellm.proxy.spend_tracking.budget_reservation import fixed_budget_window

    assert fixed_budget_window(window) is None


GEMINI_MODEL: Final = "gemini-3.8-flash"
GEMINI_ROUTES: Final = (
    f"/vertex_ai/v1/projects/p/locations/global/publishers/google/models/{GEMINI_MODEL}:generateContent",
    f"/gemini/v1beta/models/{GEMINI_MODEL}:generateContent",
    f"/v1beta/models/{GEMINI_MODEL}:generateContent",
)
GEMINI_QUESTION: Final = {"contents": [{"role": "user", "parts": [{"text": "Reply with one word: pong"}]}]}


def _gemini_rate(field: str) -> float:
    rate: Final = litellm.get_model_info(GEMINI_MODEL).get(field)
    assert isinstance(rate, float), f"{GEMINI_MODEL} has no {field} in the cost map"
    return rate


def _gemini_output_rate() -> float:
    return max(_gemini_rate("output_cost_per_token"), _gemini_rate("output_cost_per_reasoning_token"))


def _gemini_window() -> int:
    window: Final = litellm.get_model_info(GEMINI_MODEL).get("max_input_tokens")
    assert isinstance(window, int)
    return window


def _gemini_question_tokens() -> int:
    tokens: Final = count_input_tokens_for_model(request_body=GEMINI_QUESTION, model=GEMINI_MODEL)
    assert tokens is not None
    return tokens


@pytest.mark.parametrize("route", GEMINI_ROUTES)
@pytest.mark.parametrize(
    "config",
    [
        {"generationConfig": {"maxOutputTokens": 64}},
        {"generation_config": {"max_output_tokens": 64}},
        {"config": {"maxOutputTokens": 64}},
    ],
    ids=["generationConfig", "generation_config", "sdk-config"],
)
def test_a_gemini_body_reserves_its_prompt_and_output_cap_not_the_context_window(
    route: str, config: dict[str, object]
) -> None:
    reserved: Final = estimate_request_max_cost(request_body=GEMINI_QUESTION | config, route=route, llm_router=None)

    assert reserved == pytest.approx(
        _gemini_question_tokens() * _gemini_rate("input_cost_per_token") + 64 * _gemini_output_rate()
    )


@pytest.mark.parametrize("openai_cap", ["max_tokens", "max_completion_tokens", "max_output_tokens"])
def test_a_top_level_openai_cap_does_not_shrink_a_gemini_hold_because_gemini_ignores_it(openai_cap: str) -> None:
    body: Final = GEMINI_QUESTION | {openai_cap: 1, "generationConfig": {"maxOutputTokens": 64}}

    reserved: Final = estimate_request_max_cost(request_body=body, route=GEMINI_ROUTES[2], llm_router=None)

    assert reserved == pytest.approx(
        _gemini_question_tokens() * _gemini_rate("input_cost_per_token") + 64 * _gemini_output_rate()
    )


@pytest.mark.parametrize("field", ["candidateCount", "candidate_count"])
def test_a_gemini_candidate_count_reserves_each_candidates_output(field: str) -> None:
    def reserved(candidates: int) -> float:
        body: Final = GEMINI_QUESTION | {"generationConfig": {"maxOutputTokens": 64, field: candidates}}
        cost: Final = estimate_request_max_cost(request_body=body, route=GEMINI_ROUTES[0], llm_router=None)
        assert cost is not None
        return cost

    assert reserved(3) - reserved(1) == pytest.approx(2 * 64 * _gemini_output_rate())


@pytest.mark.parametrize(
    "cached_content",
    [
        {"cachedContent": "cachedContents/ctx-1"},
        {"cached_content": "cachedContents/ctx-1"},
        {"generationConfig": {"maxOutputTokens": 64, "cachedContent": "cachedContents/ctx-1"}},
        {"config": {"maxOutputTokens": 64, "cached_content": "cachedContents/ctx-1"}},
    ],
    ids=["top-level", "top-level-snake", "generation-config", "sdk-config"],
)
def test_a_named_cached_content_holds_and_floors_the_rest_of_the_window_at_the_cache_read_rate(
    cached_content: dict[str, object],
) -> None:
    body: Final = GEMINI_QUESTION | {"generationConfig": {"maxOutputTokens": 64}} | cached_content
    question_tokens: Final = _gemini_question_tokens()

    reserved: Final = estimate_request_max_cost(request_body=body, route=GEMINI_ROUTES[0], llm_router=None)
    floor: Final = estimate_request_input_cost(request_body=body, route=GEMINI_ROUTES[0], llm_router=None)

    assert reserved == pytest.approx(
        question_tokens * _gemini_rate("input_cost_per_token")
        + (_gemini_window() - question_tokens) * _gemini_rate("cache_read_input_token_cost")
        + 64 * _gemini_output_rate()
    )
    assert floor == pytest.approx(
        question_tokens * _gemini_rate("input_cost_per_token")
        + (_gemini_window() - question_tokens) * _gemini_rate("cache_read_input_token_cost")
    )


@pytest.mark.parametrize(
    "unsized",
    [
        {"contents": [{"role": "user", "parts": [{"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}}]}]},
        {"contents": [{"role": "user", "parts": [{"fileData": {"mimeType": "video/mp4", "fileUri": "gs://b/a.mp4"}}]}]},
        {
            "contents": [
                {"role": "user", "parts": [{"fileData": {"mimeType": "application/pdf", "fileUri": "gs://b/r.pdf"}}]}
            ]
        },
        GEMINI_QUESTION | {"tools": [{"googleSearch": {}}]},
    ],
    ids=["inline-image", "file-video", "file-pdf", "google-search"],
)
def test_a_gemini_body_whose_input_google_sizes_on_its_side_reserves_the_whole_window(
    unsized: dict[str, object],
) -> None:
    body: Final = unsized | {"generationConfig": {"maxOutputTokens": 64}}

    reserved: Final = estimate_request_max_cost(request_body=body, route=GEMINI_ROUTES[0], llm_router=None)

    assert reserved == pytest.approx(
        _gemini_window() * _gemini_rate("input_cost_per_token") + 64 * _gemini_output_rate()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("group", ["gpt-4o", "claude-sonnet-5-5"])
async def test_a_gemini_tool_a_gpt_or_claude_named_gemini_group_cannot_count_holds_the_whole_window(
    group: str,
) -> None:
    router: Final = _router_with_groups((group, f"gemini/{GEMINI_MODEL}"))
    body: Final = GEMINI_QUESTION | {"tools": [{"urlContext": {}}], "generationConfig": {"maxOutputTokens": 64}}
    route: Final = f"/v1beta/models/{group}:generateContent"
    counts: Final = await count_request_input_tokens(request_body=body, route=route, llm_router=router, team_id=None)

    reserved: Final = estimate_request_max_cost(
        request_body=body, route=route, llm_router=router, input_token_counts=counts
    )
    floor: Final = estimate_request_input_cost(
        request_body=body, route=route, llm_router=router, input_token_counts=counts
    )

    assert floor == pytest.approx(_gemini_window() * _gemini_rate("input_cost_per_token"))
    assert reserved == pytest.approx(floor + 64 * _gemini_output_rate())


_CACHED_BODY: Final = GEMINI_QUESTION | {
    "cachedContent": "cachedContents/ctx-1",
    "generationConfig": {"maxOutputTokens": 100},
}
_THRESHOLD_PRICED: Final = {
    "input_cost_per_token": 1e-06,
    "input_cost_per_token_above_200k_tokens": 2e-06,
    "cache_read_input_token_cost": 1e-07,
    "cache_read_input_token_cost_above_200k_tokens": 2e-07,
    "output_cost_per_token": 4e-06,
    "output_cost_per_token_above_200k_tokens": 6e-06,
}


@pytest.mark.parametrize(
    ("model_info", "expected"),
    [
        pytest.param(
            {"input_cost_per_token": 1e-06, "output_cost_per_token": 4e-06, "max_input_tokens": 10_000},
            200 * 1e-06 + 9_800 * 1e-06 + 100 * 4e-06,
            id="no-cache-read-price-uses-the-input-rate",
        ),
        pytest.param(
            {"input_cost_per_token": 1e-06, "output_cost_per_token": 4e-06, "cache_read_input_token_cost": 1e-07},
            200 * 1e-06 + 100 * 4e-06,
            id="unknown-window-adds-nothing",
        ),
        pytest.param(
            {
                "max_input_tokens": 10_000,
                "tiered_pricing": [
                    {
                        "range": [0, 1_000],
                        "input_cost_per_token": 1e-06,
                        "cache_read_input_token_cost": 1e-07,
                        "output_cost_per_token": 2e-06,
                    },
                    {
                        "range": [1_000, 10_000],
                        "input_cost_per_token": 3e-06,
                        "cache_read_input_token_cost": 3e-07,
                        "output_cost_per_token": 6e-06,
                    },
                ],
            },
            200 * 3e-06 + 9_800 * 3e-07 + 100 * 6e-06,
            id="tiered-prices-at-the-tier-the-whole-window-falls-in",
        ),
        pytest.param(
            _THRESHOLD_PRICED | {"max_input_tokens": 300_000},
            200 * 2e-06 + 299_800 * 2e-07 + 100 * 6e-06,
            id="a-window-past-a-price-threshold-holds-at-the-higher-rates",
        ),
        pytest.param(
            _THRESHOLD_PRICED | {"max_input_tokens": 150_000},
            200 * 1e-06 + 149_800 * 1e-07 + 100 * 4e-06,
            id="a-window-under-a-price-threshold-keeps-the-base-rates",
        ),
        pytest.param(
            {
                "input_cost_per_token": 1e-06,
                "cache_read_input_token_cost": 1e-07,
                "cache_read_input_audio_token_cost": 3e-07,
                "output_cost_per_token": 4e-06,
                "max_input_tokens": 10_000,
            },
            200 * 1e-06 + 9_800 * 3e-07 + 100 * 4e-06,
            id="a-cache-that-may-hold-audio-is-held-at-the-audio-cache-rate",
        ),
    ],
)
def test_cached_content_reservation_prices_the_rest_of_the_window(
    model_info: dict[str, object], expected: float
) -> None:
    reserved: Final = max_cost_for_cost_info(
        request_body=_CACHED_BODY,
        route=GEMINI_ROUTES[0],
        model_info=model_info,
        requested_output_tokens=100,
        output_multiplier=1,
        input_tokens=200,
    )

    assert reserved == pytest.approx(expected)


def test_a_gemini_prompt_past_a_price_threshold_is_held_at_the_higher_rates() -> None:
    reserved: Final = max_cost_for_cost_info(
        request_body=GEMINI_QUESTION | {"generationConfig": {"maxOutputTokens": 100}},
        route=GEMINI_ROUTES[0],
        model_info=_THRESHOLD_PRICED,
        requested_output_tokens=100,
        output_multiplier=1,
        input_tokens=250_000,
    )

    assert reserved == pytest.approx(250_000 * 2e-06 + 100 * 6e-06)


def test_a_gemini_body_held_at_the_whole_window_it_could_not_count_keeps_the_base_rates() -> None:
    image_body: Final = {
        "contents": [{"role": "user", "parts": [{"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}}]}],
        "generationConfig": {"maxOutputTokens": 100},
    }
    model_info: Final = _THRESHOLD_PRICED | {"max_input_tokens": 300_000}
    cost_inputs: Final = {"request_body": image_body, "route": GEMINI_ROUTES[0], "input_tokens": None}

    reserved: Final = max_cost_for_cost_info(
        **cost_inputs, model_info=model_info, requested_output_tokens=100, output_multiplier=1
    )
    floor: Final = input_cost_for_cost_info(**cost_inputs, model_info=model_info)

    assert reserved == pytest.approx(300_000 * 1e-06 + 100 * 4e-06)
    assert floor == pytest.approx(300_000 * 1e-06)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"generationConfig": {"maxOutputTokens": 4096}},
        {"contents": [{"role": "user", "parts": [{"text": "Summarize the history of Paris. " * 400}]}]},
    ],
    ids=["output-cap", "contents"],
)
async def test_a_gemini_hold_is_sized_from_the_body_extra_body_sends(
    spend_counter_store, overrides: dict[str, object]
) -> None:
    requested: Final = GEMINI_QUESTION | {"generationConfig": {"maxOutputTokens": 1}}
    sent: Final = requested | overrides

    reservation: Final = await reserve_budget_for_request(
        request_body=requested | {"extra_body": overrides},
        route=GEMINI_ROUTES[2],
        llm_router=None,
        valid_token=UserAPIKeyAuth(token="extra-body-key", max_budget=10000),
        team_object=None,
        user_object=None,
        prisma_client=None,
        user_api_key_cache=UserApiKeyCache(),
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
    )

    assert reservation is not None
    assert reservation["reserved_cost"] == pytest.approx(
        estimate_request_max_cost(request_body=sent, route=GEMINI_ROUTES[2], llm_router=None)
    )
    assert reservation["input_cost"] == pytest.approx(
        estimate_request_input_cost(request_body=sent, route=GEMINI_ROUTES[2], llm_router=None)
    )
    assert reservation["reserved_cost"] > (
        estimate_request_max_cost(request_body=requested, route=GEMINI_ROUTES[2], llm_router=None) or 0
    )
    await reconcile_budget_reservation(reservation, actual_cost=0.0)


@pytest.mark.asyncio
async def test_an_extra_body_generation_config_replaces_an_output_cap_sent_as_config(spend_counter_store) -> None:
    override: Final = {"generationConfig": {"maxOutputTokens": 4096}}

    reservation: Final = await reserve_budget_for_request(
        request_body=GEMINI_QUESTION | {"config": {"maxOutputTokens": 1}, "extra_body": override},
        route=GEMINI_ROUTES[2],
        llm_router=None,
        valid_token=UserAPIKeyAuth(token="extra-body-config-key", max_budget=10000),
        team_object=None,
        user_object=None,
        prisma_client=None,
        user_api_key_cache=UserApiKeyCache(),
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
    )

    assert reservation is not None
    assert reservation["reserved_cost"] == pytest.approx(
        estimate_request_max_cost(request_body=GEMINI_QUESTION | override, route=GEMINI_ROUTES[2], llm_router=None)
    )
    await reconcile_budget_reservation(reservation, actual_cost=0.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"generationConfig": {"maxOutputTokens": 1}},
        {"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]},
    ],
    ids=["output-cap", "contents"],
)
async def test_an_extra_body_that_shrinks_a_gemini_body_keeps_the_hold_of_the_body_an_adapter_sends(
    spend_counter_store, overrides: dict[str, object]
) -> None:
    requested: Final = {
        "contents": [{"role": "user", "parts": [{"text": "Summarize the history of Paris. " * 400}]}],
        "generationConfig": {"maxOutputTokens": 4096},
    }

    reservation: Final = await reserve_budget_for_request(
        request_body=requested | {"extra_body": overrides},
        route=GEMINI_ROUTES[2],
        llm_router=None,
        valid_token=UserAPIKeyAuth(token="extra-body-shrink-key", max_budget=10000),
        team_object=None,
        user_object=None,
        prisma_client=None,
        user_api_key_cache=UserApiKeyCache(),
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
    )

    assert reservation is not None
    assert reservation["reserved_cost"] == pytest.approx(
        estimate_request_max_cost(request_body=requested, route=GEMINI_ROUTES[2], llm_router=None)
    )
    assert reservation["input_cost"] == pytest.approx(
        estimate_request_input_cost(request_body=requested, route=GEMINI_ROUTES[2], llm_router=None)
    )
    await reconcile_budget_reservation(reservation, actual_cost=0.0)


@pytest.mark.parametrize(("small", "large"), [("generationConfig", "generation_config"), ("config", "generationConfig")])
def test_a_gemini_body_sending_its_config_under_two_names_is_held_at_the_larger_output_cap(
    small: str, large: str
) -> None:
    body: Final = GEMINI_QUESTION | {small: {"maxOutputTokens": 1}, large: {"maxOutputTokens": 4096}}

    reserved: Final = estimate_request_max_cost(request_body=body, route=GEMINI_ROUTES[0], llm_router=None)

    assert reserved == pytest.approx(
        estimate_request_max_cost(
            request_body=GEMINI_QUESTION | {large: {"maxOutputTokens": 4096}}, route=GEMINI_ROUTES[0], llm_router=None
        )
    )


@pytest.mark.parametrize(("uncapped", "capped"), [("config", "generationConfig"), ("generationConfig", "generation_config")])
def test_a_gemini_config_name_without_an_output_cap_keeps_the_uncapped_hold(uncapped: str, capped: str) -> None:
    uncapped_body: Final = GEMINI_QUESTION | {uncapped: {"temperature": 0.5}}

    reserved: Final = estimate_request_max_cost(
        request_body=uncapped_body | {capped: {"maxOutputTokens": 1}}, route=GEMINI_ROUTES[0], llm_router=None
    )

    assert reserved == pytest.approx(
        estimate_request_max_cost(request_body=uncapped_body, route=GEMINI_ROUTES[0], llm_router=None)
    )
    assert reserved > estimate_request_max_cost(
        request_body=GEMINI_QUESTION | {capped: {"maxOutputTokens": 1}}, route=GEMINI_ROUTES[0], llm_router=None
    )


@pytest.mark.parametrize(("uncapped", "capped"), [("generationConfig", "config"), ("generation_config", "generationConfig")])
def test_a_gemini_config_name_without_an_output_cap_keeps_a_larger_cap_from_another_name(
    uncapped: str, capped: str
) -> None:
    capped_body: Final = GEMINI_QUESTION | {capped: {"maxOutputTokens": 60000}}

    reserved: Final = estimate_request_max_cost(
        request_body=capped_body | {uncapped: {"temperature": 0.5}}, route=GEMINI_ROUTES[0], llm_router=None
    )

    assert reserved == pytest.approx(
        estimate_request_max_cost(request_body=capped_body, route=GEMINI_ROUTES[0], llm_router=None)
    )


@pytest.mark.parametrize(
    ("request_body", "input_tokens", "expected"),
    [
        pytest.param(_CACHED_BODY, 200, 200 * 2e-06 + 299_800 * 2e-07, id="a-cached-window-past-a-price-threshold"),
        pytest.param(GEMINI_QUESTION, 250_000, 250_000 * 2e-06, id="a-prompt-past-a-price-threshold"),
        pytest.param(GEMINI_QUESTION, 150_000, 150_000 * 1e-06, id="a-prompt-under-a-price-threshold"),
    ],
)
def test_the_input_floor_prices_a_gemini_prompt_at_the_rates_its_size_bills_at(
    request_body: dict[str, object], input_tokens: int, expected: float
) -> None:
    floor: Final = input_cost_for_cost_info(
        request_body=request_body,
        route=GEMINI_ROUTES[0],
        model_info=_THRESHOLD_PRICED | {"max_input_tokens": 300_000},
        input_tokens=input_tokens,
    )

    assert floor == pytest.approx(expected)


def test_a_routed_gemini_deployment_prices_its_cached_window_at_its_above_threshold_rates() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": GEMINI_MODEL,
                "litellm_params": {"model": "gemini/threshold-priced-gemini", "api_key": "k"},
                "model_info": {"id": "threshold-priced-gemini", **_THRESHOLD_PRICED, "max_input_tokens": 300_000},
            }
        ]
    )
    counts: Final = {GEMINI_MODEL: 200}

    reserved: Final = estimate_request_max_cost(
        request_body=_CACHED_BODY, route=GEMINI_ROUTES[2], llm_router=router, input_token_counts=counts
    )
    floor: Final = estimate_request_input_cost(
        request_body=_CACHED_BODY, route=GEMINI_ROUTES[2], llm_router=router, input_token_counts=counts
    )

    assert floor == pytest.approx(200 * 2e-06 + 299_800 * 2e-07)
    assert reserved == pytest.approx(200 * 2e-06 + 299_800 * 2e-07 + 100 * 6e-06)


def test_cached_content_without_an_input_or_cache_read_price_cannot_be_reserved() -> None:
    reserved: Final = max_cost_for_cost_info(
        request_body=_CACHED_BODY,
        route=GEMINI_ROUTES[0],
        model_info={"output_cost_per_token": 4e-06, "max_input_tokens": 10_000},
        requested_output_tokens=100,
        output_multiplier=1,
        input_tokens=0,
    )

    assert reserved is None


def _cache_priced_gemini_deployment(deployment_id: str, cache_read_cost: float) -> dict[str, object]:
    return Deployment(
        model_name="custom-gemini",
        litellm_params=LiteLLM_Params(
            model=f"vertex_ai/{GEMINI_MODEL}",
            vertex_project="p",
            vertex_location="global",
            input_cost_per_token=1e-06,
            output_cost_per_token=4e-06,
            cache_read_input_token_cost=cache_read_cost,
        ),
        model_info=ModelInfo(id=deployment_id, max_input_tokens=10_000),
    ).model_dump()


def test_cached_content_through_a_router_is_held_at_the_highest_deployment_cache_read_rate() -> None:
    router: Final = Router(
        model_list=[_cache_priced_gemini_deployment("cheap", 1e-07), _cache_priced_gemini_deployment("dear", 3e-07)]
    )

    reserved: Final = estimate_request_max_cost(
        request_body=_CACHED_BODY,
        route=GEMINI_ROUTES[0].replace(GEMINI_MODEL, "custom-gemini"),
        llm_router=router,
        input_token_counts={"custom-gemini": 200},
    )

    assert reserved == pytest.approx(200 * 1e-06 + 9_800 * 3e-07 + 100 * 4e-06)


def test_cached_content_through_a_router_is_held_at_a_deployments_audio_cache_read_rate() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "custom-gemini",
                "litellm_params": {"model": "gemini/audio-cache-priced-gemini", "api_key": "k"},
                "model_info": {
                    "id": "audio-cache-priced-gemini",
                    "input_cost_per_token": 1e-06,
                    "output_cost_per_token": 4e-06,
                    "cache_read_input_token_cost": 1e-07,
                    "cache_read_input_audio_token_cost": 3e-07,
                    "max_input_tokens": 10_000,
                },
            }
        ]
    )

    reserved: Final = estimate_request_max_cost(
        request_body=_CACHED_BODY,
        route=GEMINI_ROUTES[2].replace(GEMINI_MODEL, "custom-gemini"),
        llm_router=router,
        input_token_counts={"custom-gemini": 200},
    )

    assert reserved == pytest.approx(200 * 1e-06 + 9_800 * 3e-07 + 100 * 4e-06)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", GEMINI_ROUTES)
async def test_a_gemini_body_is_counted_for_the_reservation_from_its_raw_body(route: str) -> None:
    counts: Final = await count_request_input_tokens(
        request_body=GEMINI_QUESTION,
        route=route,
        llm_router=None,
        team_id=None,
        raw_body=json.dumps(GEMINI_QUESTION).encode(),
    )

    assert counts == {GEMINI_MODEL: _gemini_question_tokens()}
    assert counts[GEMINI_MODEL] > 0


_WAV_AUDIO: Final = {"mimeType": "audio/wav", "data": "UklGRg=="}


@pytest.mark.parametrize(
    "audio_part",
    [
        pytest.param({"inlineData": _WAV_AUDIO}, id="inline-data"),
        pytest.param({"inline_data": {"mime_type": "audio/ogg", "data": "T2dnUw=="}}, id="snake-case"),
        pytest.param(
            {"functionResponse": {"name": "record", "response": {}, "parts": [{"inlineData": _WAV_AUDIO}]}},
            id="in-a-function-response",
        ),
    ],
)
@pytest.mark.parametrize(
    ("model_info", "text_rate"),
    [
        pytest.param(
            {"input_cost_per_token": 1e-06, "input_cost_per_audio_token": 3e-06, "output_cost_per_token": 4e-06},
            1e-06,
            id="flat",
        ),
        pytest.param(
            {
                "tiered_pricing": [
                    {
                        "range": [0, 10_000],
                        "input_cost_per_token": 1e-06,
                        "input_cost_per_audio_token": 3e-06,
                        "output_cost_per_token": 4e-06,
                    }
                ]
            },
            1e-06,
            id="tiered",
        ),
        pytest.param(
            {"input_cost_per_token": 3e-06, "input_cost_per_audio_token": 1e-06, "output_cost_per_token": 4e-06},
            3e-06,
            id="audio-cheaper-than-text",
        ),
    ],
)
def test_a_gemini_body_with_inline_audio_holds_its_input_at_the_higher_of_the_text_and_audio_rates(
    audio_part: dict[str, object], model_info: dict[str, object], text_rate: float
) -> None:
    def held(parts: list[object]) -> tuple[float | None, float | None]:
        body: Final = {"contents": [{"role": "user", "parts": parts}], "generationConfig": {"maxOutputTokens": 100}}
        pricing: Final = {"request_body": body, "route": GEMINI_ROUTES[0]}
        return (
            max_cost_for_cost_info(
                **pricing, model_info=model_info, requested_output_tokens=100, output_multiplier=1, input_tokens=200
            ),
            input_cost_for_cost_info(**pricing, model_info=model_info, input_tokens=200),
        )

    assert held([{"text": "Transcribe this."}, audio_part]) == (
        pytest.approx(200 * 3e-06 + 100 * 4e-06),
        pytest.approx(200 * 3e-06),
    )
    assert held(["Transcribe this."]) == (
        pytest.approx(200 * text_rate + 100 * 4e-06),
        pytest.approx(200 * text_rate),
    )


def test_a_gemini_audio_hold_uses_the_cost_maps_audio_rate() -> None:
    model: Final = "gemini-2.5-flash"
    info: Final = litellm.get_model_info(model)
    audio_rate: Final = info.get("input_cost_per_audio_token")
    text_rate: Final = info.get("input_cost_per_token")
    assert isinstance(audio_rate, float) and isinstance(text_rate, float) and audio_rate > text_rate
    body: Final = {
        "contents": [{"role": "user", "parts": [{"text": "Transcribe this."}, {"inlineData": _WAV_AUDIO}]}],
        "generationConfig": {"maxOutputTokens": 64},
    }
    route: Final = f"/v1beta/models/{model}:generateContent"

    reserved: Final = estimate_request_max_cost(
        request_body=body, route=route, llm_router=None, input_token_counts={model: 200}
    )
    floor: Final = estimate_request_input_cost(
        request_body=body, route=route, llm_router=None, input_token_counts={model: 200}
    )

    output_rate: Final = max(
        rate
        for rate in (info.get("output_cost_per_token"), info.get("output_cost_per_reasoning_token"))
        if isinstance(rate, float)
    )
    assert reserved == pytest.approx(200 * audio_rate + 64 * output_rate)
    assert floor == pytest.approx(200 * audio_rate)


def _wav_base64(seconds: int) -> str:
    buffer: Final = io.BytesIO()
    with wave.open(buffer, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(16_000)
        recording.writeframes(b"\x00\x00" * 16_000 * seconds)
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.asyncio
async def test_inline_audio_sent_to_an_alias_counts_at_the_rate_of_the_gemini_model_it_serves() -> None:
    served: Final = "gemini/gemini-2.5-flash"
    router: Final = Router(model_list=[{"model_name": "voice-notes", "litellm_params": {"model": served, "api_key": "k"}}])
    body: Final = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Transcribe this."}, {"inlineData": {"mimeType": "audio/wav", "data": _wav_base64(4)}}],
            }
        ]
    }

    counts: Final = await count_request_input_tokens(
        request_body=body,
        route="/v1beta/models/voice-notes:generateContent",
        llm_router=router,
        team_id=None,
        raw_body=json.dumps(body).encode(),
    )

    assert counts == {"voice-notes": count_input_tokens_for_model(request_body=body, model=served)}


@pytest.mark.parametrize(
    ("route", "body"),
    [
        ("/v1/chat/completions", {"model": GEMINI_MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}),
        ("/v1/responses", {"model": GEMINI_MODEL, "input": "hi", "max_output_tokens": 64}),
    ],
    ids=["chat", "responses"],
)
@pytest.mark.parametrize("field", ["cached_content", "cachedContent"])
def test_a_cached_content_name_on_a_chat_or_responses_body_leaves_the_hold_unchanged(
    route: str, body: dict[str, object], field: str
) -> None:
    plain: Final = estimate_request_max_cost(request_body=body, route=route, llm_router=None)

    named: Final = estimate_request_max_cost(
        request_body=body | {field: "cachedContents/ctx-1"}, route=route, llm_router=None
    )

    assert plain is not None
    assert named == pytest.approx(plain)


def _priced_responses_router() -> Router:
    return Router(
        model_list=[
            Deployment(
                model_name="priced-model",
                litellm_params=LiteLLM_Params(model="openai/priced-model", api_key="sk-fake"),
                model_info=ModelInfo(
                    id="priced-model-deployment",
                    input_cost_per_token=0.0001,
                    output_cost_per_token=0.0002,
                    max_output_tokens=128000,
                ),
            ).model_dump()
        ]
    )


@pytest.mark.parametrize("route", ["/v1/responses", "/responses", "/openai/v1/responses", "/v1/responses/compact"])
@pytest.mark.parametrize(
    "spoofed_fields",
    [
        {"max_tokens": 1, "max_output_tokens": 1000},
        {"max_completion_tokens": 1, "max_output_tokens": 1000},
        {"max_completion_tokens": 1, "max_tokens": 1, "max_output_tokens": 1000},
    ],
)
def test_responses_routes_ignore_smaller_unforwarded_caps_when_max_output_tokens_set(
    route: str,
    spoofed_fields: dict[str, int],
) -> None:
    router: Final = _priced_responses_router()
    baseline: Final = {"model": "priced-model", "input": "hi", "max_output_tokens": 1000}
    spoofed: Final = {"model": "priced-model", "input": "hi", **spoofed_fields}

    expected_cost: Final = estimate_request_max_cost(request_body=baseline, route=route, llm_router=router)
    actual_cost: Final = estimate_request_max_cost(request_body=spoofed, route=route, llm_router=router)

    assert expected_cost is not None
    assert actual_cost == pytest.approx(expected_cost)


@pytest.mark.parametrize("route", ["/v1/responses", "/v1/chat/completions", "/v1/messages"])
@pytest.mark.parametrize(
    "invalid_cap_body",
    [
        {"max_output_tokens": -100},
        {"max_output_tokens": False},
        {"max_output_tokens": True},
        {"max_tokens": -50},
        {"max_tokens": float("inf")},
        {"max_tokens": "not-a-number"},
        {"max_completion_tokens": False},
    ],
)
def test_invalid_or_unforwarded_output_caps_never_under_reserve_below_default(
    route: str,
    invalid_cap_body: dict[str, object],
) -> None:
    router: Final = _priced_responses_router()
    input_key: Final = "input" if "responses" in route else "messages"
    input_val: Final = "hi" if "responses" in route else [{"role": "user", "content": "hi"}]
    uncapped: Final[dict[str, object]] = {"model": "priced-model", input_key: input_val}
    candidate: Final[dict[str, object]] = {"model": "priced-model", input_key: input_val, **invalid_cap_body}

    uncapped_cost: Final = estimate_request_max_cost(request_body=uncapped, route=route, llm_router=router)
    candidate_cost: Final = estimate_request_max_cost(request_body=candidate, route=route, llm_router=router)

    assert uncapped_cost is not None
    assert candidate_cost == pytest.approx(uncapped_cost)


@pytest.mark.parametrize(
    ("route", "decoy_body"),
    [
        ("/v1/responses", {"model": "priced-model", "input": "hi", "max_tokens": 1}),
        ("/v1/responses", {"model": "priced-model", "input": "hi", "max_completion_tokens": 1}),
        (
            "/v1/chat/completions",
            {
                "model": "priced-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_output_tokens": 1,
            },
        ),
        (
            "/v1/messages",
            {
                "model": "priced-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_output_tokens": 1,
            },
        ),
    ],
)
def test_unforwarded_only_output_cap_uses_default_fallback(
    route: str,
    decoy_body: dict[str, object],
) -> None:
    router: Final = _priced_responses_router()
    uncapped: Final = {key: value for key, value in decoy_body.items() if key in ("model", "input", "messages")}

    uncapped_cost: Final = estimate_request_max_cost(request_body=uncapped, route=route, llm_router=router)
    decoy_cost: Final = estimate_request_max_cost(request_body=decoy_body, route=route, llm_router=router)

    assert uncapped_cost is not None
    assert decoy_cost == pytest.approx(uncapped_cost)


def test_responses_explicit_zero_output_cap_is_preserved() -> None:
    router: Final = _priced_responses_router()
    zero_cost: Final = estimate_request_max_cost(
        request_body={"model": "priced-model", "input": "hi", "max_output_tokens": 0},
        route="/v1/responses",
        llm_router=router,
    )
    one_token_cost: Final = estimate_request_max_cost(
        request_body={"model": "priced-model", "input": "hi", "max_output_tokens": 1},
        route="/v1/responses",
        llm_router=router,
    )
    assert zero_cost is not None and one_token_cost is not None
    assert 0.0 < zero_cost < one_token_cost


@pytest.mark.parametrize(
    ("route", "spoofed_body", "baseline_body"),
    [
        (
            "/v1/messages",
            {
                "model": "priced-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 1,
                "max_output_tokens": 1,
                "max_tokens": 1000,
            },
            {"model": "priced-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1000},
        ),
        (
            "/anthropic/v1/messages",
            {
                "model": "priced-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 1,
                "max_tokens": 1000,
            },
            {"model": "priced-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1000},
        ),
        (
            "/v1/chat/completions",
            {
                "model": "priced-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_output_tokens": 1,
                "max_completion_tokens": 1000,
            },
            {"model": "priced-model", "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 1000},
        ),
        (
            "/v1/chat/completions",
            {
                "model": "priced-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 1,
                "max_tokens": 1000,
            },
            {"model": "priced-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1000},
        ),
        (
            "/v1/chat/completions",
            {
                "model": "priced-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "maxTokens": 64000,
            },
            {"model": "priced-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64000},
        ),
        (
            "/v1/messages",
            {
                "model": "priced-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "max_output_tokens": 64000,
            },
            {"model": "priced-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64000},
        ),
        (
            "/v1/messages",
            {
                "model": "priced-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "maxTokens": 64000,
            },
            {"model": "priced-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64000},
        ),
        (
            "/v1/responses",
            {"model": "priced-model", "input": "hi", "max_output_tokens": 1, "maxTokens": 64000},
            {"model": "priced-model", "input": "hi", "max_output_tokens": 64000},
        ),
    ],
)
def test_chat_and_messages_routes_prevent_output_cap_shadowing(
    route: str,
    spoofed_body: dict[str, object],
    baseline_body: dict[str, object],
) -> None:
    router: Final = _priced_responses_router()
    expected: Final = estimate_request_max_cost(request_body=baseline_body, route=route, llm_router=router)
    actual: Final = estimate_request_max_cost(request_body=spoofed_body, route=route, llm_router=router)
    assert expected is not None
    assert actual == pytest.approx(expected)


def _router_with_configured_caps(config_source: str, caps: Mapping[str, int]) -> Router:
    deployment_caps: Final = caps if config_source == "deployment" else {}
    return Router(
        model_list=[
            Deployment(
                model_name="priced-model",
                litellm_params=LiteLLM_Params(model="anthropic/claude-haiku-5-5", api_key="sk-fake", **deployment_caps),
                model_info=ModelInfo(
                    id="configured-cap-deployment",
                    input_cost_per_token=0.0001,
                    output_cost_per_token=0.0002,
                    max_output_tokens=128000,
                ),
            ).model_dump()
        ],
        default_litellm_params=dict(caps) if config_source == "router_default" else None,
    )


@pytest.mark.parametrize("config_source", ["deployment", "router_default"])
def test_messages_decoy_cap_never_holds_less_than_configured_max_tokens(config_source: str) -> None:
    router: Final = _router_with_configured_caps(config_source, {"max_tokens": 1000})
    base_body: Final = {"model": "priced-model", "messages": [{"role": "user", "content": "hi"}]}

    configured: Final = estimate_request_max_cost(
        request_body={**base_body, "max_tokens": 1000},
        route="/v1/messages",
        llm_router=router,
    )
    completion_cap_only: Final = estimate_request_max_cost(
        request_body={**base_body, "max_completion_tokens": 1},
        route="/v1/messages",
        llm_router=router,
    )
    request_override: Final = estimate_request_max_cost(
        request_body={**base_body, "max_tokens": 5},
        route="/v1/messages",
        llm_router=router,
    )

    assert configured is not None and completion_cap_only is not None and request_override is not None
    assert completion_cap_only >= configured
    assert request_override < configured


@pytest.mark.parametrize(
    "route", ["/v1/messages", "/v1/chat/completions", "/v1beta/models/priced-model:generateContent"]
)
@pytest.mark.parametrize("config_source", ["deployment", "router_default"])
@pytest.mark.parametrize("configured_field", ["max_tokens", "max_completion_tokens"])
def test_configured_caps_never_shrink_an_uncapped_hold(route: str, config_source: str, configured_field: str) -> None:
    body: Final = {"model": "priced-model", "messages": [{"role": "user", "content": "hi"}]}

    uncapped: Final = estimate_request_max_cost(
        request_body=body,
        route=route,
        llm_router=_router_with_configured_caps(config_source, {}),
    )
    configured: Final = estimate_request_max_cost(
        request_body=body,
        route=route,
        llm_router=_router_with_configured_caps(config_source, {configured_field: 1}),
    )

    assert uncapped is not None
    assert configured == pytest.approx(uncapped)


@pytest.mark.parametrize("config_source", ["deployment", "router_default"])
@pytest.mark.parametrize("field", ["n", "best_of"])
def test_a_configured_n_holds_like_a_requested_n_unless_the_request_sets_its_own(
    config_source: str, field: str
) -> None:
    body: Final = {"model": "priced-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1000}
    plain: Final = _router_with_configured_caps(config_source, {})
    configured: Final = _router_with_configured_caps(config_source, {field: 4})
    single, requested, configured_hold, request_override = (
        estimate_request_max_cost(request_body=request_body, route="/v1/chat/completions", llm_router=llm_router)
        for request_body, llm_router in (
            (body, plain),
            ({**body, field: 4}, plain),
            (body, configured),
            ({**body, field: 1}, configured),
        )
    )

    assert single is not None and requested is not None
    assert requested > single
    assert configured_hold == pytest.approx(requested)
    assert request_override == pytest.approx(single)


def _priced_router(
    backend_models: tuple[str, ...],
    additional_drop_params: list[str] | None = None,
    default_litellm_params: dict[str, list[str]] | None = None,
) -> Router:
    return Router(
        default_litellm_params=default_litellm_params,
        model_list=[
            Deployment(
                model_name="priced-model",
                litellm_params=LiteLLM_Params(
                    model=backend_model, api_key="sk-fake", additional_drop_params=additional_drop_params
                ),
                model_info=ModelInfo(
                    id=f"deployment-{index}",
                    input_cost_per_token=0.0001,
                    output_cost_per_token=0.0002,
                    max_output_tokens=128000,
                ),
            ).model_dump()
            for index, backend_model in enumerate(backend_models)
        ],
    )


def _chat_estimates(llm_router: Router | None, model: str) -> tuple[float | None, float | None, float | None]:
    body: Final = {"model": model, "messages": [{"role": "user", "content": "hi"}]}
    return tuple(
        estimate_request_max_cost(request_body={**body, **cap}, route="/v1/chat/completions", llm_router=llm_router)
        for cap in ({}, {"max_completion_tokens": 1}, {"max_tokens": 1})
    )


@pytest.mark.parametrize(
    "backend_models",
    [
        ("replicate/meta/llama-3-70b-instruct",),
        ("watsonx/ibm/granite-3-8b-instruct",),
        ("anthropic/claude-haiku-5-5", "replicate/meta/llama-3-70b-instruct"),
    ],
)
def test_chat_max_completion_tokens_cannot_lower_the_hold_when_a_deployment_drops_it(
    backend_models: tuple[str, ...],
) -> None:
    uncapped, completion_capped, max_tokens_capped = _chat_estimates(_priced_router(backend_models), "priced-model")

    assert uncapped is not None and max_tokens_capped is not None
    assert max_tokens_capped < uncapped
    assert completion_capped == pytest.approx(uncapped)


@pytest.mark.parametrize(
    "backend_models",
    [("openai/gpt-5.6",), ("anthropic/claude-haiku-5-5", "openai/gpt-5.6")],
)
def test_chat_max_completion_tokens_lowers_the_hold_when_every_deployment_supports_it(
    backend_models: tuple[str, ...],
) -> None:
    uncapped, completion_capped, max_tokens_capped = _chat_estimates(_priced_router(backend_models), "priced-model")

    assert uncapped is not None and completion_capped is not None
    assert completion_capped < uncapped
    assert completion_capped == pytest.approx(max_tokens_capped)


@pytest.mark.parametrize(
    "model, completion_cap_lowers_hold",
    [("replicate/meta/llama-3-70b-instruct", False), ("gpt-5.6", True)],
)
def test_chat_max_completion_tokens_without_router_follows_the_model_supported_params(
    model: str, completion_cap_lowers_hold: bool
) -> None:
    uncapped, completion_capped, _ = _chat_estimates(None, model)

    assert uncapped is not None and completion_capped is not None
    assert (completion_capped < uncapped) is completion_cap_lowers_hold


@pytest.mark.parametrize(
    "backend_models",
    [("v0/v0-1.0-md",), ("anthropic/claude-haiku-5-5", "v0/v0-1.0-md")],
)
def test_chat_max_tokens_cannot_lower_the_hold_when_a_deployment_drops_it(backend_models: tuple[str, ...]) -> None:
    uncapped, completion_capped, max_tokens_capped = _chat_estimates(_priced_router(backend_models), "priced-model")

    assert uncapped is not None
    assert max_tokens_capped == pytest.approx(uncapped)
    assert completion_capped == pytest.approx(uncapped)


@pytest.mark.parametrize(
    ("backend_models", "cap", "lowers_hold"),
    [
        (("gemini/gemini-2.5-pro",), {"max_output_tokens": 1}, True),
        (("vertex_ai/gemini-2.5-pro",), {"max_output_tokens": 1}, True),
        (("bedrock/converse/anthropic.claude-haiku-5-5",), {"maxTokens": 1}, True),
        (("openai/gpt-5.6",), {"max_output_tokens": 1}, False),
        (("anthropic/claude-haiku-5-5",), {"max_output_tokens": 1}, False),
        (("gemini/gemini-2.5-pro", "openai/gpt-5.6"), {"max_output_tokens": 1}, False),
        (("bedrock/converse/anthropic.claude-haiku-5-5",), {"inferenceConfig": {"maxTokens": 1}}, False),
    ],
)
def test_chat_provider_native_cap_lowers_the_hold_only_when_every_deployment_reads_it(
    backend_models: tuple[str, ...], cap: Mapping[str, object], lowers_hold: bool
) -> None:
    router: Final = _priced_router(backend_models)
    body: Final = {"model": "priced-model", "messages": [{"role": "user", "content": "hi"}]}
    uncapped: Final = estimate_request_max_cost(request_body=body, route="/v1/chat/completions", llm_router=router)
    capped: Final = estimate_request_max_cost(
        request_body={**body, **cap}, route="/v1/chat/completions", llm_router=router
    )

    assert uncapped is not None and capped is not None
    assert (capped < uncapped) is lowers_hold


@pytest.mark.parametrize(
    ("backend_model", "cap"),
    [
        ("gemini/gemini-2.5-pro", "max_output_tokens"),
        ("openai/gpt-5.6", "max_tokens"),
        ("openai/gpt-5.6", "max_completion_tokens"),
    ],
)
@pytest.mark.parametrize("dropped_by", ["request", "deployment", "router_default"])
def test_a_cap_named_in_additional_drop_params_cannot_lower_the_hold(
    backend_model: str, cap: str, dropped_by: str
) -> None:
    body: Final = {"model": "priced-model", "messages": [{"role": "user", "content": "hi"}], cap: 1}
    kept_router: Final = _priced_router((backend_model,))
    dropping_router: Final = _priced_router(
        (backend_model,),
        additional_drop_params=[cap] if dropped_by == "deployment" else None,
        default_litellm_params={"additional_drop_params": [cap]} if dropped_by == "router_default" else None,
    )
    dropping_body: Final = {**body, "additional_drop_params": [cap]} if dropped_by == "request" else body
    uncapped: Final = estimate_request_max_cost(
        request_body={"model": "priced-model", "messages": body["messages"]},
        route="/v1/chat/completions",
        llm_router=kept_router,
    )
    kept: Final = estimate_request_max_cost(request_body=body, route="/v1/chat/completions", llm_router=kept_router)
    dropped: Final = estimate_request_max_cost(
        request_body=dropping_body, route="/v1/chat/completions", llm_router=dropping_router
    )

    assert uncapped is not None and kept is not None
    assert kept < uncapped
    assert dropped == pytest.approx(uncapped)


@pytest.mark.parametrize("max_tokens", [1, 65536])
@pytest.mark.parametrize(
    ("deployment_drops", "router_drops", "request_drops"),
    [
        (None, ["max_tokens"], []),
        (["max_tokens"], None, []),
        (["max_tokens"], [], None),
    ],
)
def test_a_cap_counts_when_the_drop_list_the_router_sends_keeps_it(
    max_tokens: int,
    deployment_drops: list[str] | None,
    router_drops: list[str] | None,
    request_drops: list[str] | None,
) -> None:
    body: Final = {"model": "priced-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": max_tokens}
    plain: Final = estimate_request_max_cost(
        request_body=body, route="/v1/chat/completions", llm_router=_priced_router(("openai/gpt-5.6",))
    )
    overridden: Final = estimate_request_max_cost(
        request_body=body if request_drops is None else {**body, "additional_drop_params": request_drops},
        route="/v1/chat/completions",
        llm_router=_priced_router(
            ("openai/gpt-5.6",),
            additional_drop_params=deployment_drops,
            default_litellm_params=None if router_drops is None else {"additional_drop_params": router_drops},
        ),
    )

    assert plain is not None
    assert overridden == pytest.approx(plain)


def test_custom_handler_deployment_trusts_only_the_chat_cap_it_receives(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "provider_list", list(litellm.provider_list))
    monkeypatch.setattr(litellm, "_custom_providers", list(litellm._custom_providers))
    monkeypatch.setattr(
        litellm, "custom_provider_map", [{"provider": "budget-hold-custom", "custom_handler": litellm.CustomLLM()}]
    )
    uncapped, completion_capped, max_tokens_capped = _chat_estimates(
        _priced_router(("budget-hold-custom/priced-model",)), "priced-model"
    )
    max_output_capped: Final = estimate_request_max_cost(
        request_body={"model": "priced-model", "messages": [{"role": "user", "content": "hi"}], "max_output_tokens": 1},
        route="/v1/chat/completions",
        llm_router=_priced_router(("budget-hold-custom/priced-model",)),
    )

    assert uncapped is not None and max_tokens_capped is not None
    assert max_tokens_capped < uncapped
    assert completion_capped == pytest.approx(uncapped)
    assert max_output_capped == pytest.approx(uncapped)


@pytest.mark.parametrize("cap", [{"max_output_tokens": 1}, {"maxTokens": 1}, {"inferenceConfig": {"maxTokens": 1}}])
def test_a_provider_that_rejects_the_cap_probe_keeps_the_uncapped_hold(cap: dict[str, object]) -> None:
    router: Final = _priced_router(("scaledown/extract",))
    body: Final = {"model": "priced-model", "messages": [{"role": "user", "content": "hi"}]}
    uncapped: Final = estimate_request_max_cost(request_body=body, route="/v1/chat/completions", llm_router=router)
    capped: Final = estimate_request_max_cost(
        request_body={**body, **cap}, route="/v1/chat/completions", llm_router=router
    )

    assert uncapped is not None
    assert capped == pytest.approx(uncapped)


@pytest.mark.parametrize("ceiling", [0, -1])
def test_non_positive_model_ceiling_reserves_the_default_output_cap(ceiling: int) -> None:
    body: Final = {"model": "priced-model", "messages": [{"role": "user", "content": "hi"}]}

    def estimate(max_output_tokens: int) -> float | None:
        return estimate_request_max_cost(
            request_body=body,
            route="/v1/chat/completions",
            llm_router=_router_for_backend("openai/gpt-5.6", max_output_tokens=max_output_tokens),
        )

    assert estimate(ceiling) == pytest.approx(estimate(DEFAULT_MAX_OUTPUT_TOKENS_FALLBACK))


def test_unpriced_model_reserves_nothing_so_read_time_enforcement_applies() -> None:
    body: Final = {"model": "unpriced-model-for-reservation", "messages": [{"role": "user", "content": "hi"}]}

    assert estimate_request_max_cost(request_body=body, route="/v1/chat/completions", llm_router=None) is None
    assert estimate_request_input_cost(request_body=body, route="/v1/chat/completions", llm_router=None) is None


def test_bedrock_converse_inference_config_max_tokens_scales_cost() -> None:
    router: Final = _priced_responses_router()
    route: Final = "/bedrock/model/priced-model/converse"
    low: Final = estimate_request_max_cost(
        request_body={
            "model": "priced-model",
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "inferenceConfig": {"maxTokens": 100},
        },
        route=route,
        llm_router=router,
    )
    high: Final = estimate_request_max_cost(
        request_body={
            "model": "priced-model",
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "inferenceConfig": {"maxTokens": 1000},
        },
        route=route,
        llm_router=router,
    )
    assert low is not None and high is not None
    assert high > low


@pytest.mark.parametrize(
    ("route", "body", "inference_config_bounds_hold"),
    [
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, False),
        ("/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}, False),
        ("/v1/messages", {"messages": [{"role": "user", "content": "hi"}]}, False),
        ("/v1/responses", {"input": "hi"}, False),
        ("/bedrock/model/priced-model/converse", {"messages": [{"role": "user", "content": [{"text": "hi"}]}]}, True),
        (
            "/bedrock/model/priced-model/converse-stream",
            {"messages": [{"role": "user", "content": [{"text": "hi"}]}]},
            True,
        ),
    ],
)
def test_inference_config_max_tokens_bounds_the_hold_only_on_bedrock_converse(
    route: str,
    body: dict[str, object],
    inference_config_bounds_hold: bool,
) -> None:
    router: Final = _priced_responses_router()
    uncapped: Final = estimate_request_max_cost(
        request_body={"model": "priced-model", **body},
        route=route,
        llm_router=router,
    )
    capped: Final = estimate_request_max_cost(
        request_body={"model": "priced-model", **body, "inferenceConfig": {"maxTokens": 1}},
        route=route,
        llm_router=router,
    )
    assert uncapped is not None and capped is not None
    assert (capped < uncapped) is inference_config_bounds_hold


def test_priced_model_the_provider_resolver_rejects_still_gets_a_full_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    model: Final = "unresolvable/priced-model"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {
            "input_cost_per_token": 0.0001,
            "output_cost_per_token": 0.0002,
            "max_output_tokens": 128000,
            "litellm_provider": "openai",
            "mode": "chat",
        },
    )
    messages: Final = [{"role": "user", "content": "hi"}]
    chat_uncapped: Final = estimate_request_max_cost(
        request_body={"model": model, "messages": messages},
        route="/v1/chat/completions",
        llm_router=None,
    )
    chat_decoy: Final = estimate_request_max_cost(
        request_body={"model": model, "messages": messages, "max_output_tokens": 1},
        route="/v1/chat/completions",
        llm_router=None,
    )
    responses_uncapped: Final = estimate_request_max_cost(
        request_body={"model": model, "input": "hi"},
        route="/v1/responses",
        llm_router=None,
    )
    assert chat_uncapped is not None and responses_uncapped is not None
    assert chat_decoy == pytest.approx(chat_uncapped)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("token_id", "body", "expected_admitted"),
    [
        (
            "responses-max-tokens-and-output",
            {"model": "priced-model", "input": "hi", "max_tokens": 1, "max_output_tokens": 1000},
            5,
        ),
        ("responses-decoy-max-tokens-only", {"model": "priced-model", "input": "hi", "max_tokens": 1}, 1),
    ],
)
async def test_responses_concurrent_reservations_enforce_max_budget(
    spend_counter_store: SpendCounterStore,
    token_id: str,
    body: dict[str, object],
    expected_admitted: int,
) -> None:
    router: Final = _priced_responses_router()
    cache: Final = UserApiKeyCache()
    proxy_logging_obj: Final = ProxyLogging(user_api_key_cache=DualCache())
    valid_token: Final = UserAPIKeyAuth(api_key=token_id, token=token_id, max_budget=1.0, spend=0.0)

    outcomes: Final = await asyncio.gather(
        *(
            reserve_budget_for_request(
                request_body=dict(body),
                route="/v1/responses",
                llm_router=router,
                valid_token=valid_token,
                team_object=None,
                user_object=None,
                prisma_client=None,
                user_api_key_cache=cache,
                proxy_logging_obj=proxy_logging_obj,
            )
            for _ in range(10)
        ),
        return_exceptions=True,
    )

    admitted: Final = sum(1 for item in outcomes if isinstance(item, dict))
    rejected: Final = sum(1 for item in outcomes if isinstance(item, litellm.BudgetExceededError))
    assert admitted == expected_admitted
    assert rejected == 10 - expected_admitted
    assert spend_counter_store.get_local(f"spend:key:{token_id}") == pytest.approx(1.0)


def _router_for_backend(
    backend_model: str,
    *,
    model_name: str = "priced-model",
    input_cost: float = 0.0001,
    output_cost: float = 0.0002,
    max_output_tokens: int = 65536,
) -> Router:
    return Router(
        model_list=[
            Deployment(
                model_name=model_name,
                litellm_params=LiteLLM_Params(model=backend_model, api_key="sk-fake"),
                model_info=ModelInfo(
                    id=f"{backend_model}-deployment",
                    input_cost_per_token=input_cost,
                    output_cost_per_token=output_cost,
                    max_output_tokens=max_output_tokens,
                ),
            ).model_dump()
        ]
    )


@pytest.mark.parametrize(
    "backend_model",
    ["vertex_ai/gemini-2.5-pro", "gemini/gemini-2.5-pro", "vertex_ai/claude-opus-5-5", "openai/gpt-5.6"],
)
def test_chat_max_output_tokens_raises_the_hold_on_every_backend(backend_model: str) -> None:
    router: Final = _router_for_backend(backend_model, max_output_tokens=65536)
    base_messages: Final = [{"role": "user", "content": "hi"}]
    uncapped: Final = estimate_request_max_cost(
        request_body={"model": "priced-model", "messages": base_messages},
        route="/v1/chat/completions",
        llm_router=router,
    )
    with_max_tokens: Final = estimate_request_max_cost(
        request_body={"model": "priced-model", "messages": base_messages, "max_tokens": 65536},
        route="/v1/chat/completions",
        llm_router=router,
    )
    with_max_output: Final = estimate_request_max_cost(
        request_body={"model": "priced-model", "messages": base_messages, "max_output_tokens": 65536},
        route="/v1/chat/completions",
        llm_router=router,
    )
    with_both: Final = estimate_request_max_cost(
        request_body={"model": "priced-model", "messages": base_messages, "max_tokens": 1, "max_output_tokens": 65536},
        route="/v1/chat/completions",
        llm_router=router,
    )

    assert uncapped is not None and with_max_tokens is not None
    assert with_max_output == pytest.approx(with_max_tokens)
    assert with_both == pytest.approx(with_max_tokens)
    assert with_max_output > uncapped


def test_mixed_openai_and_gemini_group_guards_decoy_and_high_gemini_cap() -> None:
    router: Final = Router(
        model_list=[
            Deployment(
                model_name="mixed-model",
                litellm_params=LiteLLM_Params(model="openai/gpt-5.6", api_key="sk-fake"),
                model_info=ModelInfo(
                    id="openai-dep",
                    input_cost_per_token=0.0001,
                    output_cost_per_token=0.0002,
                    max_output_tokens=65536,
                ),
            ).model_dump(),
            Deployment(
                model_name="mixed-model",
                litellm_params=LiteLLM_Params(model="vertex_ai/gemini-2.5-pro", api_key="sk-fake"),
                model_info=ModelInfo(
                    id="gemini-dep",
                    input_cost_per_token=0.0001,
                    output_cost_per_token=0.0002,
                    max_output_tokens=65536,
                ),
            ).model_dump(),
        ]
    )
    base_messages: Final = [{"role": "user", "content": "hi"}]
    uncapped: Final = estimate_request_max_cost(
        request_body={"model": "mixed-model", "messages": base_messages},
        route="/v1/chat/completions",
        llm_router=router,
    )
    decoy_small: Final = estimate_request_max_cost(
        request_body={"model": "mixed-model", "messages": base_messages, "max_output_tokens": 1},
        route="/v1/chat/completions",
        llm_router=router,
    )
    high_max_tokens: Final = estimate_request_max_cost(
        request_body={"model": "mixed-model", "messages": base_messages, "max_tokens": 65536},
        route="/v1/chat/completions",
        llm_router=router,
    )
    high_max_output: Final = estimate_request_max_cost(
        request_body={"model": "mixed-model", "messages": base_messages, "max_output_tokens": 65536},
        route="/v1/chat/completions",
        llm_router=router,
    )

    assert uncapped is not None and high_max_tokens is not None
    assert decoy_small == pytest.approx(uncapped)
    assert high_max_output == pytest.approx(high_max_tokens)
    assert high_max_output > uncapped


@pytest.mark.asyncio
async def test_gemini_chat_completions_concurrent_reservations_reserve_high_max_output_tokens(
    spend_counter_store: SpendCounterStore,
) -> None:
    router: Final = _router_for_backend(
        "vertex_ai/gemini-2.5-pro",
        input_cost=0.0,
        output_cost=0.00001,
        max_output_tokens=65536,
    )
    cache: Final = UserApiKeyCache()
    proxy_logging_obj: Final = ProxyLogging(user_api_key_cache=DualCache())
    token_id: Final = "gemini-high-max-output-tokens"
    valid_token: Final = UserAPIKeyAuth(api_key=token_id, token=token_id, max_budget=1.0, spend=0.0)
    body: Final[Mapping[str, object]] = {
        "model": "priced-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_output_tokens": 65536,
    }

    outcomes: Final = await asyncio.gather(
        *(
            reserve_budget_for_request(
                request_body=dict(body),
                route="/v1/chat/completions",
                llm_router=router,
                valid_token=valid_token,
                team_object=None,
                user_object=None,
                prisma_client=None,
                user_api_key_cache=cache,
                proxy_logging_obj=proxy_logging_obj,
            )
            for _ in range(10)
        ),
        return_exceptions=True,
    )

    admitted: Final = sum(1 for item in outcomes if isinstance(item, dict))
    rejected: Final = sum(1 for item in outcomes if isinstance(item, litellm.BudgetExceededError))
    assert admitted == 2
    assert rejected == 8
    assert spend_counter_store.get_local(f"spend:key:{token_id}") == pytest.approx(1.0)


def _bridged_responses_router(
    backend_model: str,
    *,
    input_cost: float = 0.0001,
    output_cost: float = 0.0002,
    max_output_tokens: int = 65536,
    deployment_use_chat: bool = False,
    default_use_chat: bool = False,
) -> Router:
    litellm_params: Final = LiteLLM_Params(
        model=backend_model,
        api_key="sk-fake",
        use_chat_completions_api=deployment_use_chat if deployment_use_chat else None,
    )
    defaults: Final[Mapping[str, object] | None] = {"use_chat_completions_api": True} if default_use_chat else None
    return Router(
        model_list=[
            Deployment(
                model_name="priced-model",
                litellm_params=litellm_params,
                model_info=ModelInfo(
                    id=f"{backend_model}-bridged-deployment",
                    input_cost_per_token=input_cost,
                    output_cost_per_token=output_cost,
                    max_output_tokens=max_output_tokens,
                ),
            ).model_dump()
        ],
        default_litellm_params=defaults,
    )


@pytest.mark.parametrize("cap_field", ["max_completion_tokens", "max_tokens"])
@pytest.mark.parametrize(
    ("mode", "backend_model"),
    [
        ("deployment_flag", "openai/gpt-5.4"),
        ("router_default", "openai/gpt-5.4"),
        ("request_flag", "openai/gpt-5.4"),
        ("model_prefix", "openai/chat_completions/gpt-5.4"),
        ("non_native_provider", "anthropic/claude-opus-5-5"),
    ],
)
def test_bridged_responses_reserves_chat_completion_output_caps(
    cap_field: str,
    mode: str,
    backend_model: str,
) -> None:
    router: Final = _bridged_responses_router(
        backend_model,
        deployment_use_chat=mode == "deployment_flag",
        default_use_chat=mode == "router_default",
    )
    request_bridge: Final[Mapping[str, object]] = {"use_chat_completions_api": True} if mode == "request_flag" else {}
    base_body: Final[Mapping[str, object]] = {"model": "priced-model", "input": "hi", **request_bridge}

    uncapped: Final = estimate_request_max_cost(request_body=base_body, route="/v1/responses", llm_router=router)
    expected_high: Final = estimate_request_max_cost(
        request_body={**base_body, "max_output_tokens": 32768},
        route="/v1/responses",
        llm_router=router,
    )
    bridged_high: Final = estimate_request_max_cost(
        request_body={**base_body, cap_field: 32768},
        route="/v1/responses",
        llm_router=router,
    )
    bridged_paired: Final = estimate_request_max_cost(
        request_body={**base_body, "max_output_tokens": 1, cap_field: 32768},
        route="/v1/responses",
        llm_router=router,
    )

    assert uncapped is not None and expected_high is not None
    assert expected_high > uncapped
    assert bridged_high == pytest.approx(expected_high)
    assert bridged_paired == pytest.approx(expected_high)


def test_responses_chat_caps_raise_the_hold_with_or_without_a_bridge() -> None:
    bridged_request_cost: Final = estimate_request_max_cost(
        request_body={
            "model": "openai/gpt-5.4",
            "input": "hi",
            "use_chat_completions_api": True,
            "max_completion_tokens": 32768,
        },
        route="/v1/responses",
        llm_router=None,
    )
    bridged_uncapped_cost: Final = estimate_request_max_cost(
        request_body={"model": "openai/gpt-5.4", "input": "hi", "use_chat_completions_api": True},
        route="/v1/responses",
        llm_router=None,
    )
    bridged_expected_cost: Final = estimate_request_max_cost(
        request_body={
            "model": "openai/gpt-5.4",
            "input": "hi",
            "use_chat_completions_api": True,
            "max_output_tokens": 32768,
        },
        route="/v1/responses",
        llm_router=None,
    )
    native_request_cost: Final = estimate_request_max_cost(
        request_body={"model": "openai/gpt-5.4", "input": "hi", "max_completion_tokens": 32768},
        route="/v1/responses",
        llm_router=None,
    )

    assert bridged_uncapped_cost is not None and bridged_expected_cost is not None
    assert bridged_expected_cost > bridged_uncapped_cost
    assert bridged_request_cost == pytest.approx(bridged_expected_cost)
    assert native_request_cost == pytest.approx(bridged_expected_cost)


def test_mixed_native_and_bridged_responses_group_reserves_worst_case() -> None:
    router: Final = Router(
        model_list=[
            Deployment(
                model_name="mixed-responses",
                litellm_params=LiteLLM_Params(model="openai/gpt-5.4", api_key="sk-native"),
                model_info=ModelInfo(
                    id="native-deployment",
                    input_cost_per_token=0.0001,
                    output_cost_per_token=0.0002,
                    max_output_tokens=65536,
                ),
            ).model_dump(),
            Deployment(
                model_name="mixed-responses",
                litellm_params=LiteLLM_Params(
                    model="openai/gpt-5.4",
                    api_key="sk-bridged",
                    use_chat_completions_api=True,
                ),
                model_info=ModelInfo(
                    id="bridged-deployment",
                    input_cost_per_token=0.0001,
                    output_cost_per_token=0.0002,
                    max_output_tokens=65536,
                ),
            ).model_dump(),
        ]
    )
    base_body: Final[Mapping[str, object]] = {"model": "mixed-responses", "input": "hi"}

    uncapped: Final = estimate_request_max_cost(request_body=base_body, route="/v1/responses", llm_router=router)
    decoy_small: Final = estimate_request_max_cost(
        request_body={**base_body, "max_completion_tokens": 1},
        route="/v1/responses",
        llm_router=router,
    )
    high_output: Final = estimate_request_max_cost(
        request_body={**base_body, "max_output_tokens": 32768},
        route="/v1/responses",
        llm_router=router,
    )
    high_completion: Final = estimate_request_max_cost(
        request_body={**base_body, "max_completion_tokens": 32768},
        route="/v1/responses",
        llm_router=router,
    )
    paired_low_output_high_completion: Final = estimate_request_max_cost(
        request_body={**base_body, "max_output_tokens": 1, "max_completion_tokens": 32768},
        route="/v1/responses",
        llm_router=router,
    )

    assert uncapped is not None and high_output is not None
    assert high_output > uncapped
    assert decoy_small == pytest.approx(uncapped)
    assert high_completion == pytest.approx(high_output)
    assert paired_low_output_high_completion == pytest.approx(high_output)


@pytest.mark.parametrize("cap_field", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize(
    ("backend_models", "lowers_hold"),
    [
        (("anthropic/claude-haiku-5-5",), True),
        (("openai/chat_completions/gpt-5.6",), True),
        (("openai/gpt-5.6",), False),
        (("anthropic/claude-haiku-5-5", "openai/gpt-5.6"), False),
        (("v0/v0-1.0-md",), False),
    ],
)
def test_responses_chat_cap_lowers_the_hold_only_when_every_deployment_bridges_and_reads_it(
    cap_field: str, backend_models: tuple[str, ...], lowers_hold: bool
) -> None:
    router: Final = _priced_router(backend_models)
    body: Final = {"model": "priced-model", "input": "hi"}
    uncapped: Final = estimate_request_max_cost(request_body=body, route="/v1/responses", llm_router=router)
    capped: Final = estimate_request_max_cost(
        request_body={**body, cap_field: 1}, route="/v1/responses", llm_router=router
    )

    assert uncapped is not None and capped is not None
    assert (capped < uncapped) is lowers_hold


_CHAT_BODY: Final[Mapping[str, object]] = {"messages": [{"role": "user", "content": "hi"}]}
_INPUT_BODY: Final[Mapping[str, object]] = {"input": "hi"}
_CONTENTS_BODY: Final[Mapping[str, object]] = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}


@pytest.mark.parametrize(
    ("route", "backend_model", "shape", "cap_field", "raised_to_minimum"),
    [
        ("/v1/responses", "openai/gpt-5.6", _INPUT_BODY, "max_output_tokens", True),
        ("/cursor/chat/completions", "openai/gpt-5.6", _INPUT_BODY, "max_output_tokens", True),
        ("/v1/responses", "anthropic/claude-haiku-5-5", _INPUT_BODY, "max_tokens", False),
        ("/v1/chat/completions", "gemini/gemini-2.5-pro", _CHAT_BODY, "max_output_tokens", False),
    ],
)
def test_only_a_responses_max_output_tokens_below_the_forwarded_minimum_holds_the_minimum(
    route: str, backend_model: str, shape: Mapping[str, object], cap_field: str, raised_to_minimum: bool
) -> None:
    router: Final = _priced_router((backend_model,))

    def estimate(tokens: int) -> float | None:
        return estimate_request_max_cost(
            request_body={"model": "priced-model", **shape, cap_field: tokens}, route=route, llm_router=router
        )

    one_token_hold: Final = estimate(1)
    minimum_hold: Final = estimate(OPENAI_RESPONSES_API_MIN_MAX_OUTPUT_TOKENS)

    assert one_token_hold is not None and minimum_hold is not None
    assert (one_token_hold == pytest.approx(minimum_hold)) is raised_to_minimum
    assert one_token_hold <= minimum_hold


@pytest.mark.parametrize(
    ("route", "shape", "cap", "lowers_hold"),
    [
        ("/cursor/chat/completions", _INPUT_BODY, {"max_tokens": 1}, False),
        ("/cursor/chat/completions", _INPUT_BODY, {"max_output_tokens": 1}, True),
        ("/cursor/chat/completions", _CHAT_BODY, {"inferenceConfig": {"maxTokens": 1}}, False),
        ("/cursor/chat/completions", _CHAT_BODY, {"max_tokens": 1}, True),
        ("/claude_code_gateway/v1/messages", _CHAT_BODY, {"max_completion_tokens": 1}, False),
        ("/claude_code_gateway/v1/messages", _CHAT_BODY, {"max_tokens": 1}, True),
        ("/v1beta/interactions", _INPUT_BODY, {"max_tokens": 1}, False),
        ("/v1beta/interactions", _INPUT_BODY, {"max_output_tokens": 1}, False),
        ("/v1beta/models/priced-model:generateContent", _CONTENTS_BODY, {"max_tokens": 1}, False),
        ("/models/priced-model:generateContent", _CONTENTS_BODY, {"max_output_tokens": 1}, False),
        ("/v1beta/models/priced-model:streamGenerateContent", _CONTENTS_BODY, {"max_completion_tokens": 1}, False),
        ("/gemini/v1beta/models/priced-model:generateContent", _CONTENTS_BODY, {"max_tokens": 1}, False),
        (
            "/vertex_ai/v1/projects/p/locations/us-central1/publishers/google/models/priced-model:streamGenerateContent",
            _CONTENTS_BODY,
            {"inferenceConfig": {"maxTokens": 1}},
            False,
        ),
    ],
)
def test_routes_outside_the_call_type_map_trust_only_the_cap_they_forward(
    route: str, shape: Mapping[str, object], cap: Mapping[str, object], lowers_hold: bool
) -> None:
    router: Final = _priced_router(("openai/gpt-5.6",))
    body: Final = {"model": "priced-model", **shape}
    uncapped: Final = estimate_request_max_cost(request_body=body, route=route, llm_router=router)
    capped: Final = estimate_request_max_cost(request_body={**body, **cap}, route=route, llm_router=router)

    assert uncapped is not None and capped is not None
    assert (capped < uncapped) is lowers_hold


@pytest.mark.parametrize(
    ("route", "backend_models", "lowers_hold"),
    [
        ("/v1beta/models/priced-model:generateContent", ("gemini/gemini-3.8-flash",), True),
        ("/gemini/v1beta/models/priced-model:streamGenerateContent", ("vertex_ai/gemini-3.8-flash",), True),
        ("/v1beta/models/priced-model:generateContent", ("anthropic/claude-haiku-5-5", "openai/gpt-5.6"), True),
        ("/v1beta/models/priced-model:generateContent", ("gemini/gemini-3.8-flash", "v0/v0-1.0-md"), False),
    ],
)
def test_a_generation_config_cap_lowers_a_generate_content_hold_only_when_every_deployment_receives_it(
    route: str, backend_models: tuple[str, ...], lowers_hold: bool
) -> None:
    router: Final = _priced_router(backend_models)
    body: Final = {"model": "priced-model", **_CONTENTS_BODY}
    uncapped: Final = estimate_request_max_cost(request_body=body, route=route, llm_router=router)
    capped: Final = estimate_request_max_cost(
        request_body={**body, "generationConfig": {"maxOutputTokens": 1}}, route=route, llm_router=router
    )

    assert uncapped is not None and capped is not None
    assert (capped < uncapped) is lowers_hold


@pytest.mark.asyncio
async def test_bridged_responses_concurrent_reservations_reserve_max_completion_tokens(
    spend_counter_store: SpendCounterStore,
) -> None:
    output_cost: Final = 0.00001
    router: Final = _bridged_responses_router(
        "openai/gpt-5.4",
        input_cost=0.0,
        output_cost=output_cost,
        max_output_tokens=65536,
        deployment_use_chat=True,
    )
    cache: Final = UserApiKeyCache()
    proxy_logging_obj: Final = ProxyLogging(user_api_key_cache=DualCache())
    token_id: Final = "bridged-responses-max-completion-tokens"
    max_budget: Final = 1.25 * 32768 * output_cost
    valid_token: Final = UserAPIKeyAuth(api_key=token_id, token=token_id, max_budget=max_budget, spend=0.0)
    body: Final[Mapping[str, object]] = {
        "model": "priced-model",
        "input": "hi",
        "max_completion_tokens": 32768,
    }

    outcomes: Final = await asyncio.gather(
        *(
            reserve_budget_for_request(
                request_body=dict(body),
                route="/v1/responses",
                llm_router=router,
                valid_token=valid_token,
                team_object=None,
                user_object=None,
                prisma_client=None,
                user_api_key_cache=cache,
                proxy_logging_obj=proxy_logging_obj,
            )
            for _ in range(10)
        ),
        return_exceptions=True,
    )

    admitted: Final = sum(1 for item in outcomes if isinstance(item, dict))
    rejected: Final = sum(1 for item in outcomes if isinstance(item, litellm.BudgetExceededError))
    assert admitted == 2
    assert rejected == 8
    assert spend_counter_store.get_local(f"spend:key:{token_id}") == pytest.approx(max_budget)


def test_responses_router_ignores_spoofed_custom_llm_provider_override() -> None:
    native_router: Final = _bridged_responses_router("openai/gpt-5.4", max_output_tokens=65536)
    native_base: Final[Mapping[str, object]] = {"model": "priced-model", "input": "hi"}
    native_uncapped: Final = estimate_request_max_cost(
        request_body=native_base,
        route="/v1/responses",
        llm_router=native_router,
    )
    native_spoofed_max_tokens: Final = estimate_request_max_cost(
        request_body={**native_base, "custom_llm_provider": "anthropic", "max_tokens": 1},
        route="/v1/responses",
        llm_router=native_router,
    )
    native_spoofed_completion_cap: Final = estimate_request_max_cost(
        request_body={**native_base, "custom_llm_provider": "anthropic", "max_completion_tokens": 1},
        route="/v1/responses",
        llm_router=native_router,
    )

    assert native_uncapped is not None
    assert native_spoofed_max_tokens == pytest.approx(native_uncapped)
    assert native_spoofed_completion_cap == pytest.approx(native_uncapped)

    bridged_router: Final = _bridged_responses_router("anthropic/claude-opus-5-5", max_output_tokens=65536)
    bridged_uncapped: Final = estimate_request_max_cost(
        request_body=native_base,
        route="/v1/responses",
        llm_router=bridged_router,
    )
    bridged_expected: Final = estimate_request_max_cost(
        request_body={**native_base, "max_output_tokens": 32768},
        route="/v1/responses",
        llm_router=bridged_router,
    )
    bridged_spoofed_openai: Final = estimate_request_max_cost(
        request_body={**native_base, "custom_llm_provider": "openai", "max_completion_tokens": 32768},
        route="/v1/responses",
        llm_router=bridged_router,
    )

    assert bridged_uncapped is not None and bridged_expected is not None
    assert bridged_expected > bridged_uncapped
    assert bridged_spoofed_openai == pytest.approx(bridged_expected)


@pytest.mark.asyncio
async def test_responses_native_deployment_concurrent_rejects_spoofed_custom_llm_provider_and_max_tokens(
    spend_counter_store: SpendCounterStore,
) -> None:
    from litellm.proxy.spend_tracking.budget_reservation import DEFAULT_MAX_OUTPUT_TOKENS_FALLBACK

    output_cost: Final = 0.00001
    router: Final = _bridged_responses_router(
        "openai/gpt-5.4",
        input_cost=0.0,
        output_cost=output_cost,
        max_output_tokens=65536,
    )
    cache: Final = UserApiKeyCache()
    proxy_logging_obj: Final = ProxyLogging(user_api_key_cache=DualCache())
    token_id: Final = "native-responses-spoofed-provider"
    max_budget: Final = 1.5 * DEFAULT_MAX_OUTPUT_TOKENS_FALLBACK * output_cost
    valid_token: Final = UserAPIKeyAuth(api_key=token_id, token=token_id, max_budget=max_budget, spend=0.0)
    body: Final[Mapping[str, object]] = {
        "model": "priced-model",
        "input": "hi",
        "custom_llm_provider": "anthropic",
        "max_tokens": 1,
    }

    outcomes: Final = await asyncio.gather(
        *(
            reserve_budget_for_request(
                request_body=dict(body),
                route="/v1/responses",
                llm_router=router,
                valid_token=valid_token,
                team_object=None,
                user_object=None,
                prisma_client=None,
                user_api_key_cache=cache,
                proxy_logging_obj=proxy_logging_obj,
            )
            for _ in range(10)
        ),
        return_exceptions=True,
    )

    admitted: Final = sum(1 for item in outcomes if isinstance(item, dict))
    rejected: Final = sum(1 for item in outcomes if isinstance(item, litellm.BudgetExceededError))
    assert admitted == 2
    assert rejected == 8
    assert spend_counter_store.get_local(f"spend:key:{token_id}") == pytest.approx(max_budget)


@pytest.mark.parametrize(
    ("deployment_use_chat", "default_use_chat"),
    [(True, False), (False, True)],
)
@pytest.mark.asyncio
async def test_responses_concurrent_explicit_null_bridge_override_caps_spend(
    spend_counter_store: SpendCounterStore,
    deployment_use_chat: bool,
    default_use_chat: bool,
) -> None:
    from litellm.proxy.spend_tracking.budget_reservation import DEFAULT_MAX_OUTPUT_TOKENS_FALLBACK

    output_cost: Final = 0.00001
    router: Final = _bridged_responses_router(
        "openai/gpt-5.4",
        input_cost=0.0,
        output_cost=output_cost,
        max_output_tokens=65536,
        deployment_use_chat=deployment_use_chat,
        default_use_chat=default_use_chat,
    )
    cache: Final = UserApiKeyCache()
    proxy_logging_obj: Final = ProxyLogging(user_api_key_cache=DualCache())
    token_id: Final = f"native-responses-null-bridge-{deployment_use_chat}-{default_use_chat}"
    max_budget: Final = 1.5 * DEFAULT_MAX_OUTPUT_TOKENS_FALLBACK * output_cost
    valid_token: Final = UserAPIKeyAuth(api_key=token_id, token=token_id, max_budget=max_budget, spend=0.0)
    body: Final[Mapping[str, object]] = {
        "model": "priced-model",
        "input": "hi",
        "use_chat_completions_api": None,
        "max_tokens": 1,
    }

    outcomes: Final = await asyncio.gather(
        *(
            reserve_budget_for_request(
                request_body=dict(body),
                route="/v1/responses",
                llm_router=router,
                valid_token=valid_token,
                team_object=None,
                user_object=None,
                prisma_client=None,
                user_api_key_cache=cache,
                proxy_logging_obj=proxy_logging_obj,
            )
            for _ in range(10)
        ),
        return_exceptions=True,
    )

    admitted: Final = sum(1 for item in outcomes if isinstance(item, dict))
    rejected: Final = sum(1 for item in outcomes if isinstance(item, litellm.BudgetExceededError))
    assert admitted == 2
    assert rejected == 8
    assert spend_counter_store.get_local(f"spend:key:{token_id}") == pytest.approx(max_budget)


@pytest.mark.parametrize(
    "duration,end,start",
    (
        ("30d", "2026-08-14T12:00:00Z", "2026-07-15T12:00:00Z"),
        ("1d", "2026-03-08T17:00:00Z", "2026-03-07T17:00:00Z"),
        ("1d", "2026-11-01T16:00:00Z", "2026-10-31T16:00:00Z"),
    ),
)
def test_existing_windows_keep_their_elapsed_start_until_rollover(duration: str, end: str, start: str) -> None:
    assert get_budget_window_start(
        {"budget_duration": duration, "reset_at": end}, BudgetResetSettings(timezone="America/New_York")
    ) == datetime.fromisoformat(start)


@pytest.mark.parametrize("duration", ("1h", "60m", "3600s"))
@pytest.mark.parametrize(
    "end,now",
    (("2026-11-01T07:00:00Z", "2026-11-01T06:15:00Z"), ("2026-03-08T07:30:00Z", "2026-03-08T07:15:00Z")),
)
def test_customer_elapsed_windows_share_the_persisted_calendar_window(duration: str, end: str, now: str) -> None:
    settings: Final = BudgetResetSettings(timezone="America/New_York")
    reset: Final = datetime.fromisoformat(end)
    limit: Final = BudgetLimit(budget_duration=duration, max_budget=1, reset_at=reset)
    counter: Final = customer_window_counters(
        "customer_team", "hourly-team", (limit,), now=datetime.fromisoformat(now), reset_settings=settings
    )[0]
    stored: Final = {
        "budget_duration": duration,
        **get_budget_window_reset_fields(duration, reset_at=reset, settings=settings),
    }

    assert counter.window_start == reset - timedelta(hours=1)
    assert get_budget_window_start(stored, settings) == counter.window_start


@pytest.mark.parametrize(
    "duration,zone,expected",
    (
        ("daily", "Asia/Tokyo", "2026-07-16T00:00:00+09:00"),
        ("1h", "Asia/Kolkata", "2026-07-15T16:00:00+05:30"),
        ("45m", "UTC", "2026-07-15T10:45:00Z"),
        ("5h", "UTC", "2026-07-15T15:00:00Z"),
    ),
)
def test_created_customer_and_inline_windows_keep_the_same_schedule(duration: str, zone: str, expected: str) -> None:
    from litellm.litellm_core_utils.duration_parser import duration_in_seconds
    from litellm.models.budget import initialize_budget_limits
    from litellm.proxy.common_utils.timezone_utils import compute_budget_reset_at

    now: Final = datetime(2026, 7, 15, 10, 17, tzinfo=timezone.utc)
    settings: Final = BudgetResetSettings(timezone=zone)
    limits: Final = initialize_budget_limits((BudgetLimit(budget_duration=duration, max_budget=10),), settings, now)
    first: Final = datetime.fromisoformat(expected)
    interval: Final = timedelta(seconds=duration_in_seconds(duration))

    assert limits[0].reset_at == first
    for period in range(3):
        current: Final = first + period * interval
        counter: Final = customer_window_counters(
            "customer_team", "schedule-team", limits, now=current, reset_settings=settings
        )[0]
        next_reset: Final = compute_budget_reset_at(duration, settings, current, previous_reset=first)
        inline: Final = {
            "budget_duration": duration,
            **get_budget_window_reset_fields(duration, reset_at=next_reset, settings=settings),
        }

        assert counter.window_start == current
        assert counter.reset_at == current + interval
        assert get_budget_window_start(inline, settings) == current
        assert datetime.fromisoformat(inline["reset_at"]) == counter.reset_at


@pytest.mark.parametrize(
    "duration,zone,expected",
    (
        ("daily", "Asia/Tokyo", "2026-07-16T00:00:00+09:00"),
        ("1h", "Asia/Kolkata", "2026-07-15T16:00:00+05:30"),
        ("45m", "UTC", "2026-07-15T10:45:00Z"),
        ("5h", "UTC", "2026-07-15T15:00:00Z"),
    ),
)
def test_created_customer_windows_keep_local_alignment_and_full_intervals(
    duration: str, zone: str, expected: str
) -> None:
    from litellm.litellm_core_utils.duration_parser import duration_in_seconds
    from litellm.models.budget import initialize_budget_limits

    now: Final = datetime(2026, 7, 15, 10, 17, tzinfo=timezone.utc)
    settings: Final = BudgetResetSettings(timezone=zone)
    limits: Final = initialize_budget_limits((BudgetLimit(budget_duration=duration, max_budget=10),), settings, now)
    first: Final = datetime.fromisoformat(expected)
    interval: Final = timedelta(seconds=duration_in_seconds(duration))

    assert limits[0].reset_at == first
    for period in range(3):
        current: Final = first + period * interval
        counter: Final = customer_window_counters(
            "customer_team", "schedule-team", limits, now=current, reset_settings=settings
        )[0]
        assert counter.window_start == current
        assert counter.reset_at == current + interval


@pytest.mark.parametrize(
    "now,expected_start,expected_end",
    (
        ("2026-11-01T05:15:00Z", "2026-11-01T05:00:00Z", "2026-11-01T06:00:00Z"),
        ("2026-11-01T06:15:00Z", "2026-11-01T06:00:00Z", "2026-11-01T07:00:00Z"),
        ("2026-03-08T06:45:00Z", "2026-03-08T06:00:00Z", "2026-03-08T07:00:00Z"),
    ),
)
def test_new_customer_hourly_window_includes_current_usage_through_dst(
    now: str, expected_start: str, expected_end: str
) -> None:
    counter: Final = customer_window_counters(
        "customer_team",
        "hourly-team",
        (BudgetLimit(budget_duration="1h", max_budget=1),),
        now=datetime.fromisoformat(now),
        reset_settings=BudgetResetSettings(timezone="America/New_York"),
    )[0]

    assert counter.window_start == datetime.fromisoformat(expected_start)
    assert counter.reset_at == datetime.fromisoformat(expected_end)


@pytest.mark.parametrize("days_after_reset", (0, 1, 2, 30))
def test_expired_customer_elapsed_schedule_keeps_its_saved_anchor(days_after_reset: int) -> None:
    saved_reset: Final = datetime(2026, 11, 1, 4, tzinfo=timezone.utc)
    start: Final = saved_reset + timedelta(days=days_after_reset)
    counter: Final = customer_window_counters(
        "customer_team",
        "daily-team",
        (BudgetLimit(budget_duration="12h", max_budget=1, reset_at=saved_reset),),
        now=start + timedelta(minutes=1),
        reset_settings=BudgetResetSettings(timezone="America/New_York"),
    )[0]

    assert counter.window_start == start
    assert counter.reset_at == start + timedelta(hours=12)


@pytest.mark.parametrize("duration", ("24h", "daily"))
@pytest.mark.parametrize(
    "now,start,end",
    (
        ("2026-11-01T04:01:00Z", "2026-11-01T04:00:00Z", "2026-11-02T05:00:00Z"),
        ("2026-11-01T12:00:00Z", "2026-11-01T04:00:00Z", "2026-11-02T05:00:00Z"),
        ("2026-11-02T05:01:00Z", "2026-11-02T05:00:00Z", "2026-11-03T05:00:00Z"),
    ),
)
def test_expired_customer_daily_window_follows_local_midnight(duration: str, now: str, start: str, end: str) -> None:
    counter: Final = customer_window_counters(
        "customer_team",
        "daily-team",
        (BudgetLimit(budget_duration=duration, max_budget=1, reset_at=datetime(2026, 11, 1, 4, tzinfo=timezone.utc)),),
        now=datetime.fromisoformat(now),
        reset_settings=BudgetResetSettings(timezone="America/New_York"),
    )[0]

    assert counter.window_start == datetime.fromisoformat(start)
    assert counter.reset_at == datetime.fromisoformat(end)
