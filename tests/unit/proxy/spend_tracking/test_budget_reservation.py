from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import patch

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
    _get_budget_counters,
    _get_team_member_budget_counter,
    _max_cost_for_cost_info,
    count_request_input_tokens,
    estimate_request_input_cost,
    estimate_request_max_cost,
    get_budget_window_start,
    reserve_budget_for_request,
)
from litellm.proxy.spend_tracking.input_tokens import count_input_tokens_for_model
from litellm.proxy.utils import ProxyLogging
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
def test_a_named_cached_content_reserves_the_rest_of_the_window_at_the_cache_read_rate_but_not_in_the_input_floor(
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
    assert floor == pytest.approx(question_tokens * _gemini_rate("input_cost_per_token"))


@pytest.mark.parametrize(
    "unsized",
    [
        {"contents": [{"role": "user", "parts": [{"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}}]}]},
        {"contents": [{"role": "user", "parts": [{"fileData": {"mimeType": "video/mp4", "fileUri": "gs://b/a.mp4"}}]}]},
        {"contents": [{"role": "user", "parts": [{"fileData": {"mimeType": "application/pdf", "fileUri": "gs://b/r.pdf"}}]}]},
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


_CACHED_BODY: Final = GEMINI_QUESTION | {
    "cachedContent": "cachedContents/ctx-1",
    "generationConfig": {"maxOutputTokens": 100},
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
    ],
)
def test_cached_content_reservation_prices_the_rest_of_the_window(
    model_info: dict[str, object], expected: float
) -> None:
    reserved: Final = _max_cost_for_cost_info(
        request_body=_CACHED_BODY,
        route=GEMINI_ROUTES[0],
        model="custom-gemini",
        model_info=model_info,
        input_tokens=200,
    )

    assert reserved == pytest.approx(expected)


def test_cached_content_without_an_input_or_cache_read_price_cannot_be_reserved() -> None:
    reserved: Final = _max_cost_for_cost_info(
        request_body=_CACHED_BODY,
        route=GEMINI_ROUTES[0],
        model="custom-gemini",
        model_info={"output_cost_per_token": 4e-06, "max_input_tokens": 10_000},
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
