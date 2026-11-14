from __future__ import annotations

import asyncio
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
from litellm.proxy.common_utils.timezone_utils import BudgetResetSettings
from litellm.proxy.spend_tracking.budget_reservation import (
    _get_team_member_budget_counter,
    estimate_request_max_cost,
    get_budget_window_start,
    release_unbound_budget_reservation,
    count_request_input_tokens,
    reserve_budget_for_request,
)
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore

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


@pytest.mark.asyncio
@pytest.mark.parametrize(("route", "request_body"), COUNT_TOKENS_REQUESTS)
async def test_repeated_token_counting_never_touches_a_tiny_budget(
    spend_counter_store: SpendCounterStore, route: str, request_body: dict[str, object]
):
    counter_key: Final = f"spend:key:{TINY_BUDGET_KEY_TOKEN}"

    assert await _reserve_for_tiny_budget_key(route, request_body) is None
    assert await _reserve_for_tiny_budget_key(route, request_body) is None
    assert spend_counter_store.get_local(counter_key) is None

    completion: Final = await _reserve_for_tiny_budget_key(
        "/v1/messages", {"model": "claude-sonnet-5", "max_tokens": 16, "messages": ANTHROPIC_MESSAGES}
    )
    assert completion is not None
    reserved_cost: Final = completion["reserved_cost"]
    assert isinstance(reserved_cost, float)
    assert reserved_cost > 0
    assert spend_counter_store.get_local(counter_key) == pytest.approx(reserved_cost)


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


@pytest.mark.asyncio
async def test_reservation_starts_unbound_to_any_callback():
    reservation: Final = await _reserve("/v1/responses")

    assert reservation is not None
    assert reservation["callback_bound"] is False


@pytest.mark.asyncio
async def test_release_unbound_budget_reservation_frees_the_counter(spend_counter_store: SpendCounterStore):
    counter_key: Final = f"spend:key:{TINY_BUDGET_KEY_TOKEN}"
    reservation: Final = await _reserve_for_tiny_budget_key(
        "/v1/chat/completions", {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
    )
    assert reservation is not None
    assert spend_counter_store.get_local(counter_key) == pytest.approx(reservation["reserved_cost"])

    await release_unbound_budget_reservation(reservation)

    assert spend_counter_store.get_local(counter_key) == pytest.approx(0.0)
    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_release_unbound_budget_reservation_leaves_a_bound_one_to_its_callback(
    spend_counter_store: SpendCounterStore,
):
    counter_key: Final = f"spend:key:{TINY_BUDGET_KEY_TOKEN}"
    reservation: Final = await _reserve_for_tiny_budget_key(
        "/v1/chat/completions", {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
    )
    assert reservation is not None
    reservation["callback_bound"] = True

    await release_unbound_budget_reservation(reservation)

    assert spend_counter_store.get_local(counter_key) == pytest.approx(reservation["reserved_cost"])
    assert reservation["finalized"] is False


@pytest.mark.asyncio
async def test_reservation_holds_cost_against_cached_org_and_project_budgets(spend_counter_store: SpendCounterStore):
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
    assert spend_counter_store.get_local("spend:org:org-budgeted") == pytest.approx(reserved_cost)
    assert spend_counter_store.get_local("spend:project:project-budgeted") == pytest.approx(reserved_cost)


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
@pytest.mark.parametrize(
    "duration,zone,reset_at",
    (
        ("12mo", "UTC", "2029-01-01T00:00:00Z"),
        ("12mo", "America/New_York", "2029-01-01T00:00:00-05:00"),
        ("12mo", "Pacific/Apia", "2012-01-01T00:00:00+14:00"),
    ),
)
async def test_cold_annual_counter_preserves_reset_writer_spend(
    monkeypatch: pytest.MonkeyPatch, duration: str, zone: str, reset_at: str
) -> None:
    from litellm.proxy.auth.auth_checks import _virtual_key_multi_budget_check
    from litellm.proxy.common_utils.reset_budget_job import ResetBudgetJob
    from litellm.types.integrations.slack_alerting import Litellm_EntityType

    database: Final = _AnnualWindowDatabase()
    store: Final = SpendCounterStore(connect(None).spend_counters())
    monkeypatch.setattr(litellm, "timezone", zone, raising=False)
    monkeypatch.setattr(proxy_server, "prisma_client", database)
    monkeypatch.setattr(proxy_server, "spend_counter_store", store)
    monkeypatch.setattr(proxy_server, "spend_counter_cache", DualCache())
    await ResetBudgetJob._roll_window_spend_row(
        prisma_client=database,  # pyright: ignore[reportArgumentType]  # in-memory external database boundary
        entity_type=Litellm_EntityType.KEY,
        entity_id="calendar-key",
        budget_duration=duration,
        next_reset_at=datetime.fromisoformat(reset_at.replace("Z", "+00:00")),
        reset_settings=BudgetResetSettings(timezone=zone),
    )
    database.spend = 10.0
    token: Final = UserAPIKeyAuth(
        token="calendar-key",
        budget_limits=[{"budget_duration": duration, "reset_at": reset_at, "max_budget": 10.0}],
    )

    with pytest.raises(litellm.BudgetExceededError) as failure:
        await _virtual_key_multi_budget_check(token)
    assert failure.value.current_cost == 10.0
    assert store.get_local(f"spend:key:calendar-key:window:{duration}") == 10.0
    assert database.scans == 0


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


@pytest.mark.parametrize("backend_model", ["vertex_ai/gemini-2.5-pro", "gemini/gemini-2.5-pro"])
def test_gemini_chat_completions_preserves_high_max_output_tokens(backend_model: str) -> None:
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


@pytest.mark.parametrize("backend_model", ["vertex_ai/claude-opus-4-6", "openai/gpt-4o"])
def test_non_gemini_chat_completions_ignore_max_output_tokens(backend_model: str) -> None:
    router: Final = _router_for_backend(backend_model, max_output_tokens=128000)
    base_messages: Final = [{"role": "user", "content": "hi"}]
    uncapped: Final = estimate_request_max_cost(
        request_body={"model": "priced-model", "messages": base_messages},
        route="/v1/chat/completions",
        llm_router=router,
    )
    decoy_only: Final = estimate_request_max_cost(
        request_body={"model": "priced-model", "messages": base_messages, "max_output_tokens": 1},
        route="/v1/chat/completions",
        llm_router=router,
    )
    one_token: Final = estimate_request_max_cost(
        request_body={"model": "priced-model", "messages": base_messages, "max_tokens": 1},
        route="/v1/chat/completions",
        llm_router=router,
    )
    shadowed_with_high_output: Final = estimate_request_max_cost(
        request_body={"model": "priced-model", "messages": base_messages, "max_tokens": 1, "max_output_tokens": 65536},
        route="/v1/chat/completions",
        llm_router=router,
    )

    assert uncapped is not None and one_token is not None
    assert decoy_only == pytest.approx(uncapped)
    assert shadowed_with_high_output == pytest.approx(one_token)


def test_mixed_openai_and_gemini_group_guards_decoy_and_high_gemini_cap() -> None:
    router: Final = Router(
        model_list=[
            Deployment(
                model_name="mixed-model",
                litellm_params=LiteLLM_Params(model="openai/gpt-4o", api_key="sk-fake"),
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


def test_direct_non_router_gemini_chat_completions_preserves_max_output_tokens() -> None:
    base_messages: Final = [{"role": "user", "content": "hi"}]
    uncapped: Final = estimate_request_max_cost(
        request_body={"model": "gemini/gemini-2.5-pro", "messages": base_messages},
        route="/v1/chat/completions",
        llm_router=None,
    )
    capped_fallback_level: Final = estimate_request_max_cost(
        request_body={"model": "gemini/gemini-2.5-pro", "messages": base_messages, "max_output_tokens": 16384},
        route="/v1/chat/completions",
        llm_router=None,
    )
    high_output: Final = estimate_request_max_cost(
        request_body={"model": "gemini/gemini-2.5-pro", "messages": base_messages, "max_output_tokens": 65536},
        route="/v1/chat/completions",
        llm_router=None,
    )

    assert uncapped is not None and capped_fallback_level is not None and high_output is not None
    assert capped_fallback_level == pytest.approx(uncapped)
    assert high_output > capped_fallback_level


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
    body: Final[dict[str, object]] = {
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
