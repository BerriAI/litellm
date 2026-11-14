from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Final

import pytest

import litellm
from litellm.caching import DualCache
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.proxy import proxy_server
from litellm.proxy._types import (
    LiteLLM_OrganizationTable,
    LiteLLM_ProjectTableCachedObj,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    project_cache_key,
)
from litellm.proxy.common_utils.timezone_utils import BudgetResetSettings
from litellm.proxy.spend_tracking.budget_reservation import (
    release_unbound_budget_reservation,
)
from litellm.proxy.utils import ProxyLogging
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore
from tests.unit.proxy.spend_tracking.native_budgets_replica import install_native_owner, reserve_budget_for_request

pytestmark = pytest.mark.usefixtures("native_budgets_on")

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
    _ = install_native_owner(store)
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


LEGACY_KEY: Final = "spend:key:legacy"
LEGACY_TEAM: Final = "spend:team:legacy-team"
LEGACY_WINDOW: Final = "spend:key:legacy:window:1d"


def _legacy_reservation() -> dict[str, object]:
    return {
        "reserved_cost": 0.4,
        "entries": [
            {
                "counter_key": LEGACY_KEY,
                "entity_type": "Key",
                "entity_id": "legacy",
                "reserved_cost": 0.4,
                "applied_adjustment": 0.0,
            },
            {
                "counter_key": LEGACY_TEAM,
                "entity_type": "Team",
                "entity_id": "legacy-team",
                "reserved_cost": 0.4,
                "applied_adjustment": 0.0,
            },
            {
                "counter_key": LEGACY_WINDOW,
                "entity_type": "Key",
                "entity_id": "legacy",
                "reserved_cost": 0.4,
                "applied_adjustment": 0.0,
                "window_reset_at": "2026-01-01T00:00:00+00:00",
            },
        ],
        "finalized": False,
        "callback_bound": True,
        "input_cost": 0.1,
        "input_tokens": 12,
    }


def _held_by_a_legacy_reservation(store: SpendCounterStore) -> None:
    _ = install_native_owner(store)
    store.set_local(LEGACY_KEY, 1.4)
    store.set_local(LEGACY_TEAM, 2.4)
    store.set_local(LEGACY_WINDOW, 0.9)


def _legacy_counters(store: SpendCounterStore) -> dict[str, float | None]:
    return {key: store.get_local(key) for key in (LEGACY_KEY, LEGACY_TEAM, LEGACY_WINDOW)}


def _legacy_adjustments(reservation: dict[str, object]) -> list[object]:
    entries: Final = reservation["entries"]
    assert isinstance(entries, list)
    return [entry["applied_adjustment"] for entry in entries]


@pytest.mark.asyncio
async def test_a_reservation_the_python_engine_held_before_the_upgrade_settles_on_the_post_call_cost(
    spend_counter_store: SpendCounterStore,
):
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy.hooks.proxy_track_cost_callback import _update_database_and_spend_counters

    _held_by_a_legacy_reservation(spend_counter_store)
    reservation: Final = _legacy_reservation()
    proxy_logging_obj: Final = MagicMock()
    proxy_logging_obj.db_spend_update_writer.update_database = AsyncMock(return_value=True)

    charged: Final = await _update_database_and_spend_counters(
        proxy_logging_obj=proxy_logging_obj,
        increment_spend_counters=proxy_server.increment_spend_counters,
        user_api_key="legacy",
        user_id=None,
        end_user_id=None,
        team_id="legacy-team",
        org_id=None,
        kwargs={},
        completion_response=None,
        start_time=None,
        end_time=None,
        response_cost=0.25,
        budget_reservation=reservation,
        request_tags=None,
        model_access_groups=None,
    )

    assert charged is True
    assert _legacy_counters(spend_counter_store) == pytest.approx(
        {LEGACY_KEY: 1.25, LEGACY_TEAM: 2.25, LEGACY_WINDOW: 0.5}
    )
    assert _legacy_adjustments(reservation) == pytest.approx([-0.15, -0.15, -0.4])
    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_a_reservation_the_python_engine_held_before_the_upgrade_is_released_in_full(
    spend_counter_store: SpendCounterStore,
):
    from litellm.proxy.spend_tracking.budget_reservation import release_budget_reservation

    _held_by_a_legacy_reservation(spend_counter_store)
    reservation: Final = _legacy_reservation()

    await release_budget_reservation(reservation)

    assert _legacy_counters(spend_counter_store) == pytest.approx(
        {LEGACY_KEY: 1.0, LEGACY_TEAM: 2.0, LEGACY_WINDOW: 0.5}
    )
    assert _legacy_adjustments(reservation) == pytest.approx([-0.4, -0.4, -0.4])
    assert reservation["finalized"] is True
