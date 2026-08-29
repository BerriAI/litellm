"""
Which model access groups a request is charged to.

A group is attributed only when its name appears on an allowlist the caller was granted, so the
group is what authorized the call. Asking for a model that merely belongs to a group attributes
nothing, and every level that can name a group (key, team, team-member scope, project, org) is
unioned rather than ranked.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import litellm
from litellm import Router
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    Litellm_EntityType,
    LiteLLM_OrganizationTable,
    LiteLLM_ProjectTableCachedObj,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import (
    _model_access_group_max_budget_check,
    collect_matched_model_access_groups,
    common_checks,
    stamp_matched_model_access_groups,
)
from litellm.proxy.common_utils.reset_budget_job import _model_access_group_counter_key
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    model_access_group_registry_cache_key,
    model_access_group_spend_counter_key,
    team_membership_reservation_cache_key,
)
from litellm.proxy.utils import ProxyLogging

TEAM_ID = "team-1"
USER_ID = "user-1"
ORG_ID = "org-1"
BUDGETED_GROUPS = ("tier-a", "tier-b", "claude-tier")
MODEL_ACCESS_GROUP_COUNTER_KEY = model_access_group_spend_counter_key("tier-a")

MODEL_LIST = [
    {
        "model_name": "gpt-4o",
        "litellm_params": {"model": "openai/gpt-4o", "api_key": "k"},
        "model_info": {"access_groups": ["tier-a", "tier-b"]},
    },
    {
        "model_name": "claude-sonnet",
        "litellm_params": {"model": "anthropic/claude-sonnet", "api_key": "k"},
        "model_info": {"access_groups": ["claude-tier"]},
    },
]


class _ExplodingPrismaClient:
    """Every lookup in these tests is served from the injected cache; a real DB read is a bug."""

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"unexpected database access: {name}")


class _CountingRouter(Router):
    """Counts access-group lookups, so a test can prove the registry gate skipped them."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.access_group_lookups = 0

    def get_model_access_groups(self, *args, **kwargs):
        self.access_group_lookups += 1
        return super().get_model_access_groups(*args, **kwargs)


async def _cache(
    budgeted_groups: tuple[str, ...] = BUDGETED_GROUPS,
    member_allowed_models: tuple[str, ...] = (),
    org_models: tuple[str, ...] = (),
) -> UserApiKeyCache:
    cache = UserApiKeyCache()
    await cache.async_set_cache(key=model_access_group_registry_cache_key(), value=budgeted_groups)
    if member_allowed_models:
        await cache.async_set_cache(
            key=team_membership_reservation_cache_key(user_id=USER_ID, team_id=TEAM_ID),
            value=LiteLLM_TeamMembership(
                user_id=USER_ID,
                team_id=TEAM_ID,
                budget_id="member-budget",
                litellm_budget_table=LiteLLM_BudgetTable(allowed_models=list(member_allowed_models)),
            ),
            model_type=LiteLLM_TeamMembership,
        )
    if org_models:
        await cache.async_set_cache(
            key=f"org_id:{ORG_ID}",
            value=LiteLLM_OrganizationTable(
                organization_id=ORG_ID,
                budget_id="org-budget",
                models=list(org_models),
                created_by=USER_ID,
                updated_by=USER_ID,
            ),
            model_type=LiteLLM_OrganizationTable,
        )
    return cache


async def _matched(
    *,
    model: str = "gpt-4o",
    key_models: list[str] | None = None,
    team_models: list[str] | None = None,
    team_org_id: str | None = None,
    project_models: list[str] | None = None,
    valid_token: UserAPIKeyAuth | None = None,
    cache: UserApiKeyCache | None = None,
    llm_router: Router | None = None,
) -> tuple[str, ...]:
    resolved_cache = cache if cache is not None else await _cache()
    return await collect_matched_model_access_groups(
        model=model,
        valid_token=valid_token
        if valid_token is not None
        else UserAPIKeyAuth(api_key="hashed", models=key_models or [], team_id=TEAM_ID, user_id=USER_ID),
        team_object=(
            LiteLLM_TeamTable(team_id=TEAM_ID, models=team_models, organization_id=team_org_id)
            if team_models is not None
            else None
        ),
        project_object=(
            LiteLLM_ProjectTableCachedObj(project_id="project-1", models=project_models)
            if project_models is not None
            else None
        ),
        llm_router=llm_router if llm_router is not None else Router(model_list=MODEL_LIST),
        prisma_client=_ExplodingPrismaClient(),
        user_api_key_cache=resolved_cache,
        proxy_logging_obj=ProxyLogging(user_api_key_cache=resolved_cache),
    )


@pytest.mark.asyncio
async def test_group_named_on_the_key_is_attributed():
    assert await _matched(key_models=["tier-a"]) == ("tier-a",)


@pytest.mark.asyncio
async def test_model_granted_directly_on_the_key_attributes_nothing():
    assert await _matched(key_models=["gpt-4o"]) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("key_models", [["*"], [], ["all-proxy-models"]])
async def test_unrestricted_key_attributes_nothing(key_models: list[str]):
    assert await _matched(key_models=key_models) == ()


@pytest.mark.asyncio
async def test_group_that_does_not_serve_the_requested_model_is_not_attributed():
    assert await _matched(model="gpt-4o", key_models=["claude-tier"]) == ()


@pytest.mark.asyncio
async def test_both_granted_groups_covering_the_model_are_attributed():
    assert await _matched(key_models=["tier-b", "tier-a"]) == ("tier-a", "tier-b")


@pytest.mark.asyncio
async def test_group_named_only_on_the_team_is_attributed():
    assert await _matched(key_models=[], team_models=["tier-a"]) == ("tier-a",)


@pytest.mark.asyncio
async def test_group_named_only_in_a_team_members_scope_is_attributed():
    assert await _matched(
        model="claude-sonnet",
        key_models=["*"],
        team_models=["*"],
        cache=await _cache(member_allowed_models=("claude-tier",)),
    ) == ("claude-tier",)


@pytest.mark.asyncio
async def test_group_named_only_on_the_project_is_attributed():
    assert await _matched(key_models=["*"], project_models=["tier-b"]) == ("tier-b",)


@pytest.mark.asyncio
async def test_group_named_only_on_the_org_is_attributed():
    assert await _matched(
        valid_token=UserAPIKeyAuth(api_key="hashed", models=["*"], user_id=USER_ID, org_id=ORG_ID),
        cache=await _cache(org_models=("tier-a",)),
    ) == ("tier-a",)


@pytest.mark.asyncio
async def test_group_named_on_the_teams_org_is_attributed_when_the_key_names_no_org():
    assert await _matched(
        key_models=["*"],
        team_models=["*"],
        team_org_id=ORG_ID,
        cache=await _cache(org_models=("tier-b",)),
    ) == ("tier-b",)


@pytest.mark.asyncio
async def test_all_team_models_sentinel_on_the_key_resolves_to_the_teams_groups():
    assert await _matched(
        valid_token=UserAPIKeyAuth(
            api_key="hashed",
            models=["all-team-models"],
            team_models=["tier-a"],
            team_id=TEAM_ID,
            user_id=USER_ID,
        ),
    ) == ("tier-a",)


@pytest.mark.asyncio
async def test_group_without_a_budget_is_not_attributed():
    assert await _matched(key_models=["tier-a"], cache=await _cache(budgeted_groups=("tier-b",))) == ()


@pytest.mark.asyncio
async def test_empty_registry_skips_the_access_group_matching_entirely():
    router = _CountingRouter(model_list=MODEL_LIST)

    assert await _matched(key_models=["tier-a"], cache=await _cache(budgeted_groups=()), llm_router=router) == ()
    assert router.access_group_lookups == 0

    assert await _matched(key_models=["tier-a"], llm_router=router) == ("tier-a",)
    assert router.access_group_lookups == 1


@pytest.mark.asyncio
async def test_stamp_records_the_matched_groups_on_the_auth_object():
    cache = await _cache()
    valid_token = UserAPIKeyAuth(api_key="hashed", models=["tier-a", "tier-b"], team_id=TEAM_ID, user_id=USER_ID)

    await stamp_matched_model_access_groups(
        model="gpt-4o",
        valid_token=valid_token,
        team_object=None,
        project_object=None,
        llm_router=Router(model_list=MODEL_LIST),
        prisma_client=_ExplodingPrismaClient(),
        user_api_key_cache=cache,
        proxy_logging_obj=ProxyLogging(user_api_key_cache=cache),
    )

    assert valid_token.matched_model_access_groups == ["tier-a", "tier-b"]


class _BrokenRouter(Router):
    def get_model_access_groups(self, *args, **kwargs):
        raise RuntimeError("access group store unavailable")


@pytest.mark.asyncio
async def test_stamp_does_not_break_auth_when_the_access_group_lookup_fails():
    cache = await _cache()
    valid_token = UserAPIKeyAuth(api_key="hashed", models=["tier-a"], team_id=TEAM_ID, user_id=USER_ID)

    await stamp_matched_model_access_groups(
        model="gpt-4o",
        valid_token=valid_token,
        team_object=None,
        project_object=None,
        llm_router=_BrokenRouter(model_list=MODEL_LIST),
        prisma_client=_ExplodingPrismaClient(),
        user_api_key_cache=cache,
        proxy_logging_obj=ProxyLogging(user_api_key_cache=cache),
    )

    assert valid_token.matched_model_access_groups is None


@pytest.mark.asyncio
async def test_stamp_leaves_the_auth_object_untouched_when_nothing_matched():
    cache = await _cache()
    valid_token = UserAPIKeyAuth(api_key="hashed", models=["gpt-4o"], team_id=TEAM_ID, user_id=USER_ID)

    await stamp_matched_model_access_groups(
        model="gpt-4o",
        valid_token=valid_token,
        team_object=None,
        project_object=None,
        llm_router=Router(model_list=MODEL_LIST),
        prisma_client=_ExplodingPrismaClient(),
        user_api_key_cache=cache,
        proxy_logging_obj=ProxyLogging(user_api_key_cache=cache),
    )

    assert valid_token.matched_model_access_groups is None


class _MagBudgetRow:
    """One ``LiteLLM_ModelAccessGroupBudgetTable`` row as prisma hands it back."""

    def __init__(self, access_group_name: str, spend: float = 0.0, max_budget: float | None = None) -> None:
        self.access_group_name = access_group_name
        self.spend = spend
        self.litellm_budget_table = None if max_budget is None else SimpleNamespace(max_budget=max_budget)


class _RecordingPrismaClient:
    """Serves budget rows and records which groups actually reached the database."""

    def __init__(self, *rows: _MagBudgetRow) -> None:
        self.rows = {row.access_group_name: row for row in rows}
        self.batches: list[list[str]] = []
        self.db = SimpleNamespace(
            litellm_modelaccessgroupbudgettable=SimpleNamespace(find_many=self._find_many)
        )

    async def _find_many(self, **kwargs):
        requested = list(kwargs["where"]["access_group_name"]["in"])
        self.batches.append(requested)
        return [self.rows[group] for group in requested if group in self.rows]


def _spend_reader(spend_by_counter_key: dict[str, float]):
    """Stand-in for proxy_server.get_current_spend, recording every counter key it is asked for."""
    seen: list[str] = []

    async def read(counter_key, fallback_spend, max_budget=None, **kwargs):
        seen.append(counter_key)
        return spend_by_counter_key.get(counter_key, fallback_spend)

    return read, seen


async def _enforce(
    matched: tuple[str, ...],
    *rows: _MagBudgetRow,
    spend_by_counter_key: dict[str, float] | None = None,
    prisma_client: object | None = None,
    cache: UserApiKeyCache | None = None,
) -> list[str]:
    read, seen = _spend_reader(spend_by_counter_key or {})
    with patch("litellm.proxy.proxy_server.get_current_spend", read):
        await _model_access_group_max_budget_check(
            matched_model_access_groups=matched,
            prisma_client=prisma_client if prisma_client is not None else _RecordingPrismaClient(*rows),
            user_api_key_cache=cache if cache is not None else UserApiKeyCache(),
        )
    return seen


@pytest.mark.asyncio
async def test_group_under_its_max_budget_passes():
    assert await _enforce(
        ("tier-a",),
        _MagBudgetRow("tier-a", spend=4.0, max_budget=10.0),
        spend_by_counter_key={MODEL_ACCESS_GROUP_COUNTER_KEY: 4.0},
    ) == [MODEL_ACCESS_GROUP_COUNTER_KEY]


@pytest.mark.asyncio
async def test_group_exactly_at_its_max_budget_passes():
    """The ceiling is inclusive, matching the tag check it mirrors; only spend strictly above it blocks."""
    await _enforce(
        ("tier-a",),
        _MagBudgetRow("tier-a", max_budget=10.0),
        spend_by_counter_key={MODEL_ACCESS_GROUP_COUNTER_KEY: 10.0},
    )


@pytest.mark.asyncio
async def test_group_over_its_max_budget_blocks_the_request_and_names_the_group():
    with pytest.raises(litellm.BudgetExceededError) as exc_info:
        await _enforce(
            ("tier-a",),
            _MagBudgetRow("tier-a", max_budget=10.0),
            spend_by_counter_key={MODEL_ACCESS_GROUP_COUNTER_KEY: 10.5},
        )

    assert exc_info.value.entity_id == "tier-a"
    assert exc_info.value.entity_type == Litellm_EntityType.MODEL_ACCESS_GROUP.value
    assert exc_info.value.current_cost == 10.5
    assert exc_info.value.max_budget == 10.0
    assert "tier-a" in str(exc_info.value)


@pytest.mark.asyncio
async def test_group_with_a_row_but_no_budget_never_blocks():
    """An admin can register a group without a ceiling; that must not become an implicit zero budget."""
    assert (
        await _enforce(
            ("tier-a",),
            _MagBudgetRow("tier-a", spend=9999.0),
            spend_by_counter_key={MODEL_ACCESS_GROUP_COUNTER_KEY: 9999.0},
        )
        == []
    )


@pytest.mark.asyncio
async def test_a_cold_counter_falls_back_to_the_spend_recorded_on_the_row():
    """After a counter expires the DB row is the only record of the spend, so it has to be read."""
    with pytest.raises(litellm.BudgetExceededError) as exc_info:
        await _enforce(("tier-a",), _MagBudgetRow("tier-a", spend=12.0, max_budget=10.0))

    assert exc_info.value.current_cost == 12.0


@pytest.mark.asyncio
async def test_an_over_budget_group_blocks_even_when_another_matched_group_is_fine():
    with pytest.raises(litellm.BudgetExceededError) as exc_info:
        await _enforce(
            ("tier-a", "tier-b"),
            _MagBudgetRow("tier-a", max_budget=10.0),
            _MagBudgetRow("tier-b", max_budget=1.0),
            spend_by_counter_key={
                MODEL_ACCESS_GROUP_COUNTER_KEY: 1.0,
                model_access_group_spend_counter_key("tier-b"): 5.0,
            },
        )

    assert exc_info.value.entity_id == "tier-b"


@pytest.mark.asyncio
async def test_request_that_matched_no_group_touches_neither_database_nor_counters():
    assert await _enforce((), prisma_client=_ExplodingPrismaClient()) == []


@pytest.mark.asyncio
async def test_budget_check_reads_the_counter_key_the_reset_job_clears():
    """Reads and resets must agree, or a rollover clears a counter nobody reads."""
    reset_job_key = _model_access_group_counter_key(SimpleNamespace(access_group_name="tier-a"))

    assert await _enforce(("tier-a",), _MagBudgetRow("tier-a", max_budget=10.0)) == [reset_job_key]


@pytest.mark.asyncio
async def test_a_second_request_serves_the_budget_row_from_cache():
    cache = UserApiKeyCache()
    prisma_client = _RecordingPrismaClient(_MagBudgetRow("tier-a", max_budget=10.0))

    await _enforce(("tier-a",), prisma_client=prisma_client, cache=cache)
    await _enforce(("tier-a",), prisma_client=prisma_client, cache=cache)

    assert prisma_client.batches == [["tier-a"]]


@pytest.mark.asyncio
async def test_a_database_error_does_not_block_the_request():
    class _FailingPrismaClient:
        def __init__(self) -> None:
            self.db = SimpleNamespace(
                litellm_modelaccessgroupbudgettable=SimpleNamespace(find_many=self._boom)
            )

        async def _boom(self, **kwargs):
            raise RuntimeError("database unavailable")

    assert await _enforce(("tier-a",), prisma_client=_FailingPrismaClient()) == []


async def _common_checks_with_over_budget_group(*, skip_budget_checks: bool) -> bool:
    cache = await _cache()
    prisma_client = _RecordingPrismaClient(_MagBudgetRow("tier-a", max_budget=1.0))
    read, _ = _spend_reader({MODEL_ACCESS_GROUP_COUNTER_KEY: 99.0})

    with (
        patch("litellm.proxy.proxy_server.prisma_client", prisma_client),
        patch("litellm.proxy.proxy_server.user_api_key_cache", cache),
        patch("litellm.proxy.proxy_server.get_current_spend", read),
        patch("litellm.proxy.auth.auth_checks._is_api_route_allowed", return_value=True),
    ):
        return await common_checks(
            request_body={"model": "gpt-4o", "messages": []},
            team_object=None,
            user_object=None,
            end_user_object=None,
            global_proxy_spend=None,
            general_settings={},
            route="/v1/chat/completions",
            llm_router=Router(model_list=MODEL_LIST),
            proxy_logging_obj=ProxyLogging(user_api_key_cache=cache),
            valid_token=UserAPIKeyAuth(api_key="hashed", models=["tier-a"], user_id=USER_ID),
            request=SimpleNamespace(method="POST", headers={}, query_params={}, url=SimpleNamespace(path="/v1/chat/completions")),
            skip_budget_checks=skip_budget_checks,
        )


@pytest.mark.asyncio
async def test_common_checks_blocks_a_request_whose_group_is_over_budget():
    with pytest.raises(litellm.BudgetExceededError) as exc_info:
        await _common_checks_with_over_budget_group(skip_budget_checks=False)

    assert exc_info.value.entity_id == "tier-a"


@pytest.mark.asyncio
async def test_free_model_routes_skip_the_model_access_group_budget_check():
    """skip_budget_checks is how free models stay free; it has to cover this budget too."""
    assert await _common_checks_with_over_budget_group(skip_budget_checks=True) is True
