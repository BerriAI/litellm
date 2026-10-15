from typing import Final

import pytest

from litellm import Router
from litellm.caching.caching import DualCache
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_EndUserTable,
    LiteLLM_OrganizationTable,
    LiteLLM_ProjectTableCachedObj,
    LiteLLM_TagTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.budget_throttle import should_throttle_budget_exceeded
from litellm.proxy.auth.fallback_model_access import RouterFallbackAccessCheck
from litellm.proxy.auth.model_throttling import (
    ModelThrottleSource,
    enforce_model_throttling,
    load_model_throttles,
    throttles_at_spend,
)
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    model_access_group_cache_key,
    model_access_group_spend_counter_key,
    project_spend_counter_key,
    tag_cache_key,
)
from litellm.types.proxy.model_access_group_budget import ModelAccessGroupBudget
from litellm.types.proxy.model_throttling import ModelThrottling, ModelThrottlingThreshold

CONFIG: Final = ModelThrottling(
    thresholds=(
        ModelThrottlingThreshold(spend=60, models=("premium",)),
        ModelThrottlingThreshold(spend=85, models=("standard",)),
    )
)
SOURCE: Final = ModelThrottleSource("key", "spend:key:test", 0, 100, CONFIG)


class _NoDatabaseReads:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"Unexpected database access for cached budget: {name}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "counter_key"),
    (
        ("key", "spend:key:throttling-key"),
        ("team", "spend:team:throttling-team"),
        ("user", "spend:user:throttling-user"),
        ("project", project_spend_counter_key("throttling-project")),
        ("end user", "spend:end_user:throttling-customer"),
        ("organization", "spend:org:throttling-org"),
        ("team member", "spend:team_member:throttling-user:throttling-team"),
        ("default team member", "spend:team_member:throttling-user:throttling-team"),
        ("tag", "spend:tag:throttling-tag"),
        ("model access group", model_access_group_spend_counter_key("throttling-group")),
    ),
)
async def test_each_budget_scope_pauses_models_at_live_spend_and_restores_after_reset(
    scope: str, counter_key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy import proxy_server

    cache: Final = UserApiKeyCache()
    counters: Final = DualCache()
    budget: Final = LiteLLM_BudgetTable(max_budget=100, model_throttling=CONFIG)
    monkeypatch.setattr(proxy_server, "prisma_client", _NoDatabaseReads())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "spend_counter_cache", counters)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    await cache.async_set_cache(
        "org_id:throttling-org:with_budget",
        LiteLLM_OrganizationTable(
            organization_id="throttling-org",
            budget_id="org-budget",
            created_by="admin",
            updated_by="admin",
            litellm_budget_table=budget,
        ),
        model_type=LiteLLM_OrganizationTable,
    )
    await cache.async_set_cache("team_member_default_budget:member-budget", budget, model_type=LiteLLM_BudgetTable)
    await cache.async_set_cache(
        tag_cache_key("throttling-tag"),
        LiteLLM_TagTable(tag_name="throttling-tag", litellm_budget_table=budget),
        model_type=LiteLLM_TagTable,
    )
    await cache.async_set_cache(
        model_access_group_cache_key("throttling-group"),
        ModelAccessGroupBudget(access_group_name="throttling-group", max_budget=100, model_throttling=CONFIG),
        model_type=ModelAccessGroupBudget,
    )
    for key in (
        "spend:key:throttling-key",
        "spend:team:throttling-team",
        "spend:user:throttling-user",
        project_spend_counter_key("throttling-project"),
        "spend:end_user:throttling-customer",
        "spend:org:throttling-org",
        "spend:team_member:throttling-user:throttling-team",
        "spend:tag:throttling-tag",
        model_access_group_spend_counter_key("throttling-group"),
    ):
        await counters.async_set_cache(key, 60 if key == counter_key else 0)
    token: Final = UserAPIKeyAuth(
        token="throttling-key",
        team_id="throttling-team",
        user_id="throttling-user",
        max_budget=100,
        model_throttling=CONFIG,
        matched_model_access_groups=["throttling-group"],
    )
    for reset in (False, True):
        if reset:
            await counters.async_set_cache(counter_key, 0)
        active: Final = await load_model_throttles(
            valid_token=token,
            team=LiteLLM_TeamTable(
                team_id="throttling-team",
                organization_id="throttling-org",
                max_budget=100,
                model_throttling=CONFIG,
                metadata={"team_member_budget_id": "member-budget"},
            ),
            user=LiteLLM_UserTable(user_id="throttling-user", max_budget=100, model_throttling=CONFIG),
            end_user=LiteLLM_EndUserTable(user_id="throttling-customer", blocked=False, litellm_budget_table=budget),
            project=LiteLLM_ProjectTableCachedObj(project_id="throttling-project", litellm_budget_table=budget),
            membership=LiteLLM_TeamMembership(
                user_id="throttling-user",
                team_id="throttling-team",
                litellm_budget_table=None if scope == "default team member" else budget,
            ),
            apply_user_budget_to_team_keys=True,
            tags=("throttling-tag",),
        )
        request: Final = token.model_copy(update={"active_model_throttles": active})
        assert enforce_model_throttling(("economy",), request, None) is None
        if reset:
            assert enforce_model_throttling(("premium",), request, None) is None
        else:
            with pytest.raises(ProxyException, match=f"paused by {scope.removeprefix('default ')} model throttling"):
                enforce_model_throttling(("premium",), request, None)


@pytest.mark.parametrize(
    ("spend", "paused"),
    ((0, ()), (59.99, ()), (60, ("premium",)), (84.99, ("premium",)), (85, ("premium", "standard"))),
)
def test_thresholds_pause_only_selected_models_and_accumulate(spend: float, paused: tuple[str, ...]) -> None:
    token: Final = UserAPIKeyAuth(active_model_throttles=throttles_at_spend(SOURCE, spend))
    for model in ("premium", "standard", "economy"):
        if model in paused:
            with pytest.raises(ProxyException, match="Choose another available model") as error:
                enforce_model_throttling((model,), token, None)
            assert (error.value.code, error.value.type, error.value.param) == ("429", "model_throttled", "model")
        else:
            assert enforce_model_throttling((model,), token, None) is None


@pytest.mark.asyncio
async def test_reset_does_not_restore_cached_request_restrictions() -> None:
    cache: Final = UserApiKeyCache()
    reached: Final = UserAPIKeyAuth(model_throttling=CONFIG, active_model_throttles=throttles_at_spend(SOURCE, 85))
    await cache.async_set_cache("budget-key", reached, model_type=UserAPIKeyAuth)
    restored: Final = await cache.async_get_cache("budget-key", model_type=UserAPIKeyAuth)
    assert restored is not None
    assert restored.model_throttling == CONFIG
    assert restored.active_model_throttles == ()
    assert throttles_at_spend(SOURCE, 0) == ()
    assert enforce_model_throttling(("premium", "standard"), restored, None) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback_access_enabled", (True, False))
async def test_alias_and_deployment_fallbacks_cannot_bypass_throttling(fallback_access_enabled: bool) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "premium",
                "litellm_params": {"model": "openai/premium", "api_key": "test"},
                "model_info": {"id": "premium-deployment"},
            },
            {"model_name": "economy", "litellm_params": {"model": "openai/economy", "api_key": "test"}},
        ],
        model_group_alias={"expensive-alias": "premium"},
    )
    token: Final = UserAPIKeyAuth(
        models=["premium", "economy"],
        aliases={"key-alias": "expensive-alias"},
        active_model_throttles=throttles_at_spend(SOURCE, 60),
    )
    check: Final = RouterFallbackAccessCheck(is_enforced=lambda: fallback_access_enabled)
    for model in ("premium", "expensive-alias", "key-alias", "premium-deployment"):
        with pytest.raises(ProxyException, match="paused by key model throttling"):
            enforce_model_throttling((model,), token, router)
        assert not await check(
            model=model, request_kwargs={"metadata": {"user_api_key_auth": token}}, llm_router=router
        )
    assert await check(model="economy", request_kwargs={"metadata": {"user_api_key_auth": token}}, llm_router=router)


def test_model_throttling_preserves_final_cap_with_legacy_rate_throttling_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("litellm.budget_exceeded_throttle_percentage", 0.5)
    legacy: Final = UserAPIKeyAuth(metadata={"throttle_on_budget_exceeded": True}, tpm_limit=1000)
    assert should_throttle_budget_exceeded(legacy)
    assert not should_throttle_budget_exceeded(legacy.model_copy(update={"model_throttling": CONFIG}))


@pytest.mark.asyncio
async def test_selected_deployment_cannot_bypass_rules_through_a_routing_group(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.auth.model_throttling import ModelThrottlingHook

    router: Final = Router(
        model_list=[
            {
                "model_name": "premium",
                "litellm_params": {"model": "openai/gpt-6-astra", "api_key": "test"},
                "model_info": {"id": "premium-deployment"},
            },
            {
                "model_name": "economy",
                "litellm_params": {"model": "openai/gpt-6-luna", "api_key": "test"},
                "model_info": {"id": "economy-deployment"},
            },
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    token: Final = UserAPIKeyAuth(active_model_throttles=throttles_at_spend(SOURCE, 60))
    hook: Final = ModelThrottlingHook()
    for metadata_field in ("metadata", "litellm_metadata"):
        with pytest.raises(ProxyException, match="paused by key model throttling"):
            await hook.async_pre_call_deployment_hook(
                {"model_info": {"id": "premium-deployment"}, metadata_field: {"user_api_key_auth": token}}, None
            )
        assert (
            await hook.async_pre_call_deployment_hook(
                {"model_info": {"id": "economy-deployment"}, metadata_field: {"user_api_key_auth": token}}, None
            )
            is None
        )
    assert await hook.async_pre_call_deployment_hook({"model_info": {"id": "premium-deployment"}}, None) is None
