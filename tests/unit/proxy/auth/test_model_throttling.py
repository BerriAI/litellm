from typing import Final

import pytest

from litellm import Router
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.budget_throttle import should_throttle_budget_exceeded
from litellm.proxy.auth.fallback_model_access import RouterFallbackAccessCheck
from litellm.proxy.auth.model_throttling import ModelThrottleSource, enforce_model_throttling, throttles_at_spend
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.types.proxy.model_throttling import ModelThrottling, ModelThrottlingThreshold

CONFIG: Final = ModelThrottling(
    thresholds=(
        ModelThrottlingThreshold(spend=60, models=("premium",)),
        ModelThrottlingThreshold(spend=85, models=("standard",)),
    )
)
SOURCE: Final = ModelThrottleSource("key", "spend:key:test", 0, 100, CONFIG)


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
