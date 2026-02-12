import pytest

import litellm
from litellm.caching.caching import DualCache
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.router import LiteLLM_Params
from litellm.types.utils import BudgetConfig


@pytest.fixture
def disable_budget_sync(monkeypatch):
    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "litellm.router_strategy.budget_limiter.RouterBudgetLimiting.periodic_sync_in_memory_spend_with_redis",
        noop,
    )


@pytest.mark.asyncio
async def test_get_llm_provider_for_deployment_dict_does_not_require_litellm_params_instantiation(
    disable_budget_sync, monkeypatch
):
    class RaiseOnInit:
        def __init__(self, *args, **kwargs):
            raise AssertionError("LiteLLM_Params should not be instantiated in hot path")

    monkeypatch.setattr(
        "litellm.router_strategy.budget_limiter.LiteLLM_Params",
        RaiseOnInit,
    )

    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={},
    )

    deployment = {"litellm_params": {"model": "openai/gpt-4o-mini"}}
    provider = provider_budget._get_llm_provider_for_deployment(deployment)

    assert provider == "openai"


@pytest.mark.asyncio
async def test_get_llm_provider_for_deployment_dict_view_supports_mapping_and_attr_access(
    disable_budget_sync, monkeypatch
):
    observed = {}

    def _future_style_get_llm_provider(
        model,
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
        litellm_params=None,
    ):
        assert litellm_params is not None
        observed["model_attr"] = litellm_params.model
        observed["provider_get"] = litellm_params.get("custom_llm_provider")
        observed["api_base_item"] = litellm_params["api_base"]
        observed["has_api_key"] = "api_key" in litellm_params
        observed["model_dump"] = litellm_params.model_dump()
        return model, "openai", None, None

    monkeypatch.setattr(
        "litellm.router_strategy.budget_limiter.litellm.get_llm_provider",
        _future_style_get_llm_provider,
    )

    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={},
    )

    deployment = {
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "custom_llm_provider": "openai",
            "api_base": "https://api.openai.com/v1",
        }
    }
    provider = provider_budget._get_llm_provider_for_deployment(deployment)

    assert provider == "openai"
    assert observed["model_attr"] == "openai/gpt-4o-mini"
    assert observed["provider_get"] == "openai"
    assert observed["api_base_item"] == "https://api.openai.com/v1"
    assert observed["has_api_key"] is False
    assert observed["model_dump"]["model"] == "openai/gpt-4o-mini"


@pytest.mark.asyncio
async def test_async_filter_deployments_resolves_provider_once_per_deployment(
    disable_budget_sync, monkeypatch
):
    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={
            "openai": BudgetConfig(budget_duration="1d", max_budget=100.0),
        },
    )

    healthy_deployments = [
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {"model": "openai/gpt-4o-mini"},
            "model_info": {"id": "deployment-1"},
        },
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {"model": "openai/gpt-4o-mini"},
            "model_info": {"id": "deployment-2"},
        },
    ]

    provider_resolution_calls = 0

    def _count_provider_calls(deployment):
        nonlocal provider_resolution_calls
        provider_resolution_calls += 1
        return "openai"

    monkeypatch.setattr(
        provider_budget,
        "_get_llm_provider_for_deployment",
        _count_provider_calls,
    )

    filtered_deployments = await provider_budget.async_filter_deployments(
        model="gpt-4o-mini",
        healthy_deployments=healthy_deployments,
        messages=[],
        request_kwargs={},
        parent_otel_span=None,
    )

    assert len(filtered_deployments) == len(healthy_deployments)
    assert provider_resolution_calls == len(healthy_deployments)


@pytest.mark.asyncio
async def test_async_filter_deployments_does_not_recompute_provider_when_resolved_none(
    disable_budget_sync, monkeypatch
):
    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={
            "openai": BudgetConfig(budget_duration="1d", max_budget=100.0),
        },
        model_list=[
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "max_budget": 100.0,
                    "budget_duration": "1d",
                },
                "model_info": {"id": "deployment-1"},
            }
        ],
    )

    healthy_deployments = [
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {"model": "unknown-provider/model"},
            "model_info": {"id": "deployment-1"},
        }
    ]

    provider_resolution_calls = 0

    def _provider_returns_none(deployment):
        nonlocal provider_resolution_calls
        provider_resolution_calls += 1
        return None

    monkeypatch.setattr(
        provider_budget,
        "_get_llm_provider_for_deployment",
        _provider_returns_none,
    )

    filtered_deployments = await provider_budget.async_filter_deployments(
        model="gpt-4o-mini",
        healthy_deployments=healthy_deployments,
        messages=[],
        request_kwargs={},
        parent_otel_span=None,
    )

    assert len(filtered_deployments) == len(healthy_deployments)
    assert provider_resolution_calls == len(healthy_deployments)


def _legacy_provider_resolution(deployment):
    """
    Reference implementation used before hot-path optimization.
    """
    try:
        _litellm_params = LiteLLM_Params(**deployment.get("litellm_params", {"model": ""}))
        _, custom_llm_provider, _, _ = litellm.get_llm_provider(
            model=_litellm_params.model,
            litellm_params=_litellm_params,
        )
    except Exception:
        return None
    return custom_llm_provider


@pytest.mark.parametrize(
    "deployment",
    [
        {"litellm_params": {"model": "openai/gpt-4o-mini"}},
        {"litellm_params": {"model": "gpt-4o-mini", "custom_llm_provider": "openai"}},
        {"litellm_params": {"model": "unknown-provider/model"}},
    ],
)
@pytest.mark.asyncio
async def test_get_llm_provider_for_deployment_matches_legacy_behavior(
    disable_budget_sync, deployment
):
    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={},
    )

    current_provider = provider_budget._get_llm_provider_for_deployment(deployment)
    legacy_provider = _legacy_provider_resolution(deployment)

    assert current_provider == legacy_provider
