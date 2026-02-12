import pytest

from litellm.caching.caching import DualCache
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
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
