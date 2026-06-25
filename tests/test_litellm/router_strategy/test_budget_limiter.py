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


def _embedding_success_kwargs(provider: str, response_cost: float) -> dict:
    # embedding() builds its logged litellm_params from residual **kwargs, so a
    # named custom_llm_provider is bound away and logged as None; the resolved
    # provider only survives at the top level and in the standard logging object.
    return {
        "call_type": "aembedding",
        "litellm_params": {"custom_llm_provider": None},
        "custom_llm_provider": provider,
        "standard_logging_object": {
            "response_cost": response_cost,
            "model_id": "embed-model-id",
            "custom_llm_provider": provider,
        },
    }


@pytest.mark.asyncio
async def test_async_log_success_event_resolves_provider_for_embeddings(
    disable_budget_sync,
):
    """Regression for #30276.

    Embedding success events must not raise "custom_llm_provider is required",
    and the provider budget for the resolved provider must still be enforced
    even though the logged litellm_params carry custom_llm_provider as None.
    """
    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={
            "openai": BudgetConfig(time_period="1d", budget_limit=100),
        },
    )

    kwargs = _embedding_success_kwargs(provider="openai", response_cost=0.5)

    await provider_budget.async_log_success_event(
        kwargs=kwargs, response_obj=None, start_time=0, end_time=0
    )

    spend = await provider_budget.dual_cache.async_get_cache("provider_spend:openai:1d")
    assert spend == 0.5


@pytest.mark.asyncio
async def test_async_log_success_event_skips_provider_budget_when_unresolved(
    disable_budget_sync,
):
    """The silent-skip branch of #30276.

    When no provider can be resolved from litellm_params, the top-level kwargs,
    or the standard logging object, provider-budget enforcement must be skipped
    without raising and without recording any provider spend.
    """
    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={
            "openai": BudgetConfig(time_period="1d", budget_limit=100),
        },
    )

    kwargs = {
        "call_type": "aembedding",
        "litellm_params": {"custom_llm_provider": None},
        "standard_logging_object": {
            "response_cost": 0.5,
            "model_id": "embed-model-id",
            "custom_llm_provider": None,
        },
    }

    await provider_budget.async_log_success_event(
        kwargs=kwargs, response_obj=None, start_time=0, end_time=0
    )

    spend = await provider_budget.dual_cache.async_get_cache("provider_spend:openai:1d")
    assert spend is None
