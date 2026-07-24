"""Regression tests for RouterBudgetLimiting._filter_out_deployments_above_budget."""

from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.utils import BudgetConfig


def _make_limiter(provider_budget_config):
    limiter = RouterBudgetLimiting.__new__(RouterBudgetLimiting)
    limiter.provider_budget_config = provider_budget_config
    limiter.deployment_budget_config = None
    limiter.tag_budget_config = None
    return limiter


def _openai_deployment():
    return {
        "model_name": "chat",
        "litellm_params": {"model": "openai/gpt-4o-mini", "custom_llm_provider": "openai"},
        "model_info": {"id": "deployment-1"},
    }


def test_duration_only_provider_config_keeps_deployment():
    """A provider budget with budget_duration but no max_budget must NOT drop deployments.

    Regression test for https://github.com/BerriAI/litellm/issues/33327
    A missing max_budget means there is no numeric cap, not an exceeded budget.
    """
    limiter = _make_limiter({"openai": BudgetConfig(budget_duration="1d", max_budget=None)})

    result, info = limiter._filter_out_deployments_above_budget(
        potential_deployments=[],
        healthy_deployments=[_openai_deployment()],
        provider_configs={"openai": limiter.provider_budget_config["openai"]},
        deployment_configs={},
        deployment_providers=["openai"],
        spend_map={},
        request_tags=[],
    )

    assert len(result) == 1
    assert info == ""


def test_provider_over_max_budget_filters_deployment():
    """A numeric max_budget that is exceeded must still drop the deployment.

    Guards against the duration-only fix accidentally disabling budget enforcement.
    """
    limiter = _make_limiter({"openai": BudgetConfig(budget_duration="1d", max_budget=10.0)})

    result, info = limiter._filter_out_deployments_above_budget(
        potential_deployments=[],
        healthy_deployments=[_openai_deployment()],
        provider_configs={"openai": limiter.provider_budget_config["openai"]},
        deployment_configs={},
        deployment_providers=["openai"],
        spend_map={"provider_spend:openai:1d": 25.0},
        request_tags=[],
    )

    assert len(result) == 0
    assert "Exceeded budget for provider openai" in info


def test_provider_under_max_budget_keeps_deployment():
    """A numeric max_budget that is not exceeded keeps the deployment eligible."""
    limiter = _make_limiter({"openai": BudgetConfig(budget_duration="1d", max_budget=10.0)})

    result, info = limiter._filter_out_deployments_above_budget(
        potential_deployments=[],
        healthy_deployments=[_openai_deployment()],
        provider_configs={"openai": limiter.provider_budget_config["openai"]},
        deployment_configs={},
        deployment_providers=["openai"],
        spend_map={"provider_spend:openai:1d": 3.0},
        request_tags=[],
    )

    assert len(result) == 1
    assert info == ""
