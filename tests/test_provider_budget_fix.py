"""Regression test for https://github.com/BerriAI/litellm/issues/33327"""
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.utils import BudgetConfig


def test_duration_only_provider_config_keeps_deployment():
    """
    A provider budget entry with duration but no max_budget should NOT
    remove deployments from the healthy set.
    """
    limiter = RouterBudgetLimiting.__new__(RouterBudgetLimiting)
    limiter.provider_budget_config = {"openai": BudgetConfig(budget_duration="1d", max_budget=None)}
    limiter.deployment_budget_config = None
    limiter.tag_budget_config = None

    healthy_deployments = [{
        "model_name": "chat",
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "custom_llm_provider": "openai",
        },
        "model_info": {"id": "deployment-1"},
    }]

    provider_configs = {"openai": limiter.provider_budget_config["openai"]}

    result, _ = limiter._filter_out_deployments_above_budget(
        potential_deployments=healthy_deployments,
        healthy_deployments=healthy_deployments,
        provider_configs=provider_configs,
        deployment_configs={},
        deployment_providers=["openai"],
        spend_map={},
        request_tags=[],
    )

    assert len(result) == 1, (
        f"Expected 1 deployment, got {len(result)}. "
        "duration-only provider config should not remove deployments."
    )
