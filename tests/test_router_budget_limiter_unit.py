"""
Unit tests for RouterBudgetLimiting._filter_out_deployments_above_budget.

These tests run without Redis or real API calls.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(".."))

import pytest
from unittest.mock import MagicMock, patch
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.utils import BudgetConfig
from litellm.caching.caching import DualCache


def _make_limiter(provider_budget_config=None, deployment_budget_config=None):
    """Create a RouterBudgetLimiting instance with in-memory cache only."""
    dual_cache = DualCache()
    with patch("asyncio.create_task"):
        limiter = RouterBudgetLimiting.__new__(RouterBudgetLimiting)
        limiter.dual_cache = dual_cache
        limiter.redis_increment_operation_queue = []
        limiter.provider_budget_config = provider_budget_config
        limiter.deployment_budget_config = deployment_budget_config
        limiter.tag_budget_config = None
        limiter._track_provider_remaining_budget_prometheus = MagicMock()
        limiter._get_budget_config_for_tag = MagicMock(return_value=None)
        limiter._get_llm_provider_for_deployment = MagicMock(return_value="openai")
    return limiter


_DEPLOYMENT = {
    "model_name": "gpt-4",
    "litellm_params": {"model": "openai/gpt-4"},
    "model_info": {"id": "dep-1"},
}


# ---------------------------------------------------------------------------
# Bug 1: provider with max_budget=None must NOT be excluded from routing
# ---------------------------------------------------------------------------

class TestProviderNullBudget:
    """
    A provider entry with max_budget=None means 'no cap' — deployments
    belonging to that provider should always pass the budget filter.

    Before the fix, `if config.max_budget is None: continue` caused the
    loop to skip appending the deployment, effectively blocking all traffic
    to providers configured without an explicit budget limit.
    """

    def test_null_max_budget_deployment_is_included(self):
        """Deployment for a provider with max_budget=None must be returned."""
        limiter = _make_limiter(
            provider_budget_config={
                "openai": BudgetConfig(time_period="1d", budget_limit=None)
            }
        )
        provider_configs = {
            "openai": BudgetConfig(time_period="1d", budget_limit=None)
        }
        result, _ = limiter._filter_out_deployments_above_budget(
            potential_deployments=[],
            healthy_deployments=[_DEPLOYMENT],
            provider_configs=provider_configs,
            deployment_configs={},
            deployment_providers=["openai"],
            spend_map={},
            request_tags=[],
        )
        assert len(result) == 1, (
            "Deployment with provider max_budget=None should NOT be excluded. "
            f"Got: {result}"
        )

    def test_null_max_budget_skips_prometheus_tracking(self):
        """_track_provider_remaining_budget_prometheus must not be called when max_budget is None."""
        limiter = _make_limiter(
            provider_budget_config={
                "openai": BudgetConfig(time_period="1d", budget_limit=None)
            }
        )
        provider_configs = {
            "openai": BudgetConfig(time_period="1d", budget_limit=None)
        }
        limiter._filter_out_deployments_above_budget(
            potential_deployments=[],
            healthy_deployments=[_DEPLOYMENT],
            provider_configs=provider_configs,
            deployment_configs={},
            deployment_providers=["openai"],
            spend_map={},
            request_tags=[],
        )
        limiter._track_provider_remaining_budget_prometheus.assert_not_called()

    def test_zero_spend_against_explicit_budget_is_included(self):
        """Sanity-check: a deployment well under its budget is included."""
        limiter = _make_limiter(
            provider_budget_config={
                "openai": BudgetConfig(time_period="1d", budget_limit=100.0)
            }
        )
        provider_configs = {
            "openai": BudgetConfig(time_period="1d", budget_limit=100.0)
        }
        spend_map = {"provider_spend:openai:1d": 0.5}
        result, _ = limiter._filter_out_deployments_above_budget(
            potential_deployments=[],
            healthy_deployments=[_DEPLOYMENT],
            provider_configs=provider_configs,
            deployment_configs={},
            deployment_providers=["openai"],
            spend_map=spend_map,
            request_tags=[],
        )
        assert len(result) == 1

    def test_over_budget_deployment_is_excluded(self):
        """A deployment whose provider spend exceeds the limit is excluded."""
        limiter = _make_limiter(
            provider_budget_config={
                "openai": BudgetConfig(time_period="1d", budget_limit=1.0)
            }
        )
        provider_configs = {
            "openai": BudgetConfig(time_period="1d", budget_limit=1.0)
        }
        spend_map = {"provider_spend:openai:1d": 2.0}
        result, debug_info = limiter._filter_out_deployments_above_budget(
            potential_deployments=[],
            healthy_deployments=[_DEPLOYMENT],
            provider_configs=provider_configs,
            deployment_configs={},
            deployment_providers=["openai"],
            spend_map=spend_map,
            request_tags=[],
        )
        assert len(result) == 0
        assert "Exceeded budget for provider openai" in debug_info


# ---------------------------------------------------------------------------
# Bug 2: deployment budget exceeded message must reference max_budget, not budget_duration
# ---------------------------------------------------------------------------

class TestDeploymentBudgetDebugMessage:
    """
    The 'Exceeded budget for deployment' log message used to interpolate
    `config.budget_duration` (a time string like "1d") instead of
    `config.max_budget` (the dollar limit), making the message misleading
    and impossible to interpret for operators reading logs.
    """

    def test_exceeded_message_contains_max_budget_not_duration(self):
        """Debug message must show the dollar limit, not the time period."""
        dep = {
            "model_name": "gpt-4",
            "litellm_params": {"model": "openai/gpt-4"},
            "model_info": {"id": "dep-42"},
        }
        limiter = _make_limiter(
            deployment_budget_config={
                "dep-42": BudgetConfig(time_period="7d", budget_limit=5.0)
            }
        )
        deployment_configs = {
            "dep-42": BudgetConfig(time_period="7d", budget_limit=5.0)
        }
        spend_map = {"deployment_spend:dep-42:7d": 10.0}

        _, debug_info = limiter._filter_out_deployments_above_budget(
            potential_deployments=[],
            healthy_deployments=[dep],
            provider_configs={},
            deployment_configs=deployment_configs,
            deployment_providers=[],
            spend_map=spend_map,
            request_tags=[],
        )

        # The budget limit value (5.0) must appear in the message
        assert "5.0" in debug_info, (
            f"Expected max_budget (5.0) in debug message, got: {debug_info!r}"
        )
        # The time period string must NOT appear as the comparison value
        assert ">= 7d" not in debug_info, (
            f"budget_duration '7d' must not appear as the comparison operand: {debug_info!r}"
        )

    def test_exceeded_message_format(self):
        """Full message format check: 'current_spend >= max_budget'."""
        dep = {
            "model_name": "my-model",
            "litellm_params": {"model": "anthropic/claude-3"},
            "model_info": {"id": "dep-99"},
        }
        limiter = _make_limiter(
            deployment_budget_config={
                "dep-99": BudgetConfig(time_period="1d", budget_limit=2.5)
            }
        )
        deployment_configs = {
            "dep-99": BudgetConfig(time_period="1d", budget_limit=2.5)
        }
        spend_map = {"deployment_spend:dep-99:1d": 3.0}

        _, debug_info = limiter._filter_out_deployments_above_budget(
            potential_deployments=[],
            healthy_deployments=[dep],
            provider_configs={},
            deployment_configs=deployment_configs,
            deployment_providers=[],
            spend_map=spend_map,
            request_tags=[],
        )

        # Should contain "3.0 >= 2.5" not "3.0 >= 1d"
        assert "3.0" in debug_info
        assert "2.5" in debug_info
        assert "1d" not in debug_info.split(">=")[-1].strip(), (
            f"Expected a numeric max_budget after '>=', not a duration: {debug_info!r}"
        )
