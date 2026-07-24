"""
Tests for simple_shuffle router strategy.

Covers: https://github.com/BerriAI/litellm/issues/33329
Bug: simple_shuffle only checked healthy_deployments[0] to decide whether
weighted routing is enabled. If the first deployment omitted the metric but
a later deployment set it, weighted routing was silently skipped.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from litellm.router_strategy.simple_shuffle import simple_shuffle

# A minimal router stub — simple_shuffle only calls print_deployment on it.
_ROUTER_STUB = SimpleNamespace(print_deployment=lambda d: "deployment")


def _make_deployments(first_has_weight: bool):
    """Return two deployments; weight is on first or second depending on flag."""
    if first_has_weight:
        return [
            {
                "model_name": "chat",
                "litellm_params": {"model": "openai/first", "weight": 1},
            },
            {
                "model_name": "chat",
                "litellm_params": {"model": "openai/second"},
            },
        ]
    else:
        return [
            {
                "model_name": "chat",
                "litellm_params": {"model": "openai/first"},
            },
            {
                "model_name": "chat",
                "litellm_params": {"model": "openai/second", "weight": 1},
            },
        ]


class TestSimpleShuffleWeightOnFirstDeployment:
    """Baseline: weight on first deployment — weighted routing must be used."""

    def test_uses_weighted_pick_when_first_deployment_has_weight(self):
        deployments = _make_deployments(first_has_weight=True)

        with (
            patch(
                "litellm.router_strategy.simple_shuffle.random.choice",
            ) as uniform_pick,
            patch(
                "litellm.router_strategy.simple_shuffle.random.choices",
                return_value=[0],
            ) as weighted_pick,
        ):
            simple_shuffle(_ROUTER_STUB, deployments, "chat")

            assert weighted_pick.call_count == 1, "Should use weighted pick"
            assert uniform_pick.call_count == 0, "Should NOT use uniform pick"


class TestSimpleShuffleWeightOnSecondDeployment:
    """
    Regression for #33329:
    weight is only on the second deployment — weighted routing must still be used.
    Before the fix, simple_shuffle checked only index 0 and fell through to
    random.choice (uniform) because index 0 had no weight.
    """

    def test_uses_weighted_pick_when_only_second_deployment_has_weight(self):
        deployments = _make_deployments(first_has_weight=False)

        with (
            patch(
                "litellm.router_strategy.simple_shuffle.random.choice",
            ) as uniform_pick,
            patch(
                "litellm.router_strategy.simple_shuffle.random.choices",
                return_value=[1],
            ) as weighted_pick,
        ):
            simple_shuffle(_ROUTER_STUB, deployments, "chat")

            assert weighted_pick.call_count == 1, (
                "Should use weighted pick — weight exists on second deployment"
            )
            assert uniform_pick.call_count == 0, (
                "Should NOT use uniform pick when any deployment has weight"
            )

    def test_order_independence(self):
        """
        Reversing the deployment list must not change routing strategy.
        Both orderings should use weighted routing since weight is configured.
        """
        for first_has_weight in [True, False]:
            deployments = _make_deployments(first_has_weight=first_has_weight)

            with (
                patch(
                    "litellm.router_strategy.simple_shuffle.random.choice",
                ) as uniform_pick,
                patch(
                    "litellm.router_strategy.simple_shuffle.random.choices",
                    return_value=[0],
                ) as weighted_pick,
            ):
                simple_shuffle(_ROUTER_STUB, deployments, "chat")

                assert weighted_pick.call_count == 1, (
                    f"Expected weighted pick when first_has_weight={first_has_weight}"
                )
                assert uniform_pick.call_count == 0, (
                    f"Expected no uniform pick when first_has_weight={first_has_weight}"
                )


class TestSimpleShuffleNoWeights:
    """When no deployment has any weight/rpm/tpm, uniform random pick is used."""

    def test_uses_uniform_pick_when_no_weights(self):
        deployments = [
            {"model_name": "chat", "litellm_params": {"model": "openai/first"}},
            {"model_name": "chat", "litellm_params": {"model": "openai/second"}},
        ]

        with (
            patch(
                "litellm.router_strategy.simple_shuffle.random.choice",
                return_value=deployments[0],
            ) as uniform_pick,
            patch(
                "litellm.router_strategy.simple_shuffle.random.choices",
            ) as weighted_pick,
        ):
            simple_shuffle(_ROUTER_STUB, deployments, "chat")

            assert uniform_pick.call_count == 1, "Should use uniform pick when no weights"
            assert weighted_pick.call_count == 0, "Should NOT use weighted pick"


class TestSimpleShuffleRpmTpm:
    """Same order-independence fix applies to rpm and tpm metrics."""

    @pytest.mark.parametrize("metric", ["rpm", "tpm"])
    def test_uses_weighted_pick_when_only_second_deployment_has_metric(self, metric):
        deployments = [
            {"model_name": "chat", "litellm_params": {"model": "openai/first"}},
            {"model_name": "chat", "litellm_params": {"model": "openai/second", metric: 100}},
        ]

        with (
            patch(
                "litellm.router_strategy.simple_shuffle.random.choice",
            ) as uniform_pick,
            patch(
                "litellm.router_strategy.simple_shuffle.random.choices",
                return_value=[1],
            ) as weighted_pick,
        ):
            simple_shuffle(_ROUTER_STUB, deployments, "chat")

            assert weighted_pick.call_count == 1, (
                f"Should use weighted pick — {metric} exists on second deployment"
            )
            assert uniform_pick.call_count == 0
