# SPDX-License-Identifier: MIT
# Tests for fix: simple_shuffle ignored weights when the first healthy
# deployment had none, falling through to uniform random.choice (fixes #33329).

from types import SimpleNamespace
from unittest.mock import patch

from litellm.router_strategy.simple_shuffle import simple_shuffle

_router = SimpleNamespace(print_deployment=lambda d: d)


def _deployments(first_has_weight: bool):
    first = {"model_name": "chat", "litellm_params": {"model": "openai/first"}}
    second = {"model_name": "chat", "litellm_params": {"model": "openai/second", "weight": 1}}
    if first_has_weight:
        first["litellm_params"]["weight"] = 0
    return [first, second]


def test_weight_on_second_deployment_triggers_weighted_pick():
    """Weight present only on second deployment must use random.choices, not random.choice."""
    deployments = _deployments(first_has_weight=False)
    with (
        patch("litellm.router_strategy.simple_shuffle.random.choice") as uniform_pick,
        patch(
            "litellm.router_strategy.simple_shuffle.random.choices",
            return_value=[1],
        ) as weighted_pick,
    ):
        simple_shuffle(_router, deployments, "chat")
        assert weighted_pick.call_count == 1, "weighted pick not used"
        assert uniform_pick.call_count == 0, "uniform pick used unexpectedly"


def test_reversed_deployment_order_same_result():
    """Reversing the list must not change whether weighted routing is used."""
    deployments = list(reversed(_deployments(first_has_weight=False)))
    with (
        patch("litellm.router_strategy.simple_shuffle.random.choice") as uniform_pick,
        patch(
            "litellm.router_strategy.simple_shuffle.random.choices",
            return_value=[0],
        ) as weighted_pick,
    ):
        simple_shuffle(_router, deployments, "chat")
        assert weighted_pick.call_count == 1, "weighted pick not used after reversing"
        assert uniform_pick.call_count == 0, "uniform pick used after reversing"


def test_no_weights_falls_through_to_uniform():
    """When no deployment has any weight, random.choice must be used."""
    deployments = [
        {"model_name": "chat", "litellm_params": {"model": "openai/a"}},
        {"model_name": "chat", "litellm_params": {"model": "openai/b"}},
    ]
    with (
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            return_value=deployments[0],
        ) as uniform_pick,
        patch("litellm.router_strategy.simple_shuffle.random.choices") as weighted_pick,
    ):
        simple_shuffle(_router, deployments, "chat")
        assert uniform_pick.call_count == 1
        assert weighted_pick.call_count == 0


def test_all_weights_present_uses_weighted_pick():
    """Sanity check: original behaviour with all weights present is preserved."""
    deployments = [
        {"model_name": "chat", "litellm_params": {"model": "openai/a", "weight": 2}},
        {"model_name": "chat", "litellm_params": {"model": "openai/b", "weight": 1}},
    ]
    with (
        patch("litellm.router_strategy.simple_shuffle.random.choice") as uniform_pick,
        patch(
            "litellm.router_strategy.simple_shuffle.random.choices",
            return_value=[0],
        ) as weighted_pick,
    ):
        simple_shuffle(_router, deployments, "chat")
        assert weighted_pick.call_count == 1
        assert uniform_pick.call_count == 0
