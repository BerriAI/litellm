"""Tests for litellm/router_strategy/simple_shuffle.py"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))  # Adds the parent directory to the system path

from litellm.router_strategy.simple_shuffle import simple_shuffle


@pytest.mark.parametrize("weight_by", ["weight", "rpm", "tpm"])
def test_weighted_pick_used_when_only_a_later_deployment_sets_metric(weight_by: str):
    """
    Regression test for https://github.com/BerriAI/litellm/issues/33329

    Weighted routing must be enabled when any healthy deployment sets the
    metric, regardless of list order. Previously only the first deployment
    was inspected, so a metric set only on a later deployment silently
    degraded routing to a uniform random pick.
    """
    router = SimpleNamespace(print_deployment=lambda deployment: str(deployment))
    unweighted = {"model_name": "chat", "litellm_params": {"model": "openai/first"}}
    weighted = {"model_name": "chat", "litellm_params": {"model": "openai/second", weight_by: 1}}

    for deployments in ([unweighted, weighted], [weighted, unweighted]):
        weighted_index = deployments.index(weighted)
        with (
            patch("litellm.router_strategy.simple_shuffle.random.choice") as uniform_pick,
            patch(
                "litellm.router_strategy.simple_shuffle.random.choices",
                return_value=[weighted_index],
            ) as weighted_pick,
        ):
            selected = simple_shuffle(router, deployments, "chat")

        uniform_pick.assert_not_called()
        weighted_pick.assert_called_once()
        expected_weights = [0.0, 1.0] if weighted_index == 1 else [1.0, 0.0]
        assert weighted_pick.call_args.kwargs["weights"] == expected_weights
        assert selected["litellm_params"]["model"] == "openai/second"
