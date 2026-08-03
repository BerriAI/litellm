from collections import Counter
from unittest.mock import MagicMock

from litellm.router_strategy.simple_shuffle import simple_shuffle


def _deployment(deployment_id: str, **litellm_params) -> dict:
    return {
        "model_name": "test-model",
        "litellm_params": {"model": "gpt-4o", **litellm_params},
        "model_info": {"id": deployment_id},
    }


def test_weights_respected_when_first_deployment_has_no_weight():
    """Regression for #33329: `simple_shuffle` decided whether weighted routing
    was enabled by inspecting only the first healthy deployment. When the first
    deployment had no `weight`, every deployment's weight was ignored and the
    pick fell back to uniform random. The weighted pick must be driven by any
    deployment declaring a weight, regardless of ordering.
    """
    healthy_deployments = [
        _deployment("A"),
        _deployment("B", weight=1),
        _deployment("C", weight=99),
    ]

    counts: Counter = Counter()
    for _ in range(2000):
        deployment = simple_shuffle(MagicMock(), healthy_deployments, "test-model")
        counts[deployment["model_info"]["id"]] += 1

    assert counts["A"] == 0
    assert counts["C"] > counts["B"] * 2
