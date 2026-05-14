"""
Unit tests for the simple-shuffle routing strategy weighting logic.
"""

from collections import Counter
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from litellm.router_strategy.simple_shuffle import _pick_weight_field, simple_shuffle


def _deployment(name: str, **litellm_params: Any) -> Dict[str, Any]:
    return {
        "model_name": name,
        "litellm_params": {"model": f"openai/{name}", **litellm_params},
    }


class TestPickWeightField:
    def test_returns_none_when_no_deployment_declares_anything(self):
        deployments = [_deployment("a"), _deployment("b")]
        assert _pick_weight_field(deployments) is None

    def test_returns_weight_when_declared_on_first(self):
        deployments = [_deployment("a", weight=10), _deployment("b")]
        assert _pick_weight_field(deployments) == "weight"

    def test_returns_weight_when_declared_only_on_later_deployments(self):
        """
        Regression for the silent-fallthrough bug: previously the code
        consulted only ``healthy_deployments[0]`` to decide whether to do
        a weighted pick. A config like::

            - model_name: gpt-4o-mini
              litellm_params: { model: openai/gpt-4o-mini }            # no weight
            - model_name: gpt-4o-mini
              litellm_params: { model: openai/gpt-4o-mini, rpm: 1000 } # weighted

        would silently fall back to uniform random because the *first*
        entry had no ``rpm``. Users perceived this as 'deployment-level
        ``rpm:`` is ignored under simple-shuffle' — actually it was just
        ordering-sensitive. Fix: scan all deployments.
        """
        deployments = [
            _deployment("a"),
            _deployment("b", rpm=1000),
            _deployment("c", rpm=500),
        ]
        assert _pick_weight_field(deployments) == "rpm"

    def test_precedence_weight_over_rpm_over_tpm(self):
        # ``weight`` declared anywhere should win over rpm/tpm.
        deployments = [_deployment("a", tpm=1000), _deployment("b", weight=5)]
        assert _pick_weight_field(deployments) == "weight"

        # ``rpm`` should win over ``tpm`` if no ``weight`` declared.
        deployments = [_deployment("a", tpm=1000), _deployment("b", rpm=5)]
        assert _pick_weight_field(deployments) == "rpm"

    def test_handles_missing_litellm_params(self):
        deployments: List[Dict[str, Any]] = [{"model_name": "broken"}]
        assert _pick_weight_field(deployments) is None


def _run_shuffle(deployments: List[Dict[str, Any]], n: int = 5000) -> Counter:
    router = MagicMock()
    router.print_deployment = lambda d: d.get("model_name", "?")
    picks: Counter = Counter()
    for _ in range(n):
        picked = simple_shuffle(
            llm_router_instance=router,
            healthy_deployments=deployments,
            model="model",
        )
        picks[picked["model_name"]] += 1
    return picks


class TestSimpleShuffle:
    def test_uniform_random_when_no_weights(self):
        deployments = [_deployment("a"), _deployment("b"), _deployment("c")]
        picks = _run_shuffle(deployments, n=6000)
        # Each deployment should be picked ~33% of the time. Generous
        # tolerance to keep the test non-flaky.
        for name in ("a", "b", "c"):
            assert (
                1500 < picks[name] < 2500
            ), f"Expected ~2000 picks for {name}, got {picks[name]}"

    def test_rpm_on_later_deployment_is_respected(self):
        """
        End-to-end regression for the silent-fallthrough bug. With the old
        code (which only checked deployments[0]) this distribution would
        be uniform across a/b/c. With the fix the weights become
        [0, 1000, 500] -> b gets ~2/3, c gets ~1/3, a gets 0.
        """
        deployments = [
            _deployment("a"),
            _deployment("b", rpm=1000),
            _deployment("c", rpm=500),
        ]
        picks = _run_shuffle(deployments, n=6000)
        assert picks["a"] == 0, "Deployment with no rpm should get weight 0"
        # b should be ~2/3 of picks, c ~1/3.
        assert 3500 < picks["b"] < 4500
        assert 1500 < picks["c"] < 2500

    def test_zero_total_weight_falls_through_to_uniform_random(self):
        """
        Defensive: if every deployment declares the weight field as 0
        (pathological config) we used to divide by zero. Now we fall
        through to uniform random instead of crashing.
        """
        deployments = [
            _deployment("a", weight=0),
            _deployment("b", weight=0),
            _deployment("c", weight=0),
        ]
        picks = _run_shuffle(deployments, n=3000)
        # No crash + every deployment is reachable.
        assert sum(picks.values()) == 3000
        for name in ("a", "b", "c"):
            assert picks[name] > 0

    def test_single_deployment_always_picked(self):
        deployments = [_deployment("only")]
        picks = _run_shuffle(deployments, n=100)
        assert picks["only"] == 100

    def test_weight_field_wins_over_rpm(self):
        """
        Precedence regression: ``weight`` should always win over ``rpm``
        when both are declared somewhere, so an operator using ``weight``
        for explicit routing isn't accidentally overridden by ``rpm``
        that's set for cap-enforcement purposes elsewhere.
        """
        deployments = [
            _deployment("a", weight=9, rpm=1),
            _deployment("b", weight=1, rpm=9),
        ]
        picks = _run_shuffle(deployments, n=6000)
        # ``weight``-driven: a should dominate.
        assert (
            picks["a"] > picks["b"] * 4
        ), f"weight should win over rpm, got picks={dict(picks)}"
