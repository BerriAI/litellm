from typing import Final
from unittest.mock import patch

import pytest

import litellm
from litellm.cost_calculator import _apply_cost_margin
from litellm.litellm_core_utils.llm_cost_calc.finalization import finalize_cost
from litellm.proxy.spend_tracking.team_cost_margin import apply_team_cost_margin


@pytest.mark.parametrize(("percent", "expected"), [(2, 1.02), (0, 1.0), (1000, 11.0)])
def test_team_margin_uses_percentage_points(percent: float, expected: float) -> None:
    cost: Final = apply_team_cost_margin(finalize_cost(1.0, None), percent)
    assert cost.amount == pytest.approx(expected)
    assert cost.logging_breakdown() == {
        "team_cost_before_margin": 1.0,
        "team_cost_margin_percent": percent,
        "total_cost": pytest.approx(expected),
    }


@pytest.mark.parametrize("percent", [None, "", -1, 1001, float("nan"), float("inf"), True])
def test_no_valid_team_margin_leaves_the_cost_unchanged(percent: object) -> None:
    cost: Final = finalize_cost(3.0, None)
    assert apply_team_cost_margin(cost, percent) is cost


@pytest.mark.parametrize("provider", ["anthropic", "other"])
def test_team_margin_composes_after_provider_or_global_margin(provider: str) -> None:
    with patch.dict(litellm.cost_margin_config, {"anthropic": 0.1, "global": 0.2}, clear=True):
        provider_cost, _, _, _ = _apply_cost_margin(5.0, provider)
    cost: Final = apply_team_cost_margin(finalize_cost(provider_cost, None), 2)
    assert cost.amount == pytest.approx(5.0 * (1.1 if provider == "anthropic" else 1.2) * 1.02)
    breakdown: Final = cost.logging_breakdown()
    assert breakdown is not None
    assert breakdown["team_cost_before_margin"] == provider_cost
    assert apply_team_cost_margin(cost, 2) is cost


def test_a_free_request_stays_free_with_a_team_margin() -> None:
    cost: Final = apply_team_cost_margin(finalize_cost(0.0, None), 2)
    assert cost.amount == 0.0
