import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

import pytest

from litellm.proxy.spend_tracking.autorouter_savings import compute_autorouter_savings


def _metadata(escalated: bool = False, input_cost: float = 1e-5, output_cost: float = 3e-5) -> dict:
    return {
        "autorouter_savings": {
            "autorouter_name": "smart-router",
            "baseline_model": "claude-opus-4-1",
            "baseline_input_cost_per_token": input_cost,
            "baseline_output_cost_per_token": output_cost,
            "escalated": escalated,
        }
    }


def test_savings_is_baseline_cost_minus_actual_spend():
    result = compute_autorouter_savings(
        _metadata(input_cost=1e-5, output_cost=3e-5),
        prompt_tokens=1000,
        completion_tokens=500,
        actual_spend=0.002,
    )
    baseline_cost = 1000 * 1e-5 + 500 * 3e-5
    assert result.savings_spend == pytest.approx(baseline_cost - 0.002)
    assert result.savings_spend > 0
    assert result.requests == 1
    assert result.escalated_requests == 0


def test_escalated_flag_counts_toward_escalated_requests():
    result = compute_autorouter_savings(
        _metadata(escalated=True),
        prompt_tokens=100,
        completion_tokens=100,
        actual_spend=0.0,
    )
    assert result.requests == 1
    assert result.escalated_requests == 1


def test_no_autorouter_metadata_returns_zero_and_is_not_counted():
    result = compute_autorouter_savings({}, prompt_tokens=1000, completion_tokens=500, actual_spend=0.002)
    assert result == (0.0, 0, 0)


def test_savings_clamped_to_zero_when_routed_model_pricier_than_baseline():
    result = compute_autorouter_savings(
        _metadata(input_cost=1e-6, output_cost=1e-6),
        prompt_tokens=10,
        completion_tokens=10,
        actual_spend=5.0,
    )
    assert result.savings_spend == 0.0
    assert result.requests == 1


def test_malformed_metadata_ignored():
    bad = {"autorouter_savings": {"autorouter_name": "smart-router", "escalated": "yes"}}
    result = compute_autorouter_savings(bad, prompt_tokens=1000, completion_tokens=500, actual_spend=0.0)
    assert result == (0.0, 0, 0)


def test_negative_token_counts_clamp_to_zero():
    result = compute_autorouter_savings(
        _metadata(),
        prompt_tokens=-1000,
        completion_tokens=-500,
        actual_spend=0.0,
    )
    assert result.savings_spend == 0.0
    assert result.requests == 1
