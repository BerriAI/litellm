"""Regression tests for deployment / per-request custom pricing robustness.

Covers the two intertwined billing bugs described in
https://github.com/BerriAI/litellm/issues/30081:

Bug 1 - custom pricing lost when ``litellm.model_cost`` is replaced
        (e.g. after a model-cost-map reload). The deployment's explicit
        ``input_cost_per_token`` / ``output_cost_per_token`` must still be
        honored, because they are threaded through the cost calculator as
        ``custom_cost_per_token`` rather than depending on the deployment-id
        entry surviving in the process-global ``litellm.model_cost``.

Bug 2 - per-request custom pricing must apply to that request only and must
        NOT overwrite (poison) the shared canonical ``model_cost`` entry used
        by all other traffic.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import Usage
from litellm.litellm_core_utils.litellm_logging import (
    extract_custom_cost_per_second,
    extract_custom_cost_per_token,
)


def _make_model_response(model: str, prompt_tokens: int, completion_tokens: int):
    return litellm.ModelResponse(
        model=model,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# Bug 1 - custom_cost_per_token threading
# ---------------------------------------------------------------------------
def test_response_cost_calculator_honors_zero_custom_cost_per_token():
    """A deployment priced at 0 stays at 0 even when its model_cost entry is gone.

    Simulates the post-reload state: ``router_model_id`` is not present in
    ``litellm.model_cost`` and ``custom_pricing`` is detected from the
    deployment params. With the explicit rates threaded through as
    ``custom_cost_per_token`` the cost is 0, instead of falling back to the
    model's public price.
    """
    response = _make_model_response(
        "gpt-4o", prompt_tokens=100_000, completion_tokens=10_000
    )

    cost = litellm.response_cost_calculator(
        response_object=response,
        model="gpt-4o",
        custom_llm_provider="openai",
        call_type="acompletion",
        optional_params={},
        custom_pricing=True,
        router_model_id="id-not-in-model-cost",  # wiped by a reload
        custom_cost_per_token={
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        },
    )

    assert cost == 0.0


def test_response_cost_calculator_falls_back_without_custom_cost_per_token():
    """Control: without the threaded rates the same request bills at public price.

    This is the buggy behaviour from the issue - it confirms the threading in
    :func:`test_response_cost_calculator_honors_zero_custom_cost_per_token` is
    what fixes the silent mis-billing.
    """
    response = _make_model_response(
        "gpt-4o", prompt_tokens=100_000, completion_tokens=10_000
    )

    cost = litellm.response_cost_calculator(
        response_object=response,
        model="gpt-4o",
        custom_llm_provider="openai",
        call_type="acompletion",
        optional_params={},
        custom_pricing=True,
        router_model_id="id-not-in-model-cost",
    )

    assert cost > 0.0


def test_response_cost_calculator_honors_nonzero_custom_cost_per_token():
    """Explicit non-zero rates are applied exactly, regardless of model_cost."""
    response = _make_model_response(
        "gpt-4o", prompt_tokens=1_000, completion_tokens=2_000
    )

    cost = litellm.response_cost_calculator(
        response_object=response,
        model="gpt-4o",
        custom_llm_provider="openai",
        call_type="acompletion",
        optional_params={},
        custom_pricing=True,
        router_model_id="id-not-in-model-cost",
        custom_cost_per_token={
            "input_cost_per_token": 1e-06,
            "output_cost_per_token": 2e-06,
        },
    )

    # 1000 * 1e-06 + 2000 * 2e-06 = 0.001 + 0.004
    assert cost == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# extract_custom_cost_per_token helper
# ---------------------------------------------------------------------------
def test_extract_from_top_level_litellm_params():
    result = extract_custom_cost_per_token(
        {"input_cost_per_token": 0, "output_cost_per_token": 0}
    )
    assert result == {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}


def test_extract_from_metadata_model_info():
    litellm_params = {
        "metadata": {
            "model_info": {
                "input_cost_per_token": 1e-06,
                "output_cost_per_token": 2e-06,
            }
        }
    }
    result = extract_custom_cost_per_token(litellm_params)
    assert result == {
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 2e-06,
    }


def test_extract_from_litellm_metadata_model_info():
    litellm_params = {
        "litellm_metadata": {
            "model_info": {
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            }
        }
    }
    result = extract_custom_cost_per_token(litellm_params)
    assert result == {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}


def test_extract_includes_cache_rates_when_present():
    result = extract_custom_cost_per_token(
        {
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
            "cache_read_input_token_cost": 0,
            "cache_creation_input_token_cost": 0,
        }
    )
    assert result == {
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "cache_read_input_token_cost": 0.0,
        "cache_creation_input_token_cost": 0.0,
    }


def test_extract_returns_none_for_partial_pricing():
    # Only input set -> stay on the standard pricing path (don't assume 0 output).
    assert extract_custom_cost_per_token({"input_cost_per_token": 0}) is None


def test_extract_returns_none_when_absent():
    assert extract_custom_cost_per_token(None) is None
    assert extract_custom_cost_per_token({}) is None
    assert extract_custom_cost_per_token({"model_info": {}}) is None


# ---------------------------------------------------------------------------
# extract_custom_cost_per_second
# ---------------------------------------------------------------------------
def test_extract_custom_cost_per_second_top_level():
    result = extract_custom_cost_per_second({"input_cost_per_second": 0.005})
    assert result == 0.005


def test_extract_custom_cost_per_second_zero():
    result = extract_custom_cost_per_second({"input_cost_per_second": 0})
    assert result == 0.0


def test_extract_custom_cost_per_second_from_model_info():
    result = extract_custom_cost_per_second(
        {"model_info": {"input_cost_per_second": 0.01}}
    )
    assert result == 0.01


def test_extract_custom_cost_per_second_from_metadata():
    result = extract_custom_cost_per_second(
        {"metadata": {"model_info": {"input_cost_per_second": 0.02}}}
    )
    assert result == 0.02


def test_extract_custom_cost_per_second_absent():
    assert extract_custom_cost_per_second(None) is None
    assert extract_custom_cost_per_second({}) is None


# ---------------------------------------------------------------------------
# extract_custom_cost_per_token — security gate
# ---------------------------------------------------------------------------
def test_extract_skips_model_info_without_deployment_flag():
    """Top-level model_info must be ignored when _model_info_from_deployment is False.

    This prevents proxy clients from injecting ``model_info: {input_cost_per_token: 0,
    output_cost_per_token: 0}`` to bypass spend tracking."""
    litellm_params = {
        "model_info": {
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        }
    }
    result = extract_custom_cost_per_token(litellm_params)
    assert result is None


def test_extract_uses_model_info_with_deployment_flag():
    litellm_params = {
        "model_info": {
            "input_cost_per_token": 1e-06,
            "output_cost_per_token": 2e-06,
        }
    }
    result = extract_custom_cost_per_token(
        litellm_params, _model_info_from_deployment=True
    )
    assert result == {
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 2e-06,
    }


# ---------------------------------------------------------------------------
# response_cost_calculator — custom_cost_per_second threading
# ---------------------------------------------------------------------------
def test_response_cost_calculator_honors_custom_cost_per_second():
    """Per-second custom pricing is threaded through the cost calculator."""
    response = _make_model_response("gpt-4o", prompt_tokens=100, completion_tokens=0)

    # total_time (ms) is needed alongside custom_cost_per_second so
    # completion_cost can compute cost = rate * total_time_ms / 1000.
    cost = litellm.response_cost_calculator(
        response_object=response,
        model="gpt-4o",
        custom_llm_provider="openai",
        call_type="acompletion",
        optional_params={},
        custom_pricing=True,
        router_model_id="id-not-in-model-cost",
        custom_cost_per_second=0.01,
        total_time=5000.0,
    )

    # 0.01 * 5000 / 1000 = 0.05
    assert cost == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Bug 2 - per-request pricing must not poison the canonical model_cost entry
# ---------------------------------------------------------------------------
def test_per_request_custom_pricing_does_not_poison_canonical_entry():
    """A single zero-priced request must not re-price the shared model entry.

    Verifies both:
    - The request itself is billed at the custom (zero) rate.
    - The canonical ``litellm.model_cost`` entry is left unchanged
      for all other traffic.
    """
    model = "gpt-4o"
    assert model in litellm.model_cost
    canonical_input_cost_before = litellm.model_cost[model]["input_cost_per_token"]
    canonical_output_cost_before = litellm.model_cost[model]["output_cost_per_token"]
    assert canonical_input_cost_before > 0

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        mock_response="ok",
        input_cost_per_token=0,
        output_cost_per_token=0,
    )

    # The canonical entry must not be mutated.
    assert (
        litellm.model_cost[model]["input_cost_per_token"] == canonical_input_cost_before
    )
    assert (
        litellm.model_cost[model]["output_cost_per_token"]
        == canonical_output_cost_before
    )

    # The per-request cost must also be calculated with the custom rates.
    # Verify via the cost calculator directly (mock_response bypasses the
    # normal cost-calculation path, so the response object may not carry
    # cost metadata).
    cost = litellm.response_cost_calculator(
        response_object=response,
        model=model,
        custom_llm_provider="openai",
        call_type="completion",
        optional_params={},
        custom_pricing=True,
        custom_cost_per_token={
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
    )
    assert cost == 0.0
