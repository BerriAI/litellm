"""Parity tests: Rust cost calculator must produce identical results to Python.

These tests verify that the Rust bridge implementation matches the Python
implementation for cost calculation across various models and token configurations.
"""

from __future__ import annotations

import os

import pytest

import litellm
from litellm.cost_calculator import cost_per_token as python_cost_per_token
from litellm.rust_bridge.cost_calculator import try_rust_completion_cost


@pytest.fixture(autouse=True)
def enable_rust_cost_calculator(monkeypatch):
    monkeypatch.setenv("LITELLM_RUST_COST_CALCULATOR", "1")


def _python_cost(model, prompt_tokens, completion_tokens, **kwargs):
    prompt_cost, completion_cost = python_cost_per_token(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        **kwargs,
    )
    return prompt_cost + completion_cost


def _rust_cost(model, prompt_tokens, completion_tokens, **kwargs):
    return try_rust_completion_cost(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        **kwargs,
    )


class TestRustMatchesPythonBasic:
    @pytest.mark.parametrize(
        "model",
        ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
    )
    def test_rust_matches_python_basic_tokens(self, model):
        python_cost = _python_cost(model=model, prompt_tokens=100, completion_tokens=50)
        rust_cost = _rust_cost(model=model, prompt_tokens=100, completion_tokens=50)

        if rust_cost is None:
            pytest.skip("Rust bridge unavailable or model not in pricing DB")

        assert abs(rust_cost - python_cost) < 1e-12, (
            f"Rust ({rust_cost}) != Python ({python_cost}) for model={model}"
        )

    @pytest.mark.parametrize(
        "python_model,rust_model",
        [
            ("anthropic/claude-3-opus-20240229", "claude-3-opus-20240229"),
        ],
    )
    def test_rust_matches_python_anthropic_models(self, python_model, rust_model):
        """Anthropic models: Python uses provider prefix, Rust strips it."""
        python_cost = _python_cost(model=python_model, prompt_tokens=100, completion_tokens=50)
        rust_cost = _rust_cost(model=rust_model, prompt_tokens=100, completion_tokens=50)

        if rust_cost is None:
            pytest.skip("Rust bridge unavailable")

        assert abs(rust_cost - python_cost) < 1e-12, (
            f"Rust ({rust_cost}) != Python ({python_cost}) for model={python_model}"
        )

    def test_rust_finds_anthropic_pricing_by_direct_name(self):
        """Rust can find Anthropic pricing by direct model name (no prefix needed)."""
        rust_cost = _rust_cost(model="claude-3-opus-20240229", prompt_tokens=100, completion_tokens=50)
        if rust_cost is None:
            pytest.skip("Rust bridge unavailable")
        assert rust_cost > 0, "Should find pricing for claude-3-opus-20240229"

    def test_zero_tokens_zero_cost(self):
        python_cost = _python_cost(model="gpt-4o", prompt_tokens=0, completion_tokens=0)
        rust_cost = _rust_cost(model="gpt-4o", prompt_tokens=0, completion_tokens=0)

        if rust_cost is None:
            pytest.skip("Rust bridge unavailable")

        assert rust_cost == 0.0
        assert python_cost == 0.0


class TestRustMatchesPythonCacheTokens:
    def test_rust_matches_python_with_cache_hit(self):
        model = "gpt-4o"

        python_cost = _python_cost(
            model=model, prompt_tokens=1000, completion_tokens=100,
            cache_read_input_tokens=800,
        )
        rust_cost = _rust_cost(
            model=model, prompt_tokens=1000, completion_tokens=100,
            cache_hit_tokens=800,
        )

        if rust_cost is None:
            pytest.skip("Rust bridge unavailable")

        assert abs(rust_cost - python_cost) < 1e-12, (
            f"Cache hit: Rust ({rust_cost}) != Python ({python_cost})"
        )

    def test_rust_matches_python_with_cache_creation(self):
        model = "anthropic/claude-3-opus-20240229"

        python_cost = _python_cost(
            model=model, prompt_tokens=1000, completion_tokens=100,
            cache_creation_input_tokens=500,
        )
        rust_cost = _rust_cost(
            model="claude-3-opus-20240229",
            prompt_tokens=1000, completion_tokens=100,
            cache_creation_tokens=500,
        )

        if rust_cost is None:
            pytest.skip("Rust bridge unavailable")

        assert abs(rust_cost - python_cost) < 1e-12, (
            f"Cache creation: Rust ({rust_cost}) != Python ({python_cost})"
        )


class TestRustMatchesPythonServiceTier:
    def test_rust_matches_python_priority_tier(self):
        model = "gpt-4o"

        python_cost = _python_cost(
            model=model, prompt_tokens=100, completion_tokens=50,
            service_tier="priority",
        )
        rust_cost = _rust_cost(
            model=model, prompt_tokens=100, completion_tokens=50,
            service_tier="priority",
        )

        if rust_cost is None:
            pytest.skip("Rust bridge unavailable")

        assert abs(rust_cost - python_cost) < 1e-12


class TestRustFallback:
    def test_fallback_on_unknown_model(self):
        result = _rust_cost(model="nonexistent-model-xyz", prompt_tokens=100, completion_tokens=50)
        assert result is None, "Rust should return None for unknown models"

    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("LITELLM_RUST_COST_CALCULATOR", raising=False)
        result = _rust_cost(model="gpt-4o", prompt_tokens=100, completion_tokens=50)
        assert result is None, "Rust should be disabled without env var"


class TestRustEdgeCases:
    def test_large_token_counts(self):
        model = "gpt-4o"

        python_cost = _python_cost(model=model, prompt_tokens=1_000_000, completion_tokens=100_000)
        rust_cost = _rust_cost(model=model, prompt_tokens=1_000_000, completion_tokens=100_000)

        if rust_cost is None:
            pytest.skip("Rust bridge unavailable")

        assert abs(rust_cost - python_cost) < 1e-6

    def test_threshold_pricing(self):
        model = "claude-3-opus-20240229"

        python_cost = _python_cost(model=model, prompt_tokens=250_000, completion_tokens=1000)
        rust_cost = _rust_cost(model=model, prompt_tokens=250_000, completion_tokens=1000)

        if rust_cost is None:
            pytest.skip("Rust bridge unavailable")

        assert abs(rust_cost - python_cost) < 1e-9
