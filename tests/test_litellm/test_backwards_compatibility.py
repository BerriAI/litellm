"""Backwards compatibility verification: Rust paths must not break Python paths.

These tests verify that:
1. Without LITELLM_RUST_* env vars, Python paths work exactly as before
2. With LITELLM_RUST_* env vars, Rust paths produce identical results
3. When Rust fails/unavailable, Python fallback works transparently
"""

from __future__ import annotations

import os

import pytest


class TestBackwardsCompatibilityWithoutRust:
    """Verify Python paths work identically when Rust is disabled."""

    @pytest.fixture(autouse=True)
    def disable_rust(self, monkeypatch):
        monkeypatch.delenv("LITELLM_RUST_TOKEN_COUNTER", raising=False)
        monkeypatch.delenv("LITELLM_RUST_COST_CALCULATOR", raising=False)
        monkeypatch.delenv("LITELLM_RUST_AUTH", raising=False)
        monkeypatch.delenv("LITELLM_RUST_PIPELINE", raising=False)

    def test_token_counter_uses_python(self):
        from litellm.litellm_core_utils.token_counter import token_counter
        from litellm.rust_bridge.token_counter import try_rust_token_counter

        result = try_rust_token_counter(model="gpt-4", text="hello")
        assert result is None, "Rust should be disabled"

        py_result = token_counter(model="gpt-4", text="hello")
        assert py_result > 0, "Python path should work"

    def test_cost_calculator_uses_python(self):
        from litellm.cost_calculator import cost_per_token
        from litellm.rust_bridge.cost_calculator import try_rust_completion_cost

        result = try_rust_completion_cost(model="gpt-4", prompt_tokens=100, completion_tokens=50)
        assert result is None, "Rust should be disabled"

        py_prompt, py_completion = cost_per_token(model="gpt-4", prompt_tokens=100, completion_tokens=50)
        assert (py_prompt + py_completion) > 0, "Python path should work"

    def test_auth_hash_uses_python(self):
        from litellm.proxy.utils import hash_token
        from litellm.rust_bridge.auth import try_rust_hash_token

        result = try_rust_hash_token("sk-test")
        assert result is None, "Rust should be disabled"

        py_result = hash_token("sk-test")
        assert len(py_result) == 64, "Python path should work"

    def test_pipeline_uses_python(self):
        from litellm.rust_bridge.pipeline import process_request

        result = process_request("/v1/chat/completions", {"model": "gpt-4", "messages": []})
        assert result is None, "Rust should be disabled"


class TestRustPythonParity:
    """Verify Rust and Python produce identical results when both are available."""

    @pytest.fixture(autouse=True)
    def enable_rust(self, monkeypatch):
        monkeypatch.setenv("LITELLM_RUST_TOKEN_COUNTER", "1")
        monkeypatch.setenv("LITELLM_RUST_COST_CALCULATOR", "1")
        monkeypatch.setenv("LITELLM_RUST_AUTH", "1")

    def test_token_counter_parity(self):
        from litellm.litellm_core_utils.token_counter import token_counter as python_tc
        from litellm.rust_bridge.token_counter import try_rust_token_counter

        for model in ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]:
            for text in ["hello", "A" * 1000, "unicode: áéíóú"]:
                py = python_tc(model=model, text=text)
                rust = try_rust_token_counter(model=model, text=text)
                if rust is not None:
                    assert rust == py, f"Mismatch for model={model} text={text[:20]!r}"

    def test_cost_calculator_parity(self):
        from litellm.cost_calculator import cost_per_token
        from litellm.rust_bridge.cost_calculator import try_rust_completion_cost

        for model in ["gpt-4o", "gpt-4"]:
            py_prompt, py_completion = cost_per_token(model=model, prompt_tokens=100, completion_tokens=50)
            py_total = py_prompt + py_completion
            rust = try_rust_completion_cost(model=model, prompt_tokens=100, completion_tokens=50)
            if rust is not None:
                assert abs(rust - py_total) < 1e-12, f"Mismatch for model={model}"

    def test_auth_hash_parity(self):
        from litellm.proxy.utils import hash_token as python_ht
        from litellm.rust_bridge.auth import try_rust_hash_token

        for token in ["sk-test", "sk-" + "a" * 100, "sk-!@#$%^&*()"]:
            py = python_ht(token)
            rust = try_rust_hash_token(token)
            if rust is not None:
                assert rust == py, f"Mismatch for token={token[:20]!r}"


class TestFallbackBehavior:
    """Verify graceful fallback when Rust is unavailable or fails."""

    @pytest.fixture(autouse=True)
    def enable_rust(self, monkeypatch):
        monkeypatch.setenv("LITELLM_RUST_TOKEN_COUNTER", "1")
        monkeypatch.setenv("LITELLM_RUST_COST_CALCULATOR", "1")
        monkeypatch.setenv("LITELLM_RUST_AUTH", "1")

    def test_token_counter_fallback_on_unsupported_model(self):
        """Models not in the pricing DB should fall back to Python."""
        from litellm.litellm_core_utils.token_counter import token_counter

        result = token_counter(model="unknown-model-xyz", text="hello")
        assert result > 0, "Should fall back to Python and still work"

    def test_cost_calculator_fallback_on_unknown_model(self):
        """Models not in the pricing DB: Rust returns None (falls back to Python)."""
        from litellm.rust_bridge.cost_calculator import try_rust_completion_cost

        rust = try_rust_completion_cost(model="unknown-model-xyz", prompt_tokens=100, completion_tokens=50)
        assert rust is None, "Rust should return None for unknown models, triggering Python fallback"

    def test_auth_hash_always_works(self):
        """Auth hash should always work regardless of model."""
        from litellm.proxy.utils import hash_token

        result = hash_token("sk-any-token")
        assert len(result) == 64, "Should always produce a valid hash"
