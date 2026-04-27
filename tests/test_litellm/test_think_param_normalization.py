"""
Tests for the Ollama-style ``think`` -> OpenAI ``reasoning_effort`` normalization
performed inside ``litellm.utils.get_optional_params``.

Regression tests for https://github.com/BerriAI/litellm/issues/26413.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from litellm.utils import _think_to_reasoning_effort, get_optional_params


class TestThinkToReasoningEffortHelper:
    def test_think_false_bool_sets_reasoning_effort_disable(self):
        non_default_params: dict = {}
        supported_params = ["reasoning_effort"]

        _think_to_reasoning_effort(
            think_value=False,
            non_default_params=non_default_params,
            supported_params=supported_params,
            custom_llm_provider="gemini",
        )

        assert non_default_params["reasoning_effort"] == "disable"

    @pytest.mark.parametrize("falsy", ["false", "FALSE", "0", "no", "off", " False "])
    def test_think_false_string_variants(self, falsy):
        non_default_params: dict = {}

        _think_to_reasoning_effort(
            think_value=falsy,
            non_default_params=non_default_params,
            supported_params=["reasoning_effort"],
            custom_llm_provider="gemini",
        )

        assert non_default_params["reasoning_effort"] == "disable"

    def test_think_true_does_not_set_reasoning_effort(self):
        non_default_params: dict = {}

        _think_to_reasoning_effort(
            think_value=True,
            non_default_params=non_default_params,
            supported_params=["reasoning_effort"],
            custom_llm_provider="gemini",
        )

        assert "reasoning_effort" not in non_default_params

    def test_think_false_when_provider_does_not_support_reasoning(self):
        """``think`` should be silently dropped if the provider does not list
        ``reasoning_effort`` in its supported params, instead of raising."""
        non_default_params: dict = {}

        _think_to_reasoning_effort(
            think_value=False,
            non_default_params=non_default_params,
            supported_params=["temperature", "top_p"],
            custom_llm_provider="cohere_chat",
        )

        assert "reasoning_effort" not in non_default_params

    def test_think_false_does_not_override_explicit_reasoning_effort(self):
        non_default_params = {"reasoning_effort": "high"}

        _think_to_reasoning_effort(
            think_value=False,
            non_default_params=non_default_params,
            supported_params=["reasoning_effort"],
            custom_llm_provider="gemini",
        )

        assert non_default_params["reasoning_effort"] == "high"

    def test_think_false_respects_falsy_explicit_reasoning_effort(self):
        """An explicit ``reasoning_effort=""`` is a present (if empty) value and
        must NOT be overwritten by ``think=False``. Guards against the falsy
        membership-vs-presence trap."""
        non_default_params: dict = {"reasoning_effort": ""}

        _think_to_reasoning_effort(
            think_value=False,
            non_default_params=non_default_params,
            supported_params=["reasoning_effort"],
            custom_llm_provider="gemini",
        )

        assert non_default_params["reasoning_effort"] == ""

    @pytest.mark.parametrize(
        "provider",
        ["anthropic", "bedrock", "openai", "azure", "vertex_ai_anthropic"],
    )
    def test_think_false_dropped_for_providers_that_reject_disable(self, provider):
        """Providers whose `_map_reasoning_effort` does not accept the literal
        ``"disable"`` (Anthropic, Bedrock, OpenAI o-series, ...) would raise a
        runtime ``ValueError`` / ``BadRequestError`` if we injected
        ``reasoning_effort="disable"``. For these providers, ``think:false``
        must be silently dropped instead. Regression test for the P1 review
        comment on PR #26642."""
        non_default_params: dict = {}

        _think_to_reasoning_effort(
            think_value=False,
            non_default_params=non_default_params,
            supported_params=["reasoning_effort"],
            custom_llm_provider=provider,
        )

        assert "reasoning_effort" not in non_default_params

    def test_think_unrecognized_string_is_ignored(self):
        non_default_params: dict = {}

        _think_to_reasoning_effort(
            think_value="maybe",
            non_default_params=non_default_params,
            supported_params=["reasoning_effort"],
            custom_llm_provider="gemini",
        )

        assert "reasoning_effort" not in non_default_params


class TestGetOptionalParamsThinkFalse:
    """End-to-end tests through ``get_optional_params`` against real provider
    configs -- no network calls."""

    def test_gemini_think_false_disables_reasoning(self):
        """For a reasoning-capable Gemini model, ``think=False`` in the request
        body should cause reasoning to be disabled upstream (mirrors the
        scenario in issue #26413)."""
        optional_params = get_optional_params(
            model="gemini-2.5-flash",
            custom_llm_provider="gemini",
            think=False,
        )

        # Gemini disables thinking by setting thinking_config.thinking_budget=0
        thinking_config = optional_params.get("thinkingConfig") or optional_params.get(
            "thinking_config"
        )
        assert thinking_config is not None, (
            f"expected gemini thinkingConfig in optional_params, got: "
            f"{optional_params}"
        )
        assert (
            thinking_config.get("thinkingBudget") == 0
            or thinking_config.get("thinking_budget") == 0
        )

    def test_ollama_chat_think_false_propagates(self):
        """For Ollama (which natively understands ``think``), the resulting
        optional params should signal reasoning-off via the existing
        ``reasoning_effort=disable`` -> ``think=False`` mapping."""
        optional_params = get_optional_params(
            model="llama3.1",
            custom_llm_provider="ollama_chat",
            think=False,
        )

        # ollama_chat maps reasoning_effort != {low,medium,high} -> think=False
        assert optional_params.get("think") is False

    def test_think_false_dropped_silently_for_non_reasoning_provider(self):
        """A provider that doesn't expose ``reasoning_effort`` should not raise
        when ``think:false`` is supplied -- ``think`` is silently dropped."""
        # openai chat completion params do include reasoning_effort, so use a
        # provider whose supported params do NOT include it. ``cohere`` chat
        # does not currently advertise ``reasoning_effort``.
        optional_params = get_optional_params(
            model="command-r",
            custom_llm_provider="cohere_chat",
            think=False,
        )

        assert "think" not in optional_params
        assert "reasoning_effort" not in optional_params
