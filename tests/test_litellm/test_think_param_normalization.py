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


class _FakeConfigSupportsDisable:
    """Stand-in for a provider config (e.g. ``GeminiConfig`` /
    ``OllamaConfig``) that opts in to ``reasoning_effort='disable'`` via the
    ``supports_reasoning_disable`` class flag."""

    supports_reasoning_disable = True


class _FakeConfigRejectsDisable:
    """Stand-in for a provider config (e.g. ``AnthropicConfig`` /
    ``BedrockConverseConfig``) whose ``_map_reasoning_effort`` would raise on
    the literal ``'disable'``; default value of the ``BaseConfig`` class flag
    is ``False``."""

    supports_reasoning_disable = False


class TestThinkToReasoningEffortHelper:
    def test_think_false_bool_sets_reasoning_effort_disable(self):
        non_default_params: dict = {}
        supported_params = ["reasoning_effort"]

        _think_to_reasoning_effort(
            think_value=False,
            non_default_params=non_default_params,
            supported_params=supported_params,
            provider_config=_FakeConfigSupportsDisable(),
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
            provider_config=_FakeConfigSupportsDisable(),
            custom_llm_provider="gemini",
        )

        assert non_default_params["reasoning_effort"] == "disable"

    def test_think_true_does_not_set_reasoning_effort(self):
        non_default_params: dict = {}

        _think_to_reasoning_effort(
            think_value=True,
            non_default_params=non_default_params,
            supported_params=["reasoning_effort"],
            provider_config=_FakeConfigSupportsDisable(),
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
            provider_config=_FakeConfigRejectsDisable(),
            custom_llm_provider="cohere_chat",
        )

        assert "reasoning_effort" not in non_default_params

    def test_think_false_does_not_override_explicit_reasoning_effort(self):
        non_default_params = {"reasoning_effort": "high"}

        _think_to_reasoning_effort(
            think_value=False,
            non_default_params=non_default_params,
            supported_params=["reasoning_effort"],
            provider_config=_FakeConfigSupportsDisable(),
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
            provider_config=_FakeConfigSupportsDisable(),
            custom_llm_provider="gemini",
        )

        assert non_default_params["reasoning_effort"] == ""

    def test_think_false_dropped_when_provider_config_rejects_disable(self):
        """Providers whose config has ``supports_reasoning_disable=False`` (the
        ``BaseConfig`` default — Anthropic, Bedrock, OpenAI o-series, ...)
        would raise on ``reasoning_effort='disable'``. ``think:false`` must be
        silently dropped instead. Regression for the P1 review on PR #26642."""
        non_default_params: dict = {}

        _think_to_reasoning_effort(
            think_value=False,
            non_default_params=non_default_params,
            supported_params=["reasoning_effort"],
            provider_config=_FakeConfigRejectsDisable(),
            custom_llm_provider="anthropic",
        )

        assert "reasoning_effort" not in non_default_params

    def test_think_false_dropped_when_no_provider_config_passed(self):
        """If we cannot determine the provider's capability (no config), be
        conservative and drop rather than risk a runtime ValueError."""
        non_default_params: dict = {}

        _think_to_reasoning_effort(
            think_value=False,
            non_default_params=non_default_params,
            supported_params=["reasoning_effort"],
            provider_config=None,
            custom_llm_provider="some_unknown_provider",
        )

        assert "reasoning_effort" not in non_default_params

    def test_think_unrecognized_string_is_ignored(self):
        non_default_params: dict = {}

        _think_to_reasoning_effort(
            think_value="maybe",
            non_default_params=non_default_params,
            supported_params=["reasoning_effort"],
            provider_config=_FakeConfigSupportsDisable(),
            custom_llm_provider="gemini",
        )

        assert "reasoning_effort" not in non_default_params


class TestProviderConfigsOptInToReasoningDisable:
    """Verifies the per-provider opt-in flags wired up on the actual config
    classes — these are what the real ``get_optional_params`` flow consults."""

    def test_gemini_configs_opt_in(self):
        from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig
        from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
            VertexGeminiConfig,
        )

        assert VertexGeminiConfig.supports_reasoning_disable is True
        assert GoogleAIStudioGeminiConfig.supports_reasoning_disable is True

    def test_ollama_configs_opt_in(self):
        from litellm.llms.ollama.chat.transformation import OllamaChatConfig
        from litellm.llms.ollama.completion.transformation import OllamaConfig

        assert OllamaConfig.supports_reasoning_disable is True
        assert OllamaChatConfig.supports_reasoning_disable is True

    def test_anthropic_config_does_not_opt_in(self):
        """Anthropic's ``_map_reasoning_effort`` raises on ``'disable'``, so it
        must keep the ``BaseConfig`` default of ``False``."""
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        assert AnthropicConfig.supports_reasoning_disable is False

    def test_bedrock_config_does_not_opt_in(self):
        """Bedrock Converse rejects any ``reasoning_effort`` value outside
        ``low|medium|high``, so it must not opt in."""
        from litellm.llms.bedrock.chat.converse_transformation import (
            AmazonConverseConfig,
        )

        assert AmazonConverseConfig.supports_reasoning_disable is False


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
