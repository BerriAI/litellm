"""
Tests for reasoning effort capability fields and normalize_reasoning_effort_value.

Covers:
- Commit 1: get_model_info returns supports_minimal/supports_max fields
- Commit 2: Model registry entries have correct reasoning effort fields
- Commit 3: normalize_reasoning_effort_value degradation chains + adapter translation
"""

import json
import os
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest

import litellm

from litellm.llms.anthropic.experimental_pass_through.utils import (
    normalize_reasoning_effort_value,
)
from litellm.utils import get_model_info


def _load_model_registry() -> Dict[str, Any]:
    """Load the root model_prices_and_context_window.json."""
    json_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../../model_prices_and_context_window.json",
    )
    with open(json_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Commit 1: get_model_info returns supports_minimal and supports_max fields
# ---------------------------------------------------------------------------


class TestGetModelInfoReasoningEffortFields:
    """get_model_info should expose supports_minimal_reasoning_effort and
    supports_max_reasoning_effort from the model registry."""

    def test_opus_4_6_has_supports_minimal(self):
        info = get_model_info("claude-opus-4-6")
        assert "supports_minimal_reasoning_effort" in info

    def test_opus_4_6_has_supports_max(self):
        info = get_model_info("claude-opus-4-6")
        assert "supports_max_reasoning_effort" in info

    def test_opus_4_7_has_supports_minimal(self):
        info = get_model_info("claude-opus-4-7")
        assert "supports_minimal_reasoning_effort" in info

    def test_opus_4_7_has_supports_max(self):
        info = get_model_info("claude-opus-4-7")
        assert "supports_max_reasoning_effort" in info


# ---------------------------------------------------------------------------
# Commit 2: JSON registry has correct reasoning effort fields
# ---------------------------------------------------------------------------


class TestModelRegistryReasoningEffortFields:
    """Verify specific models have the expected reasoning effort capability
    values in the JSON registry file.

    Claude models intentionally OMIT ``supports_minimal_reasoning_effort``:
    ``minimal`` is not a real Anthropic effort level (the API accepts only
    low/medium/high/xhigh/max), so LiteLLM degrades ``minimal`` to ``low``
    regardless of the flag. These tests guard against the flag being
    re-added to the Claude fleet."""

    @pytest.fixture(autouse=True)
    def _load_registry(self):
        self.registry = _load_model_registry()

    def test_opus_4_7_supports_max(self):
        entry = self.registry["claude-opus-4-7"]
        assert entry.get("supports_max_reasoning_effort") is True

    def test_opus_4_6_supports_max(self):
        entry = self.registry["claude-opus-4-6"]
        assert entry.get("supports_max_reasoning_effort") is True

    def test_opus_4_7_omits_minimal(self):
        entry = self.registry["claude-opus-4-7"]
        assert "supports_minimal_reasoning_effort" not in entry

    def test_opus_4_6_omits_minimal(self):
        entry = self.registry["claude-opus-4-6"]
        assert "supports_minimal_reasoning_effort" not in entry

    def test_sonnet_4_6_omits_minimal(self):
        entry = self.registry["anthropic.claude-sonnet-4-6"]
        assert "supports_minimal_reasoning_effort" not in entry

    def test_bedrock_opus_4_7_supports_max(self):
        entry = self.registry["anthropic.claude-opus-4-7"]
        assert entry.get("supports_max_reasoning_effort") is True
        assert "supports_minimal_reasoning_effort" not in entry

    def test_vertex_opus_4_7_supports_max(self):
        entry = self.registry["vertex_ai/claude-opus-4-7"]
        assert entry.get("supports_max_reasoning_effort") is True
        assert "supports_minimal_reasoning_effort" not in entry

    def test_vertex_opus_4_6_supports_max(self):
        entry = self.registry["vertex_ai/claude-opus-4-6"]
        assert entry.get("supports_max_reasoning_effort") is True
        assert "supports_minimal_reasoning_effort" not in entry

    def test_azure_ai_opus_4_6_omits_minimal(self):
        entry = self.registry["azure_ai/claude-opus-4-6"]
        assert "supports_minimal_reasoning_effort" not in entry

    def test_azure_ai_opus_4_7_supports_max(self):
        entry = self.registry["azure_ai/claude-opus-4-7"]
        assert entry.get("supports_max_reasoning_effort") is True
        assert "supports_minimal_reasoning_effort" not in entry


# ---------------------------------------------------------------------------
# Commit 3: normalize_reasoning_effort_value
# ---------------------------------------------------------------------------


def _mock_model_info(**flags):
    """Return a mock model_info dict with given capability flags."""
    return flags


class TestNormalizeReasoningEffortValue:
    """Test degradation chains for normalize_reasoning_effort_value."""

    # --- "max" degradation chain ---

    def test_max_stays_max_when_supported(self):
        with patch(
            "litellm.utils.get_model_info",
            return_value=_mock_model_info(
                supports_max_reasoning_effort=True,
                supports_xhigh_reasoning_effort=True,
            ),
        ):
            assert normalize_reasoning_effort_value("max", model="test") == "max"

    def test_max_degrades_to_xhigh(self):
        with patch(
            "litellm.utils.get_model_info",
            return_value=_mock_model_info(
                supports_max_reasoning_effort=False,
                supports_xhigh_reasoning_effort=True,
            ),
        ):
            assert normalize_reasoning_effort_value("max", model="test") == "xhigh"

    def test_max_degrades_to_high(self):
        with patch(
            "litellm.utils.get_model_info",
            return_value=_mock_model_info(
                supports_max_reasoning_effort=False,
                supports_xhigh_reasoning_effort=False,
            ),
        ):
            assert normalize_reasoning_effort_value("max", model="test") == "high"

    # --- "xhigh" degradation chain ---

    def test_xhigh_stays_xhigh_when_supported(self):
        with patch(
            "litellm.utils.get_model_info",
            return_value=_mock_model_info(supports_xhigh_reasoning_effort=True),
        ):
            assert normalize_reasoning_effort_value("xhigh", model="test") == "xhigh"

    def test_xhigh_degrades_to_high(self):
        with patch(
            "litellm.utils.get_model_info",
            return_value=_mock_model_info(supports_xhigh_reasoning_effort=False),
        ):
            assert normalize_reasoning_effort_value("xhigh", model="test") == "high"

    # --- "minimal" degradation chain ---

    def test_minimal_stays_minimal_when_supported(self):
        with patch(
            "litellm.utils.get_model_info",
            return_value=_mock_model_info(supports_minimal_reasoning_effort=True),
        ):
            assert (
                normalize_reasoning_effort_value("minimal", model="test") == "minimal"
            )

    def test_minimal_degrades_to_low(self):
        with patch(
            "litellm.utils.get_model_info",
            return_value=_mock_model_info(supports_minimal_reasoning_effort=False),
        ):
            assert normalize_reasoning_effort_value("minimal", model="test") == "low"

    # --- passthrough values ---

    def test_high_passes_through(self):
        assert normalize_reasoning_effort_value("high", model="test") == "high"

    def test_medium_passes_through(self):
        assert normalize_reasoning_effort_value("medium", model="test") == "medium"

    def test_low_passes_through(self):
        assert normalize_reasoning_effort_value("low", model="test") == "low"

    # --- exception fallback ---

    def test_exception_fallback_uses_empty_model_info(self):
        """When get_model_info raises, treat model_info as {} (no capabilities)."""
        with patch(
            "litellm.utils.get_model_info",
            side_effect=Exception("model not found"),
        ):
            # "max" with no capabilities -> "high"
            assert normalize_reasoning_effort_value("max", model="unknown") == "high"
            # "minimal" with no capabilities -> "low"
            assert normalize_reasoning_effort_value("minimal", model="unknown") == "low"


# ---------------------------------------------------------------------------
# Commit 3: Adapter translation — adaptive thinking + output_config.effort
# ---------------------------------------------------------------------------


class TestAdapterAdaptiveThinking:
    """Test that adaptive thinking type maps correctly through the adapters."""

    def test_messages_adapter_adaptive_returns_medium_default(self):
        """Adaptive thinking returns 'medium' as default reasoning_effort."""
        from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )

        adapter = LiteLLMAnthropicMessagesAdapter()
        result = adapter.translate_anthropic_thinking_to_reasoning_effort(
            {"type": "adaptive"}
        )
        assert result == "medium"

    def test_messages_adapter_adaptive_overridden_by_output_config(self):
        """For adaptive thinking, output_config.effort overrides reasoning_effort."""
        from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )
        from litellm.types.llms.anthropic import AnthropicMessagesRequest

        adapter = LiteLLMAnthropicMessagesAdapter()
        request = AnthropicMessagesRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1024,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
        )
        openai_kwargs, _ = adapter.translate_anthropic_to_openai(request)
        # reasoning_effort should be set (either as string or dict with effort)
        re = openai_kwargs.get("reasoning_effort")
        if isinstance(re, dict):
            assert re["effort"] == "high"
        else:
            assert re == "high"

    def test_responses_adapter_adaptive_with_output_config(self):
        """Responses adapter: adaptive thinking + output_config.effort."""
        from litellm.llms.anthropic.experimental_pass_through.responses_adapters.transformation import (
            LiteLLMAnthropicToResponsesAPIAdapter,
        )

        result = LiteLLMAnthropicToResponsesAPIAdapter.translate_thinking_to_reasoning(
            thinking={"type": "adaptive"},
            output_config={"effort": "xhigh"},
        )
        assert result is not None
        assert result["effort"] == "xhigh"

    def test_responses_adapter_adaptive_default_medium(self):
        """Responses adapter: adaptive thinking without output_config defaults to medium."""
        from litellm.llms.anthropic.experimental_pass_through.responses_adapters.transformation import (
            LiteLLMAnthropicToResponsesAPIAdapter,
        )

        result = LiteLLMAnthropicToResponsesAPIAdapter.translate_thinking_to_reasoning(
            thinking={"type": "adaptive"},
        )
        assert result is not None
        assert result["effort"] == "medium"


class TestDeclaredEffortsAnswerTheDegradationGate:
    """Without this the chain reads only the per-level booleans, so a kimi-k3 request asking for
    max silently arrives as high."""

    @pytest.mark.parametrize(
        "model, provider",
        [("kimi-k3", "moonshot"), ("kimi-k3", "fireworks_ai"), ("kimi-k3-us", "fireworks_ai")],
    )
    def test_a_declared_level_survives_instead_of_degrading(self, local_model_cost_map, model, provider):
        assert normalize_reasoning_effort_value("max", model, provider) == "max"

    def test_a_level_the_entry_does_not_declare_still_degrades(self, local_model_cost_map):
        """xhigh is not on kimi-k3's declaration, so it must keep degrading rather than be waved
        past by the mere presence of one."""
        assert normalize_reasoning_effort_value("xhigh", "kimi-k3", "moonshot") == "high"
        assert normalize_reasoning_effort_value("minimal", "kimi-k3", "moonshot") == "low"

    def test_the_wider_perplexity_entry_keeps_the_levels_it_declares(self, local_model_cost_map):
        assert normalize_reasoning_effort_value("xhigh", "perplexity/kimi-k3", "perplexity") == "xhigh"
        assert normalize_reasoning_effort_value("minimal", "perplexity/kimi-k3", "perplexity") == "minimal"

    @pytest.mark.parametrize(
        "model, provider, effort, expected",
        [
            ("claude-opus-4-7", "anthropic", "max", "max"),
            ("claude-sonnet-4-6", "anthropic", "minimal", "low"),
            ("gpt-5-mini", "azure", "max", "high"),
        ],
    )
    def test_an_entry_on_the_per_level_flags_is_untouched(
        self, local_model_cost_map, model, provider, effort, expected
    ):
        """The negative class that bounds this change to entries carrying the key."""
        assert normalize_reasoning_effort_value(effort, model, provider) == expected


class TestDeclarationBeatsThePerLevelFlags:
    """An entry can carry both shapes. The declaration wins whole, or /model_group/info and this
    path would disagree about the same deployment. Driven through the public entry point over a
    seeded map entry rather than a patched get_model_info, so it pins behaviour and not wiring."""

    MODEL = "declared-and-flagged"

    @pytest.fixture
    def seeded(self, local_model_cost_map, monkeypatch):
        def _seed(**entry):
            monkeypatch.setitem(
                litellm.model_cost,
                self.MODEL,
                {"litellm_provider": "openai", "mode": "chat", "supports_reasoning": True, **entry},
            )
            litellm.get_model_info.cache_clear()

        return _seed

    @pytest.mark.parametrize("effort, expected", [("max", "max"), ("xhigh", "high"), ("minimal", "low")])
    def test_a_flag_cannot_re_add_a_level_the_declaration_omits(self, seeded, effort, expected):
        seeded(
            reasoning_effort_levels=["low", "high", "max"],
            supports_xhigh_reasoning_effort=True,
            supports_minimal_reasoning_effort=True,
            supports_max_reasoning_effort=False,
        )

        assert normalize_reasoning_effort_value(effort, self.MODEL, "openai") == expected

    def test_a_flag_cannot_keep_max_when_the_declaration_drops_it(self, seeded):
        seeded(
            reasoning_effort_levels=["low", "high"],
            supports_max_reasoning_effort=True,
            supports_xhigh_reasoning_effort=True,
        )

        assert normalize_reasoning_effort_value("max", self.MODEL, "openai") == "high"

    def test_a_false_flag_cannot_remove_a_level_the_declaration_names(self, seeded):
        seeded(reasoning_effort_levels=["high", "xhigh"], supports_xhigh_reasoning_effort=False)

        assert normalize_reasoning_effort_value("xhigh", self.MODEL, "openai") == "xhigh"
        assert normalize_reasoning_effort_value("max", self.MODEL, "openai") == "xhigh"

    def test_a_chain_the_declaration_omits_entirely_lands_on_its_terminal(self, seeded):
        """Documented residual: no strength ordering exists to pick a nearer declared level."""
        seeded(reasoning_effort_levels=["high", "xhigh"])

        assert normalize_reasoning_effort_value("minimal", self.MODEL, "openai") == "low"
