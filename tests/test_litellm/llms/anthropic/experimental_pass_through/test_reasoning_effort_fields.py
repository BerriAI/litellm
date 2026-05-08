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
    values in the JSON registry file."""

    @pytest.fixture(autouse=True)
    def _load_registry(self):
        self.registry = _load_model_registry()

    def test_opus_4_7_supports_max(self):
        entry = self.registry["claude-opus-4-7"]
        assert entry.get("supports_max_reasoning_effort") is True

    def test_opus_4_6_supports_max(self):
        entry = self.registry["claude-opus-4-6"]
        assert entry.get("supports_max_reasoning_effort") is True

    def test_opus_4_7_supports_minimal(self):
        entry = self.registry["claude-opus-4-7"]
        assert entry.get("supports_minimal_reasoning_effort") is True

    def test_opus_4_6_supports_minimal(self):
        entry = self.registry["claude-opus-4-6"]
        assert entry.get("supports_minimal_reasoning_effort") is True

    def test_sonnet_4_6_supports_minimal(self):
        entry = self.registry["anthropic.claude-sonnet-4-6"]
        assert entry.get("supports_minimal_reasoning_effort") is True

    def test_bedrock_opus_4_7_supports_max(self):
        entry = self.registry["anthropic.claude-opus-4-7"]
        assert entry.get("supports_max_reasoning_effort") is True
        assert entry.get("supports_minimal_reasoning_effort") is True

    def test_vertex_opus_4_7_supports_max(self):
        entry = self.registry["vertex_ai/claude-opus-4-7"]
        assert entry.get("supports_max_reasoning_effort") is True
        assert entry.get("supports_minimal_reasoning_effort") is True

    def test_vertex_opus_4_6_supports_max(self):
        entry = self.registry["vertex_ai/claude-opus-4-6"]
        assert entry.get("supports_max_reasoning_effort") is True
        assert entry.get("supports_minimal_reasoning_effort") is True

    def test_azure_ai_opus_4_6_supports_minimal(self):
        entry = self.registry["azure_ai/claude-opus-4-6"]
        assert entry.get("supports_minimal_reasoning_effort") is True

    def test_azure_ai_opus_4_7_supports_max(self):
        entry = self.registry["azure_ai/claude-opus-4-7"]
        assert entry.get("supports_max_reasoning_effort") is True
        assert entry.get("supports_minimal_reasoning_effort") is True


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
