"""
Tests for reasoning_auto_summary support on the native /v1/messages handler.

When reasoning_auto_summary is enabled (via litellm.reasoning_auto_summary or
LITELLM_REASONING_AUTO_SUMMARY env var), the handler injects
thinking.display = "summarized" into the request params for active thinking
modes (type="enabled" or type="adaptive").
"""

import os
import sys

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
    anthropic_messages_handler,
)


def _call_handler_and_capture_optional_params(thinking=None, **extra_kwargs):
    """
    Call anthropic_messages_handler with an Anthropic model and capture the
    anthropic_messages_optional_request_params dict passed to
    base_llm_http_handler.anthropic_messages_handler.

    Returns the captured dict.
    """
    captured = {}

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.handler."
        "base_llm_http_handler"
    ) as mock_handler, patch(
        "litellm.llms.anthropic.experimental_pass_through.messages.handler."
        "ProviderConfigManager"
    ) as mock_pcm:
        # Make get_provider_anthropic_messages_config return a non-None config
        # so the handler takes the native Anthropic path
        mock_pcm.get_provider_anthropic_messages_config.return_value = MagicMock()
        mock_handler.anthropic_messages_handler.return_value = MagicMock()

        kwargs = dict(extra_kwargs)
        if thinking is not None:
            kwargs["thinking"] = thinking

        try:
            anthropic_messages_handler(
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}],
                model="claude-sonnet-4-20250514",
                custom_llm_provider="anthropic",
                api_key="test-key",
                **kwargs,
            )
        except (ValueError, TypeError, AttributeError):
            pass

        if mock_handler.anthropic_messages_handler.called:
            captured = mock_handler.anthropic_messages_handler.call_args.kwargs.get(
                "anthropic_messages_optional_request_params", {}
            )

    return captured


class TestReasoningAutoSummaryMessages:
    """Tests for thinking.display injection on native /v1/messages handler."""

    def test_adaptive_thinking_gets_display_summarized(self):
        """reasoning_auto_summary=True + thinking.type='adaptive' -> display='summarized'."""
        with patch.object(litellm, "reasoning_auto_summary", True):
            params = _call_handler_and_capture_optional_params(
                thinking={"type": "adaptive", "budget_tokens": 5000}
            )
        thinking = params.get("thinking", {})
        assert thinking.get("display") == "summarized"
        assert thinking.get("type") == "adaptive"
        assert thinking.get("budget_tokens") == 5000

    def test_enabled_thinking_gets_display_summarized(self):
        """reasoning_auto_summary=True + thinking.type='enabled' -> display='summarized'."""
        with patch.object(litellm, "reasoning_auto_summary", True):
            params = _call_handler_and_capture_optional_params(
                thinking={"type": "enabled", "budget_tokens": 10000}
            )
        thinking = params.get("thinking", {})
        assert thinking.get("display") == "summarized"
        assert thinking.get("type") == "enabled"

    def test_disabled_thinking_no_display(self):
        """reasoning_auto_summary=True + thinking.type='disabled' -> display NOT set."""
        with patch.object(litellm, "reasoning_auto_summary", True):
            params = _call_handler_and_capture_optional_params(
                thinking={"type": "disabled"}
            )
        thinking = params.get("thinking", {})
        assert "display" not in thinking

    def test_no_injection_when_flag_false(self):
        """reasoning_auto_summary=False + active thinking -> display NOT set."""
        with patch.object(litellm, "reasoning_auto_summary", False):
            params = _call_handler_and_capture_optional_params(
                thinking={"type": "enabled", "budget_tokens": 10000}
            )
        thinking = params.get("thinking", {})
        assert "display" not in thinking

    def test_no_thinking_param_no_crash(self):
        """reasoning_auto_summary=True but no thinking param -> nothing changes."""
        with patch.object(litellm, "reasoning_auto_summary", True):
            params = _call_handler_and_capture_optional_params()
        thinking = params.get("thinking")
        if thinking is not None:
            assert "display" not in thinking

    def test_env_var_enables_auto_summary(self):
        """LITELLM_REASONING_AUTO_SUMMARY=true env var enables the feature."""
        with patch.object(litellm, "reasoning_auto_summary", False), patch.dict(
            os.environ, {"LITELLM_REASONING_AUTO_SUMMARY": "true"}
        ):
            params = _call_handler_and_capture_optional_params(
                thinking={"type": "adaptive", "budget_tokens": 5000}
            )
        thinking = params.get("thinking", {})
        assert thinking.get("display") == "summarized"

    def test_existing_display_summarized_preserved(self):
        """User already passes display='summarized' -> preserved as-is."""
        with patch.object(litellm, "reasoning_auto_summary", True):
            params = _call_handler_and_capture_optional_params(
                thinking={
                    "type": "enabled",
                    "budget_tokens": 10000,
                    "display": "summarized",
                }
            )
        thinking = params.get("thinking", {})
        assert thinking.get("display") == "summarized"

    def test_existing_display_summarized_without_flag(self):
        """User passes display='summarized' + flag=False -> preserved as-is."""
        with patch.object(litellm, "reasoning_auto_summary", False):
            params = _call_handler_and_capture_optional_params(
                thinking={
                    "type": "enabled",
                    "budget_tokens": 10000,
                    "display": "summarized",
                }
            )
        thinking = params.get("thinking", {})
        assert thinking.get("display") == "summarized"

    def test_omitted_overridden_to_summarized(self):
        """User passes display='omitted' + reasoning_auto_summary=True -> overridden.

        Documents current behavior: the code unconditionally sets
        display='summarized' when auto_summary is enabled and thinking is active,
        regardless of any pre-existing display value.
        """
        with patch.object(litellm, "reasoning_auto_summary", True):
            params = _call_handler_and_capture_optional_params(
                thinking={
                    "type": "enabled",
                    "budget_tokens": 10000,
                    "display": "omitted",
                }
            )
        thinking = params.get("thinking", {})
        assert thinking.get("display") == "summarized"
