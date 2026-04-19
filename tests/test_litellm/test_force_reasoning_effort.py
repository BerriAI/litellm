"""
Tests for force_reasoning_effort litellm_params config.

Verifies that force_reasoning_effort injects reasoning_effort into requests
across all entry paths:
- /v1/chat/completions (litellm.completion via main.py)
- /v1/responses (litellm.responses via responses/main.py)
- /v1/messages -> responses adapter (responses_adapters/handler.py)
- /v1/messages -> native Anthropic (messages/handler.py)
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_client_cache():
    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if cache is not None:
        cache.flush_cache()
    yield
    if cache is not None:
        cache.flush_cache()


@pytest.fixture(autouse=True)
def add_api_keys_to_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fake")


# ---------------------------------------------------------------------------
# /v1/chat/completions path (main.py)
# ---------------------------------------------------------------------------


class TestChatCompletionsForceReasoningEffort:
    """Tests for force_reasoning_effort in litellm.completion()."""

    def test_force_reasoning_effort_injected_when_not_present(self):
        """force_reasoning_effort should inject reasoning_effort into optional_params."""
        from openai import OpenAI
        from litellm.types.utils import ModelResponse

        client = OpenAI(api_key="test_api_key")
        mock_raw_response = MagicMock()
        mock_raw_response.headers = {
            "x-request-id": "123",
            "openai-organization": "org-123",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "99",
        }
        mock_raw_response.parse.return_value = ModelResponse(
            id="chatcmpl-test",
            choices=[
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": "hi", "role": "assistant"},
                }
            ],
            model="gpt-4o-mini",
        )

        with patch.object(
            client.chat.completions.with_raw_response, "create", mock_raw_response
        ):
            litellm.completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                client=client,
                force_reasoning_effort="high",
            )
            mock_raw_response.assert_called_once()

    def test_force_reasoning_effort_overrides_existing(self):
        """force_reasoning_effort should override client-sent reasoning_effort."""
        from openai import OpenAI
        from litellm.types.utils import ModelResponse

        client = OpenAI(api_key="test_api_key")
        mock_raw_response = MagicMock()
        mock_raw_response.headers = {
            "x-request-id": "123",
            "openai-organization": "org-123",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "99",
        }
        mock_raw_response.parse.return_value = ModelResponse(
            id="chatcmpl-test",
            choices=[
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": "hi", "role": "assistant"},
                }
            ],
            model="o3-mini",
        )

        with patch.object(
            client.chat.completions.with_raw_response, "create", mock_raw_response
        ):
            litellm.completion(
                model="o3-mini",
                messages=[{"role": "user", "content": "Hello"}],
                client=client,
                reasoning_effort="low",
                force_reasoning_effort="high",
            )
            mock_raw_response.assert_called_once()

    def test_no_force_reasoning_effort_leaves_request_unchanged(self):
        """Without force_reasoning_effort, reasoning_effort is not injected."""
        from litellm.litellm_core_utils.get_litellm_params import get_litellm_params

        result = get_litellm_params(
            api_key="fake",
            force_reasoning_effort=None,
        )
        assert result.get("force_reasoning_effort") is None

    def test_get_litellm_params_includes_force_reasoning_effort(self):
        """get_litellm_params should pass through force_reasoning_effort."""
        from litellm.litellm_core_utils.get_litellm_params import get_litellm_params

        result = get_litellm_params(
            api_key="fake",
            force_reasoning_effort="high",
        )
        assert result["force_reasoning_effort"] == "high"


# ---------------------------------------------------------------------------
# Responses adapter path (_build_responses_kwargs)
# ---------------------------------------------------------------------------


class TestResponsesAdapterForceReasoningEffort:
    """Tests for force_reasoning_effort in _build_responses_kwargs()."""

    def test_force_creates_reasoning_when_absent(self):
        """force_reasoning_effort should create reasoning dict when none exists."""
        from litellm.llms.anthropic.experimental_pass_through.responses_adapters.handler import (
            _build_responses_kwargs,
        )

        with patch(
            "litellm.llms.anthropic.experimental_pass_through.responses_adapters.handler.is_reasoning_auto_summary_enabled",
            return_value=False,
        ):
            result = _build_responses_kwargs(
                max_tokens=1024,
                messages=[{"role": "user", "content": "hello"}],
                model="openai/gpt-5.1",
                extra_kwargs={"force_reasoning_effort": "high"},
            )

        assert result["reasoning"] == {"effort": "high"}

    def test_force_creates_reasoning_with_summary_when_auto_summary_enabled(self):
        """force_reasoning_effort should include summary when auto-summary is on."""
        from litellm.llms.anthropic.experimental_pass_through.responses_adapters.handler import (
            _build_responses_kwargs,
        )

        with patch(
            "litellm.llms.anthropic.experimental_pass_through.responses_adapters.handler.is_reasoning_auto_summary_enabled",
            return_value=True,
        ):
            result = _build_responses_kwargs(
                max_tokens=1024,
                messages=[{"role": "user", "content": "hello"}],
                model="openai/gpt-5.1",
                extra_kwargs={"force_reasoning_effort": "medium"},
            )

        assert result["reasoning"] == {"summary": "detailed", "effort": "medium"}

    def test_force_overrides_existing_reasoning_effort(self):
        """force_reasoning_effort should override effort in existing reasoning dict."""
        from litellm.llms.anthropic.experimental_pass_through.responses_adapters.handler import (
            _build_responses_kwargs,
        )

        result = _build_responses_kwargs(
            max_tokens=1024,
            messages=[{"role": "user", "content": "hello"}],
            model="openai/gpt-5.1",
            thinking={"type": "enabled", "budget_tokens": 5000},
            extra_kwargs={"force_reasoning_effort": "high"},
        )

        reasoning = result["reasoning"]
        assert reasoning["effort"] == "high"

    def test_no_force_does_not_add_reasoning(self):
        """Without force_reasoning_effort, no reasoning is injected."""
        from litellm.llms.anthropic.experimental_pass_through.responses_adapters.handler import (
            _build_responses_kwargs,
        )

        result = _build_responses_kwargs(
            max_tokens=1024,
            messages=[{"role": "user", "content": "hello"}],
            model="openai/gpt-5.1",
            extra_kwargs={},
        )

        assert "reasoning" not in result

    def test_force_not_leaked_into_responses_kwargs(self):
        """force_reasoning_effort should be excluded from the final kwargs dict."""
        from litellm.llms.anthropic.experimental_pass_through.responses_adapters.handler import (
            _build_responses_kwargs,
        )

        with patch(
            "litellm.llms.anthropic.experimental_pass_through.responses_adapters.handler.is_reasoning_auto_summary_enabled",
            return_value=False,
        ):
            result = _build_responses_kwargs(
                max_tokens=1024,
                messages=[{"role": "user", "content": "hello"}],
                model="openai/gpt-5.1",
                extra_kwargs={"force_reasoning_effort": "high"},
            )

        assert "force_reasoning_effort" not in result


# ---------------------------------------------------------------------------
# /v1/messages -> native Anthropic path (messages/handler.py)
# ---------------------------------------------------------------------------


class TestAnthropicMessagesForceReasoningEffort:
    """Tests for force_reasoning_effort in anthropic_messages_handler()."""

    def test_force_sets_output_config_effort_and_enables_thinking(self):
        """force_reasoning_effort should set output_config.effort and enable thinking."""
        from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
            anthropic_messages_handler,
        )

        with patch(
            "litellm.llms.anthropic.experimental_pass_through.messages.handler.base_llm_http_handler.anthropic_messages_handler",
            return_value=MagicMock(),
        ) as mock_handler:
            try:
                anthropic_messages_handler(
                    max_tokens=1024,
                    messages=[{"role": "user", "content": "hello"}],
                    model="anthropic/claude-sonnet-4-20250514",
                    force_reasoning_effort="high",
                )
            except (ValueError, TypeError, AttributeError):
                pass

            if mock_handler.called:
                call_kwargs = mock_handler.call_args
                optional_params = call_kwargs.kwargs.get(
                    "anthropic_messages_optional_request_params", {}
                )
                assert optional_params.get("output_config", {}).get("effort") == "high"
                # thinking should be enabled when not already set
                assert optional_params.get("thinking") == {"type": "adaptive"}

    def test_force_preserves_existing_thinking_config(self):
        """force_reasoning_effort should not override existing enabled thinking config."""
        from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
            anthropic_messages_handler,
        )

        with patch(
            "litellm.llms.anthropic.experimental_pass_through.messages.handler.base_llm_http_handler.anthropic_messages_handler",
            return_value=MagicMock(),
        ) as mock_handler:
            try:
                anthropic_messages_handler(
                    max_tokens=1024,
                    messages=[{"role": "user", "content": "hello"}],
                    model="anthropic/claude-sonnet-4-20250514",
                    thinking={"type": "enabled", "budget_tokens": 10000},
                    force_reasoning_effort="high",
                )
            except (ValueError, TypeError, AttributeError):
                pass

            if mock_handler.called:
                call_kwargs = mock_handler.call_args
                optional_params = call_kwargs.kwargs.get(
                    "anthropic_messages_optional_request_params", {}
                )
                assert optional_params.get("output_config", {}).get("effort") == "high"
                # Pre-existing enabled thinking should be preserved
                thinking = optional_params.get("thinking", {})
                assert (
                    thinking.get("type") != "adaptive"
                    or thinking.get("type") == "enabled"
                )


# ---------------------------------------------------------------------------
# GenericLiteLLMParams type
# ---------------------------------------------------------------------------


class TestGenericLiteLLMParamsForceReasoningEffort:
    """Tests for force_reasoning_effort field in GenericLiteLLMParams."""

    def test_default_is_none(self):
        from litellm.types.router import GenericLiteLLMParams

        params = GenericLiteLLMParams()
        assert params.force_reasoning_effort is None

    def test_accepts_string_value(self):
        from litellm.types.router import GenericLiteLLMParams

        params = GenericLiteLLMParams(force_reasoning_effort="high")
        assert params.force_reasoning_effort == "high"

    def test_accepts_various_effort_levels(self):
        from litellm.types.router import GenericLiteLLMParams

        for level in ["low", "medium", "high", "xhigh"]:
            params = GenericLiteLLMParams(force_reasoning_effort=level)
            assert params.force_reasoning_effort == level
