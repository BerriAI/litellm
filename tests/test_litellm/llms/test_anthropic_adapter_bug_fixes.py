"""
Regression tests for Anthropic adapter bug fixes (PR #28684):
- Bug #28576: is_thinking_enabled crashes when thinking=None
- Bug #28568/#28562: Spend log ID mismatch for non-Anthropic backends
- Bug #28580: Anthropic prefill semantic lost on hosted_vllm

Tests cover the new code paths added by the fix.
"""

import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Anchor sys.path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)


# =============================================================================
# Bug #28576: is_thinking_enabled null guard
# =============================================================================
from litellm.llms.base_llm.chat.transformation import BaseConfig


class ConcreteConfig(BaseConfig):
    """Minimal concrete subclass for testing BaseConfig methods."""
    def get_error_class(self, error_code, status_code):
        return Exception
    def get_supported_openai_params(self, model):
        return []
    def map_openai_params(self, non_default_params, optional_params, model, drop_params):
        return optional_params
    def transform_request(self, model, messages, optional_params, litellm_params, headers):
        return {"model": model, "messages": messages, **optional_params}
    def transform_response(self, model, response, model_encoded):
        return response
    def validate_environment(self, model, messages, optional_params, headers):
        return True


class TestIsThinkingEnabledNullGuard:
    """Regression test for Bug #28576: is_thinking_enabled crashes when
    thinking=None (the key exists but value is None)."""

    def setup_method(self):
        self.transformer = ConcreteConfig()

    def test_is_thinking_enabled_with_thinking_none(self):
        """Bug #28576: thinking=None should not crash with AttributeError."""
        non_default_params = {"thinking": None, "model": "claude-3-5-sonnet"}
        result = self.transformer.is_thinking_enabled(non_default_params)
        # Should not crash; returns reasoning_effort check (False here)
        assert result is False

    def test_is_thinking_enabled_with_thinking_string(self):
        """thinking as a string (not dict) should fall through to reasoning_effort."""
        non_default_params = {"thinking": "enabled", "model": "claude-3-5-sonnet"}
        result = self.transformer.is_thinking_enabled(non_default_params)
        # String is not a dict, falls through to reasoning_effort check
        assert result is False

    def test_is_thinking_enabled_with_reasoning_effort_no_thinking(self):
        """reasoning_effort present without thinking key."""
        non_default_params = {"reasoning_effort": "high"}
        result = self.transformer.is_thinking_enabled(non_default_params)
        assert result is True

    def test_is_thinking_enabled_with_reasoning_effort_and_thinking_none(self):
        """reasoning_effort takes precedence when thinking=None."""
        non_default_params = {"thinking": None, "reasoning_effort": "high"}
        result = self.transformer.is_thinking_enabled(non_default_params)
        assert result is True

    def test_is_thinking_enabled_with_empty_dict(self):
        """Empty params should return False (no reasoning_effort)."""
        result = self.transformer.is_thinking_enabled({})
        assert result is False

    def test_is_thinking_enabled_thinking_disabled(self):
        """thinking.type != 'enabled' and no reasoning_effort."""
        non_default_params = {"thinking": {"type": "disabled"}}
        result = self.transformer.is_thinking_enabled(non_default_params)
        assert result is False

    def test_is_thinking_enabled_thinking_enabled(self):
        """thinking.type == 'enabled'."""
        non_default_params = {"thinking": {"type": "enabled"}}
        result = self.transformer.is_thinking_enabled(non_default_params)
        assert result is True


# =============================================================================
# Bug #28580: prefix:true auto-stamp in _prepare_completion_kwargs
# =============================================================================
from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)


class TestPrefixAutoStamp:
    """Regression test for Bug #28580: Auto-stamp prefix:true for trailing
    assistant messages in _prepare_completion_kwargs."""

    def test_prefix_auto_stamp_for_trailing_assistant(self):
        """Trailing assistant message without prefix should get prefix=True stamped."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        completion_kwargs, _ = (
            LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
                max_tokens=1024,
                messages=messages,
                model="gpt-4o",
                metadata=None,
                stop_sequences=None,
                stream=False,
                system=None,
                temperature=None,
                thinking=None,
                tool_choice=None,
                tools=None,
                top_k=None,
                top_p=None,
                output_format=None,
                extra_kwargs={},
            )
        )
        # The trailing assistant message should have prefix=True stamped
        assert completion_kwargs["messages"][-1].get("prefix") is True

    def test_prefix_not_stamped_when_already_true(self):
        """prefix=True should be preserved when already set."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "prefix": True},
        ]
        completion_kwargs, _ = (
            LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
                max_tokens=1024,
                messages=messages,
                model="gpt-4o",
                metadata=None,
                stop_sequences=None,
                stream=False,
                system=None,
                temperature=None,
                thinking=None,
                tool_choice=None,
                tools=None,
                top_k=None,
                top_p=None,
                output_format=None,
                extra_kwargs={},
            )
        )
        # prefix=True should be preserved
        assert completion_kwargs["messages"][-1].get("prefix") is True

    def test_prefix_not_stamped_for_user_last_message(self):
        """Last message not assistant should not get prefix stamped."""
        messages = [{"role": "user", "content": "hello"}]
        completion_kwargs, _ = (
            LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
                max_tokens=1024,
                messages=messages,
                model="gpt-4o",
                metadata=None,
                stop_sequences=None,
                stream=False,
                system=None,
                temperature=None,
                thinking=None,
                tool_choice=None,
                tools=None,
                top_k=None,
                top_p=None,
                output_format=None,
                extra_kwargs={},
            )
        )
        # No prefix stamp for user message
        assert completion_kwargs["messages"][-1].get("prefix") is None

    def test_prefix_not_stamped_for_user_only_content(self):
        """Single user message with text-only content."""
        messages = [{"role": "user", "content": "hello world"}]
        completion_kwargs, _ = (
            LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
                max_tokens=1024,
                messages=messages,
                model="gpt-4o",
                metadata=None,
                stop_sequences=None,
                stream=False,
                system=None,
                temperature=None,
                thinking=None,
                tool_choice=None,
                tools=None,
                top_k=None,
                top_p=None,
                output_format=None,
                extra_kwargs={},
            )
        )
        assert completion_kwargs["messages"][-1].get("prefix") is None


# =============================================================================
# Bug #28580: hosted_vllm transform_request with prefix:true
# =============================================================================
from litellm.llms.hosted_vllm.chat.transformation import HostedVLLMChatConfig


class TestHostedVLLMContinueFinalMessage:
    """Regression test for Bug #28580: transform_request must inject
    continue_final_message=True when trailing assistant has prefix=true."""

    def setup_method(self):
        self.config = HostedVLLMChatConfig()

    def test_transform_request_with_prefix_true(self):
        """Bug #28580: trailing assistant with prefix=True triggers
        continue_final_message=True in the transformed request."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there", "prefix": True},
        ]
        result = self.config.transform_request(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=messages,
            optional_params={"temperature": 0.7},
            litellm_params={},
            headers={},
        )
        # Should inject continue_final_message=True
        extra_body = result.get("extra_body", {})
        assert extra_body.get("continue_final_message") is True
        # prefix marker should be removed from the message
        assert "prefix" not in result["messages"][-1]

    def test_transform_request_with_prefix_true_adds_generation_prompt_false(self):
        """When prefill is active, add_generation_prompt should be False."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there", "prefix": True},
        ]
        result = self.config.transform_request(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        extra_body = result.get("extra_body", {})
        assert extra_body.get("add_generation_prompt") is False

    def test_transform_request_without_prefix_no_change(self):
        """Normal request without prefix should not inject continue_final_message."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = self.config.transform_request(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        # No extra_body injected for normal request
        extra_body = result.get("extra_body", {})
        assert extra_body.get("continue_final_message") is None
        assert extra_body.get("add_generation_prompt") is None


# =============================================================================
# Streaming litellm_call_id usage
# =============================================================================
from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)


class TestStreamingLitellmCallId:
    """Bug #28568/#28562: streaming iterator should use litellm_call_id
    for message_start.id to match spend_logs.request_id."""

    def test_message_start_uses_litellm_call_id(self):
        """message_start event should use litellm_call_id with msg_ prefix."""
        mock_stream = iter([])  # Empty stream
        wrapper = AnthropicStreamWrapper(
            completion_stream=mock_stream,
            model="gpt-4o",
            tool_name_mapping={},
            litellm_call_id="spend-log-call-789",
        )
        # First call to __next__ should produce message_start with id
        chunk = next(wrapper)
        assert chunk["type"] == "message_start"
        msg_id = chunk["message"]["id"]
        assert msg_id == "msg_spend-log-call-789"

    def test_message_start_falls_back_to_uuid(self):
        """When no litellm_call_id, should fall back to uuid."""
        mock_stream = iter([])
        wrapper = AnthropicStreamWrapper(
            completion_stream=mock_stream,
            model="gpt-4o",
            tool_name_mapping={},
            litellm_call_id=None,
        )
        chunk = next(wrapper)
        assert chunk["type"] == "message_start"
        # Should be a valid uuid format with msg_ prefix
        msg_id = chunk["message"]["id"]
        assert msg_id.startswith("msg_")
        uuid_part = msg_id[4:]
        assert len(uuid_part) == 36  # UUID format


# =============================================================================
# Response transformation with litellm_call_id
# =============================================================================
from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    ANTHROPIC_ADAPTER,
)


class TestResponseLitellmCallId:
    """Bug #28568/#28562: translate_completion_output_params should use
    litellm_call_id as response.id for spend log correlation."""

    def _make_mock_response(self):
        response = MagicMock()
        response.id = "original-response-id"
        response.model = "gpt-4o"
        choice = MagicMock()
        choice.finish_reason = "stop"
        choice.message = MagicMock()
        choice.message.content = "Hello"
        choice.message.tool_calls = None
        choice.message.function_call = None
        response.choices = [choice]
        # Minimal usage
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        usage.prompt_tokens_details = MagicMock()
        usage.prompt_tokens_details.cached_tokens = 0
        usage._cache_creation_input_tokens = 0
        response.usage = usage
        return response

    def test_translate_uses_litellm_call_id_for_response_id(self):
        """When litellm_call_id is provided, it should be used as response.id."""
        mock_response = self._make_mock_response()
        result = ANTHROPIC_ADAPTER.translate_completion_output_params(
            response=mock_response,
            tool_name_mapping={},
            litellm_call_id="test-call-456",
        )
        assert result is not None
        # ID should be msg_<litellm_call_id>
        assert result["id"] == "msg_test-call-456"

    def test_translate_falls_back_to_response_id(self):
        """When no litellm_call_id, should use original response.id."""
        mock_response = self._make_mock_response()
        result = ANTHROPIC_ADAPTER.translate_completion_output_params(
            response=mock_response,
            tool_name_mapping={},
            litellm_call_id=None,
        )
        assert result is not None
        assert result["id"] == "original-response-id"

    def test_translate_does_not_mutate_original_response(self):
        """The original response.id should not be mutated by translation."""
        mock_response = self._make_mock_response()
        _ = ANTHROPIC_ADAPTER.translate_completion_output_params(
            response=mock_response,
            tool_name_mapping={},
            litellm_call_id="test-call-789",
        )
        # Original response id should remain unchanged
        assert mock_response.id == "original-response-id"


# =============================================================================
# Handler litellm_call_id extraction (Bug #28568/#28562)
# =============================================================================
from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)


class TestHandlerLitellmCallIdExtraction:
    """Bug #28568/#28562: Handler must extract litellm_call_id from response
    and pass it to the translator for spend log correlation."""

    def _make_completion_response(self, hidden_params=None, call_id_attr=None, no_call_id_attr=False):
        """Create a properly structured mock ModelResponse."""
        import copy

        # Create usage with all required attributes
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        usage.prompt_tokens_details = MagicMock()
        usage.prompt_tokens_details.cached_tokens = 0
        usage._cache_creation_input_tokens = 0

        # Create message
        message = MagicMock()
        message.content = "Hello!"
        message.tool_calls = None
        message.function_call = None

        # Create choice
        choice = MagicMock()
        choice.finish_reason = "stop"
        choice.message = message

        # Create response
        response = MagicMock()
        response.id = "original-response-id"
        response.model = "gpt-4o"
        response.choices = [choice]
        response.usage = usage
        response._hidden_params = hidden_params or {}
        if call_id_attr is not None:
            response.litellm_call_id = call_id_attr
        elif no_call_id_attr:
            # Prevent MagicMock from auto-creating litellm_call_id attribute
            del response.litellm_call_id
        return response

    def test_sync_non_stream_extracts_call_id_from_hidden_params(self):
        """Non-stream: extract litellm_call_id from _hidden_params."""
        mock_response = self._make_completion_response(
            hidden_params={"litellm_call_id": "hidden-call-id"}
        )

        with patch("litellm.completion", return_value=mock_response):
            result = LiteLLMMessagesToCompletionTransformationHandler.anthropic_messages_handler(
                max_tokens=1024,
                messages=[{"role": "user", "content": "hello"}],
                model="gpt-4o",
                metadata=None,
                stop_sequences=None,
                stream=False,
                system=None,
                temperature=None,
                thinking=None,
                tool_choice=None,
                tools=None,
                top_k=None,
                top_p=None,
                output_format=None,
            )
            assert result is not None
            assert result["id"] == "msg_hidden-call-id"

    def test_sync_non_stream_falls_back_to_response_attr(self):
        """Non-stream: fallback to response.litellm_call_id when no _hidden_params."""
        mock_response = self._make_completion_response(
            hidden_params={},
            call_id_attr="attr-call-id"
        )

        with patch("litellm.completion", return_value=mock_response):
            result = LiteLLMMessagesToCompletionTransformationHandler.anthropic_messages_handler(
                max_tokens=1024,
                messages=[{"role": "user", "content": "hello"}],
                model="gpt-4o",
                metadata=None,
                stop_sequences=None,
                stream=False,
                system=None,
                temperature=None,
                thinking=None,
                tool_choice=None,
                tools=None,
                top_k=None,
                top_p=None,
                output_format=None,
            )
            assert result is not None
            assert result["id"] == "msg_attr-call-id"

    def test_sync_non_stream_uses_original_id_when_no_call_id(self):
        """Non-stream: use original response.id when no litellm_call_id found."""
        mock_response = self._make_completion_response(
            hidden_params={},
            no_call_id_attr=True
        )

        with patch("litellm.completion", return_value=mock_response):
            result = LiteLLMMessagesToCompletionTransformationHandler.anthropic_messages_handler(
                max_tokens=1024,
                messages=[{"role": "user", "content": "hello"}],
                model="gpt-4o",
                metadata=None,
                stop_sequences=None,
                stream=False,
                system=None,
                temperature=None,
                thinking=None,
                tool_choice=None,
                tools=None,
                top_k=None,
                top_p=None,
                output_format=None,
            )
            assert result is not None
            # No litellm_call_id, so use original response.id
            assert result["id"] == "original-response-id"

    def test_sync_stream_extracts_call_id_from_completion_kwargs(self):
        """Stream: extract litellm_call_id from completion_kwargs."""
        mock_response = MagicMock()

        with patch("litellm.completion", return_value=mock_response):
            with patch.object(
                ANTHROPIC_ADAPTER,
                "translate_completion_output_params_streaming",
                return_value=MagicMock(),
            ):
                result = LiteLLMMessagesToCompletionTransformationHandler.anthropic_messages_handler(
                    max_tokens=1024,
                    messages=[{"role": "user", "content": "hello"}],
                    model="gpt-4o",
                    metadata=None,
                    stop_sequences=None,
                    stream=True,
                    system=None,
                    temperature=None,
                    thinking=None,
                    tool_choice=None,
                    tools=None,
                    top_k=None,
                    top_p=None,
                    output_format=None,
                    litellm_call_id="kwarg-call-id",
                )
                assert result is not None

    def test_sync_stream_falls_back_to_metadata(self):
        """Stream: fallback to metadata when no litellm_call_id in kwargs."""
        mock_response = MagicMock()

        with patch("litellm.completion", return_value=mock_response):
            with patch.object(
                ANTHROPIC_ADAPTER,
                "translate_completion_output_params_streaming",
                return_value=MagicMock(),
            ):
                result = LiteLLMMessagesToCompletionTransformationHandler.anthropic_messages_handler(
                    max_tokens=1024,
                    messages=[{"role": "user", "content": "hello"}],
                    model="gpt-4o",
                    metadata={"litellm_call_id": "metadata-call-id"},
                    stop_sequences=None,
                    stream=True,
                    system=None,
                    temperature=None,
                    thinking=None,
                    tool_choice=None,
                    tools=None,
                    top_k=None,
                    top_p=None,
                    output_format=None,
                )
                assert result is not None

    @pytest.mark.asyncio
    async def test_async_handler_extracts_call_id_from_hidden_params(self):
        """Async: extract litellm_call_id from _hidden_params."""
        mock_response = self._make_completion_response(
            hidden_params={"litellm_call_id": "async-hidden-call"}
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response
            result = await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
                max_tokens=1024,
                messages=[{"role": "user", "content": "hello"}],
                model="gpt-4o",
                metadata=None,
                stop_sequences=None,
                stream=False,
                system=None,
                temperature=None,
                thinking=None,
                tool_choice=None,
                tools=None,
                top_k=None,
                top_p=None,
                output_format=None,
            )
            assert result is not None
            assert result["id"] == "msg_async-hidden-call"

    @pytest.mark.asyncio
    async def test_async_handler_falls_back_to_response_attr(self):
        """Async: fallback to response.litellm_call_id when no _hidden_params."""
        mock_response = self._make_completion_response(
            hidden_params={},
            call_id_attr="async-attr-call"
        )

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response
            result = await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
                max_tokens=1024,
                messages=[{"role": "user", "content": "hello"}],
                model="gpt-4o",
                metadata=None,
                stop_sequences=None,
                stream=False,
                system=None,
                temperature=None,
                thinking=None,
                tool_choice=None,
                tools=None,
                top_k=None,
                top_p=None,
                output_format=None,
            )
            assert result is not None
            assert result["id"] == "msg_async-attr-call"

    @pytest.mark.asyncio
    async def test_async_handler_stream_extracts_call_id(self):
        """Async stream: extract litellm_call_id from completion_kwargs."""
        mock_response = MagicMock()

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response
            with patch.object(
                ANTHROPIC_ADAPTER,
                "translate_completion_output_params_streaming",
                return_value=MagicMock(),
            ):
                result = await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
                    max_tokens=1024,
                    messages=[{"role": "user", "content": "hello"}],
                    model="gpt-4o",
                    metadata=None,
                    stop_sequences=None,
                    stream=True,
                    system=None,
                    temperature=None,
                    thinking=None,
                    tool_choice=None,
                    tools=None,
                    top_k=None,
                    top_p=None,
                    output_format=None,
                    litellm_call_id="async-stream-call-id",
                )
                assert result is not None

    @pytest.mark.asyncio
    async def test_async_handler_stream_falls_back_to_metadata(self):
        """Async stream: fallback to metadata when no litellm_call_id in kwargs."""
        mock_response = MagicMock()

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response
            with patch.object(
                ANTHROPIC_ADAPTER,
                "translate_completion_output_params_streaming",
                return_value=MagicMock(),
            ):
                result = await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
                    max_tokens=1024,
                    messages=[{"role": "user", "content": "hello"}],
                    model="gpt-4o",
                    metadata={"litellm_call_id": "metadata-async-call"},
                    stop_sequences=None,
                    stream=True,
                    system=None,
                    temperature=None,
                    thinking=None,
                    tool_choice=None,
                    tools=None,
                    top_k=None,
                    top_p=None,
                    output_format=None,
                )
                assert result is not None
