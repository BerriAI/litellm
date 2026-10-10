import asyncio
import json
import os
import uuid
from typing import Any, Dict, Final, List

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError


from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm.anthropic_interface import messages
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.router import Router
from litellm.types.utils import (
    Delta,
    ModelResponse,
    StandardLoggingPayloadErrorInformation,
    StreamingChoices,
)
import unittest.mock


def test_anthropic_experimental_pass_through_messages_handler():
    """
    Test that api key is passed to litellm.responses for OpenAI models.
    OpenAI and Azure models are routed directly to the Responses API.
    """
    from litellm.llms.anthropic.pass_through.messages.handler import (
        anthropic_messages_handler,
    )

    with patch("litellm.responses", return_value="test-response") as mock_responses:
        try:
            anthropic_messages_handler(
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                model="openai/claude-3-5-sonnet-20240620",
                api_key="test-api-key",
            )
        except (ValueError, TypeError, AttributeError) as e:
            print(f"Error: {e}")
        mock_responses.assert_called_once()
        assert mock_responses.call_args.kwargs["api_key"] == "test-api-key"


@pytest.mark.asyncio
async def test_openai_model_does_not_forward_stream_options_to_responses_api():
    """
    Regression test for LIT-4779. `always_include_stream_usage` injects
    stream_options={'include_usage': True} into every streaming request, but OpenAI
    models on /v1/messages go to the Responses API, which 400s on that param.
    """
    responses_payload = {
        "id": "resp_stream_options",
        "object": "response",
        "created_at": 1734366691,
        "status": "completed",
        "model": "gpt-5.5",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi", "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": None,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": None,
        "truncation": None,
        "user": None,
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = json.dumps(responses_payload)
    mock_response.headers = httpx.Headers({})
    mock_response.json.return_value = responses_payload

    with patch.object(AsyncHTTPHandler, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        await litellm.anthropic.messages.acreate(
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            model="openai/gpt-5.5",
            api_key="test-api-key",
            stream_options={"include_usage": True},
        )

    mock_post.assert_called_once()
    post_kwargs = mock_post.call_args.kwargs
    request_body = post_kwargs["json"] if "json" in post_kwargs else json.loads(post_kwargs["data"])
    assert "stream_options" not in request_body


def test_anthropic_experimental_pass_through_messages_handler_dynamic_api_key_and_api_base_and_custom_values():
    """
    Test that api key, api base, and extra kwargs are forwarded to litellm.completion for Azure models.
    Azure models are routed through chat/completions (not the Responses API).
    """
    from litellm.llms.anthropic.pass_through.messages.handler import (
        anthropic_messages_handler,
    )

    with patch("litellm.completion", return_value=MagicMock()) as mock_completion:
        try:
            anthropic_messages_handler(
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                model="azure/o1",
                api_key="test-api-key",
                api_base="test-api-base",
                custom_key="custom_value",
            )
        except (ValueError, TypeError, AttributeError) as e:
            print(f"Error: {e}")
        mock_completion.assert_called_once()
        assert mock_completion.call_args.kwargs["api_key"] == "test-api-key"
        assert mock_completion.call_args.kwargs["api_base"] == "test-api-base"
        assert mock_completion.call_args.kwargs["custom_key"] == "custom_value"


@pytest.mark.asyncio
async def test_anthropic_messages_sanitizes_empty_text_blocks_before_dispatch():
    """Regression test for #22930.  The unified /v1/messages path must
    strip empty text blocks before forwarding, otherwise Anthropic
    returns 400 "text content blocks must be non-empty"."""
    from litellm.llms.anthropic.pass_through.messages import handler

    msgs = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ""},
                {"type": "tool_use", "id": "t", "name": "B", "input": {}},
            ],
        }
    ]
    captured = {}

    def fake_handler(*args, **kwargs):
        captured["messages"] = kwargs.get("messages")
        return "stub"

    fake_loop = MagicMock()
    fake_loop.run_in_executor = lambda _e, func: _async_return(func())

    with (
        patch.object(handler, "anthropic_messages_handler", side_effect=fake_handler),
        patch("asyncio.get_event_loop", return_value=fake_loop),
    ):
        await handler.anthropic_messages(
            max_tokens=100,
            messages=msgs,
            model="anthropic/claude-sonnet-4-5-20250929",
            custom_llm_provider="anthropic",
            api_key="k",
        )

    assert [b["type"] for b in captured["messages"][0]["content"]] == ["tool_use"]
    assert len(msgs[0]["content"]) == 2  # caller untouched


@pytest.mark.asyncio
async def test_anthropic_messages_sanitizes_tool_use_ids_before_dispatch():
    from litellm.llms.anthropic.pass_through.messages import handler

    msgs = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "functions.Bash:0",
                    "name": "Bash",
                    "input": {},
                }
            ],
        }
    ]
    captured = {}

    def fake_handler(*args, **kwargs):
        captured["messages"] = kwargs.get("messages")
        return "stub"

    fake_loop = MagicMock()
    fake_loop.run_in_executor = lambda _e, func: _async_return(func())

    with (
        patch.object(handler, "anthropic_messages_handler", side_effect=fake_handler),
        patch("asyncio.get_event_loop", return_value=fake_loop),
    ):
        await handler.anthropic_messages(
            max_tokens=100,
            messages=msgs,
            model="anthropic/claude-sonnet-4-5-20250929",
            custom_llm_provider="anthropic",
            api_key="k",
        )

    assert captured["messages"][0]["content"][0]["id"] == "functions_Bash_0"
    assert msgs[0]["content"][0]["id"] == "functions.Bash:0"


async def _async_return(value):
    return value


def test_anthropic_experimental_pass_through_messages_handler_custom_llm_provider():
    """
    Test that litellm.completion is called when a custom LLM provider is given.

    Provider resolution now happens exactly once, inside litellm.completion itself
    (BerriAI/litellm#37716), so the handler passes the original unresolved model through.
    """
    from litellm.llms.anthropic.pass_through.messages.handler import (
        anthropic_messages_handler,
    )

    with patch("litellm.completion", return_value="test-response") as mock_completion:
        try:
            anthropic_messages_handler(
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                model="my-custom-model",
                custom_llm_provider="my-custom-llm",
                api_key="test-api-key",
            )
        except (ValueError, TypeError, AttributeError) as e:
            print(f"Error: {e}")

        # Assert that litellm.completion was called when using a custom LLM provider
        mock_completion.assert_called_once()

        # Verify that the custom provider was passed through
        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["custom_llm_provider"] == "my-custom-llm"
        assert call_kwargs["model"] == "my-custom-model"
        assert call_kwargs["api_key"] == "test-api-key"


@pytest.mark.asyncio
async def test_bedrock_converse_budget_tokens_preserved():
    """
    Test that budget_tokens value in thinking parameter is correctly passed to Bedrock Converse API
    when using messages.acreate with bedrock/converse model.

    The bug was that the messages -> completion adapter was converting thinking to reasoning_effort
    and losing the original budget_tokens value, causing it to use the default (128) instead.
    """
    # Mock litellm.acompletion which is called internally by anthropic_messages_handler
    mock_response = ModelResponse(
        id="test-id",
        model="bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "4"},
                "finish_reason": "stop",
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_response

        try:
            await messages.acreate(
                max_tokens=1024,
                messages=[{"role": "user", "content": "What is 2+2?"}],
                model="bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0",
                thinking={"budget_tokens": 1024, "type": "enabled"},
            )
        except (ValueError, TypeError, AttributeError):
            pass  # Expected due to response format conversion

        mock_acompletion.assert_called_once()

        call_kwargs = mock_acompletion.call_args.kwargs
        print("acompletion call kwargs: ", json.dumps(call_kwargs, indent=4, default=str))

        # Verify thinking parameter is passed through with budget_tokens preserved
        thinking_param = call_kwargs.get("thinking")
        assert thinking_param is not None, "thinking parameter should be passed to acompletion"
        assert thinking_param.get("type") == "enabled", "thinking.type should be 'enabled'"
        assert thinking_param.get("budget_tokens") == 1024, (
            f"thinking.budget_tokens should be 1024, but got {thinking_param.get('budget_tokens')}"
        )


def test_openai_model_with_thinking_converts_to_reasoning():
    """
    Test that when using an OpenAI model with thinking parameter, the thinking is
    converted to a Responses API `reasoning` param (NOT passed as thinking).

    OpenAI models are routed directly to the Responses API, so we verify that
    litellm.responses() is called with `reasoning` properly set.
    """
    from litellm.llms.anthropic.pass_through.messages.handler import (
        anthropic_messages_handler,
    )

    with patch("litellm.responses", return_value="test-response") as mock_responses:
        try:
            anthropic_messages_handler(
                max_tokens=1024,
                messages=[{"role": "user", "content": "What is 2+2?"}],
                model="openai/gpt-5.2",
                api_key="test-api-key",
                thinking={"type": "enabled", "budget_tokens": 1024},
            )
        except (ValueError, TypeError, AttributeError) as e:
            print(f"Error: {e}")

        mock_responses.assert_called_once()

        call_kwargs = mock_responses.call_args.kwargs

        # Verify reasoning is set (converted from thinking)
        assert "reasoning" in call_kwargs, "reasoning should be passed to litellm.responses"

        # budget_tokens=1024 -> effort="low" (at the LOW budget threshold)
        # reasoning_auto_summary is False by default, so no summary key
        expected_reasoning = {"effort": "low"}
        assert call_kwargs["reasoning"] == expected_reasoning, (
            f"reasoning should be {expected_reasoning} for budget_tokens=1024, got {call_kwargs.get('reasoning')}"
        )
        assert "summary" not in call_kwargs["reasoning"]

        # Verify thinking is NOT passed directly to the Responses API
        assert "thinking" not in call_kwargs, "thinking should NOT be passed directly to litellm.responses"


class TestThinkingParameterTransformation:
    """Core tests for thinking parameter transformation logic."""

    def test_claude_model_preserves_thinking_with_budget_tokens(self):
        """Test that Claude models get thinking parameter passed through with exact budget_tokens."""
        from litellm.llms.anthropic.pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )

        thinking = {"type": "enabled", "budget_tokens": 5000}
        result = LiteLLMAnthropicMessagesAdapter.translate_thinking_for_model(
            thinking=thinking,
            model="bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0",
        )

        assert result == {"thinking": thinking}
        assert result["thinking"]["budget_tokens"] == 5000

    def test_non_claude_model_converts_thinking_to_reasoning_effort(self):
        """Test that non-Claude models convert thinking to reasoning_effort."""
        from litellm.llms.anthropic.pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )

        thinking = {"type": "enabled", "budget_tokens": 1024}
        result = LiteLLMAnthropicMessagesAdapter.translate_thinking_for_model(
            thinking=thinking,
            model="openai/gpt-5.2",
        )

        # reasoning_auto_summary is False by default, so no summary key
        assert result == {"reasoning_effort": "low"}
        assert "thinking" not in result
        assert "summary" not in str(result["reasoning_effort"])

    def test_translate_thinking_for_model_summary_when_enabled(self):
        """When reasoning_auto_summary is True, summary='detailed' is injected."""
        import litellm
        from litellm.llms.anthropic.pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )

        original = litellm.reasoning_auto_summary
        try:
            litellm.reasoning_auto_summary = True
            thinking = {"type": "enabled", "budget_tokens": 5000}
            result = LiteLLMAnthropicMessagesAdapter.translate_thinking_for_model(
                thinking=thinking,
                model="openai/gpt-5.2",
            )
            assert result == {"reasoning_effort": {"effort": "high", "summary": "detailed"}}
        finally:
            litellm.reasoning_auto_summary = original

    def test_translate_thinking_for_model_preserves_user_summary(self):
        """User-provided summary is always preserved regardless of flag."""
        from litellm.llms.anthropic.pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )

        thinking = {"type": "enabled", "budget_tokens": 10000, "summary": "concise"}
        result = LiteLLMAnthropicMessagesAdapter.translate_thinking_for_model(
            thinking=thinking,
            model="openai/gpt-5.2",
        )
        assert result == {"reasoning_effort": {"effort": "high", "summary": "concise"}}


class TestThinkingSummaryPreservation:
    """Tests for thinking.summary preservation and reasoning_auto_summary flag."""

    def test_thinking_summary_concise_preserved_for_openai(self):
        """User-provided summary='concise' should not be replaced with 'detailed'."""
        from litellm.llms.anthropic.pass_through.adapters.handler import (
            LiteLLMMessagesToCompletionTransformationHandler,
        )

        thinking = {"type": "enabled", "budget_tokens": 5000, "summary": "concise"}
        completion_kwargs = {"model": "openai/gpt-5.1", "reasoning_effort": "medium"}
        LiteLLMMessagesToCompletionTransformationHandler._route_openai_thinking_to_responses_api_if_needed(
            completion_kwargs, thinking=thinking
        )
        assert completion_kwargs["reasoning_effort"] == {
            "effort": "medium",
            "summary": "concise",
        }

    def test_thinking_summary_auto_preserved_for_openai(self):
        """User-provided summary='auto' should be preserved."""
        from litellm.llms.anthropic.pass_through.adapters.handler import (
            LiteLLMMessagesToCompletionTransformationHandler,
        )

        thinking = {"type": "enabled", "budget_tokens": 10000, "summary": "auto"}
        completion_kwargs = {"model": "openai/gpt-5.1", "reasoning_effort": "high"}
        LiteLLMMessagesToCompletionTransformationHandler._route_openai_thinking_to_responses_api_if_needed(
            completion_kwargs, thinking=thinking
        )
        assert completion_kwargs["reasoning_effort"] == {
            "effort": "high",
            "summary": "auto",
        }

    def test_summary_added_when_auto_summary_enabled(self):
        """When reasoning_auto_summary is True, summary='detailed' is added."""
        import litellm
        from litellm.llms.anthropic.pass_through.adapters.handler import (
            LiteLLMMessagesToCompletionTransformationHandler,
        )

        original = litellm.reasoning_auto_summary
        try:
            litellm.reasoning_auto_summary = True
            completion_kwargs = {
                "model": "responses/gpt-5.2",
                "custom_llm_provider": "openai",
                "reasoning_effort": "medium",
            }
            LiteLLMMessagesToCompletionTransformationHandler._route_openai_thinking_to_responses_api_if_needed(
                completion_kwargs, thinking={"type": "enabled", "budget_tokens": 5000}
            )
            assert completion_kwargs["reasoning_effort"] == {
                "effort": "medium",
                "summary": "detailed",
            }
        finally:
            litellm.reasoning_auto_summary = original

    def test_no_summary_by_default_string_reasoning(self):
        """By default (reasoning_auto_summary=False), summary is not added for string reasoning_effort."""
        import litellm
        from litellm.llms.anthropic.pass_through.adapters.handler import (
            LiteLLMMessagesToCompletionTransformationHandler,
        )

        original = litellm.reasoning_auto_summary
        try:
            litellm.reasoning_auto_summary = False
            completion_kwargs = {
                "model": "responses/gpt-5.2",
                "custom_llm_provider": "openai",
                "reasoning_effort": "high",
            }
            LiteLLMMessagesToCompletionTransformationHandler._route_openai_thinking_to_responses_api_if_needed(
                completion_kwargs, thinking={"type": "enabled", "budget_tokens": 10000}
            )
            assert completion_kwargs["reasoning_effort"] == {"effort": "high"}
            assert "summary" not in completion_kwargs["reasoning_effort"]
        finally:
            litellm.reasoning_auto_summary = original

    def test_no_summary_by_default_dict_reasoning(self):
        """By default (reasoning_auto_summary=False), summary is not injected into dict reasoning_effort."""
        import litellm
        from litellm.llms.anthropic.pass_through.adapters.handler import (
            LiteLLMMessagesToCompletionTransformationHandler,
        )

        original = litellm.reasoning_auto_summary
        try:
            litellm.reasoning_auto_summary = False
            completion_kwargs = {
                "model": "responses/gpt-5.2",
                "custom_llm_provider": "openai",
                "reasoning_effort": {"effort": "medium"},
            }
            LiteLLMMessagesToCompletionTransformationHandler._route_openai_thinking_to_responses_api_if_needed(
                completion_kwargs, thinking={"type": "enabled", "budget_tokens": 5000}
            )
            assert completion_kwargs["reasoning_effort"] == {"effort": "medium"}
            assert "summary" not in completion_kwargs["reasoning_effort"]
        finally:
            litellm.reasoning_auto_summary = original

    def test_summary_added_when_env_var_set(self, monkeypatch):
        """When LITELLM_REASONING_AUTO_SUMMARY env var is true, summary is added."""
        import litellm
        from litellm.llms.anthropic.pass_through.adapters.handler import (
            LiteLLMMessagesToCompletionTransformationHandler,
        )

        original = litellm.reasoning_auto_summary
        try:
            litellm.reasoning_auto_summary = False
            monkeypatch.setenv("LITELLM_REASONING_AUTO_SUMMARY", "true")
            completion_kwargs = {
                "model": "responses/gpt-5.2",
                "custom_llm_provider": "openai",
                "reasoning_effort": "high",
            }
            LiteLLMMessagesToCompletionTransformationHandler._route_openai_thinking_to_responses_api_if_needed(
                completion_kwargs, thinking={"type": "enabled", "budget_tokens": 10000}
            )
            assert completion_kwargs["reasoning_effort"] == {
                "effort": "high",
                "summary": "detailed",
            }
        finally:
            litellm.reasoning_auto_summary = original
            os.environ.pop("LITELLM_REASONING_AUTO_SUMMARY", None)

    def test_user_provided_summary_preserved_even_when_flag_off(self):
        """When user already set summary in dict reasoning_effort, it's preserved regardless of flag."""
        import litellm
        from litellm.llms.anthropic.pass_through.adapters.handler import (
            LiteLLMMessagesToCompletionTransformationHandler,
        )

        original = litellm.reasoning_auto_summary
        try:
            litellm.reasoning_auto_summary = False
            completion_kwargs = {
                "model": "responses/gpt-5.2",
                "custom_llm_provider": "openai",
                "reasoning_effort": {"effort": "high", "summary": "concise"},
            }
            LiteLLMMessagesToCompletionTransformationHandler._route_openai_thinking_to_responses_api_if_needed(
                completion_kwargs, thinking={"type": "enabled", "budget_tokens": 10000}
            )
            assert completion_kwargs["reasoning_effort"]["summary"] == "concise"
        finally:
            litellm.reasoning_auto_summary = original

    def test_openai_model_with_thinking_summary_end_to_end(self):
        """End-to-end: anthropic_messages_handler should preserve thinking.summary for OpenAI models."""
        from litellm.llms.anthropic.pass_through.messages.handler import (
            anthropic_messages_handler,
        )

        with patch("litellm.responses", return_value="test-response") as mock_responses:
            try:
                anthropic_messages_handler(
                    max_tokens=1024,
                    messages=[{"role": "user", "content": "What is 2+2?"}],
                    model="openai/gpt-5.2",
                    api_key="test-api-key",
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 5000,
                        "summary": "concise",
                    },
                )
            except (ValueError, TypeError, AttributeError):
                pass

            mock_responses.assert_called_once()
            call_kwargs = mock_responses.call_args.kwargs
            reasoning = call_kwargs["reasoning"]
            assert reasoning["summary"] == "concise", (
                f"Expected summary='concise', got summary='{reasoning.get('summary')}'"
            )

    def test_responses_adapter_preserves_summary(self):
        """translate_thinking_to_reasoning should include summary when user provides it."""
        from litellm.llms.anthropic.pass_through.responses_adapters.transformation import (
            LiteLLMAnthropicToResponsesAPIAdapter,
        )

        thinking = {"type": "enabled", "budget_tokens": 5000, "summary": "concise"}
        result = LiteLLMAnthropicToResponsesAPIAdapter.translate_thinking_to_reasoning(thinking)
        assert result == {"effort": "high", "summary": "concise"}

    def test_responses_adapter_no_summary_by_default(self):
        """translate_thinking_to_reasoning should not include summary by default (opt-in)."""
        import litellm
        from litellm.llms.anthropic.pass_through.responses_adapters.transformation import (
            LiteLLMAnthropicToResponsesAPIAdapter,
        )

        original = litellm.reasoning_auto_summary
        try:
            litellm.reasoning_auto_summary = False
            thinking = {"type": "enabled", "budget_tokens": 5000}
            result = LiteLLMAnthropicToResponsesAPIAdapter.translate_thinking_to_reasoning(thinking)
            assert result == {"effort": "high"}
            assert result is not None and "summary" not in result
        finally:
            litellm.reasoning_auto_summary = original

    def test_translate_thinking_for_model_preserves_summary(self):
        """translate_thinking_for_model should include summary in reasoning_effort dict when user provides it."""
        from litellm.llms.anthropic.pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )

        thinking = {"type": "enabled", "budget_tokens": 5000, "summary": "concise"}
        result = LiteLLMAnthropicMessagesAdapter.translate_thinking_for_model(
            thinking=thinking,
            model="openai/gpt-5.2",
        )
        assert result == {"reasoning_effort": {"effort": "high", "summary": "concise"}}

    def test_translate_thinking_for_model_disabled_stays_plain_string_when_auto_summary_enabled(self):
        """Disabled thinking must stay a plain string even when reasoning_auto_summary is on."""
        import litellm
        from litellm.llms.anthropic.pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )

        original = litellm.reasoning_auto_summary
        try:
            litellm.reasoning_auto_summary = True
            thinking = {"type": "disabled"}
            result = LiteLLMAnthropicMessagesAdapter.translate_thinking_for_model(
                thinking=thinking,
                model="openai/gpt-5.2",
            )
        finally:
            litellm.reasoning_auto_summary = original

        assert result == {"reasoning_effort": "none"}


# ---------------------------------------------------------------------------
# Parity tests: redundant empty-text-block sanitization scan removal.
# The async wrapper sanitizes once and tells the handler to skip its second
# (redundant) full-messages scan; the sync entry point still sanitizes.
# ---------------------------------------------------------------------------


def _empty_block_msgs():
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "   "},  # whitespace-only -> stripped
                {"type": "tool_use", "id": "t", "name": "B", "input": {}},
            ],
        }
    ]


def test_handler_strips_when_no_presanitized_flag():
    """Sync entry point (no async wrapper): handler must still sanitize."""
    from litellm.llms.anthropic.pass_through.messages import handler, utils

    with patch.object(
        utils,
        "strip_empty_content_blocks_from_anthropic_messages",
        wraps=utils.strip_empty_content_blocks_from_anthropic_messages,
    ) as spy:
        result = handler.anthropic_messages_handler(
            max_tokens=10,
            messages=_empty_block_msgs(),
            model="anthropic/claude-3-5-sonnet-20241022",
            custom_llm_provider="anthropic",
            mock_response="hi there",
        )
    assert spy.call_count == 1  # sanitized exactly once here
    assert result is not None


def test_handler_skips_strip_when_presanitized():
    """Async wrapper already sanitized -> handler must NOT rescan."""
    from litellm.llms.anthropic.pass_through.messages import handler, utils

    with patch.object(
        utils,
        "strip_empty_content_blocks_from_anthropic_messages",
        wraps=utils.strip_empty_content_blocks_from_anthropic_messages,
    ) as spy:
        result = handler.anthropic_messages_handler(
            max_tokens=10,
            messages=_empty_block_msgs(),
            model="anthropic/claude-3-5-sonnet-20241022",
            custom_llm_provider="anthropic",
            mock_response="hi there",
            _litellm_messages_presanitized=True,
        )
    assert spy.call_count == 0  # skipped the redundant scan
    assert result is not None


def test_handler_flattens_replayed_unencrypted_web_search_results():
    """Synthesized search blocks replayed as history must reach the provider as text."""
    from litellm.llms.anthropic.pass_through.messages import handler

    captured = {}

    def fake_base_handler(*args, **kwargs):
        captured.update(kwargs)
        return "stub"

    with patch.object(
        handler.base_llm_http_handler,
        "anthropic_messages_handler",
        side_effect=fake_base_handler,
    ):
        handler.anthropic_messages_handler(
            max_tokens=10,
            messages=[
                {"role": "user", "content": "latest litellm version?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "server_tool_use",
                            "id": "srvtoolu_1",
                            "name": "web_search",
                            "input": {"query": "latest litellm version"},
                        },
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": "srvtoolu_1",
                            "content": [
                                {
                                    "type": "web_search_result",
                                    "url": "https://github.com/BerriAI/litellm/releases",
                                    "title": "Releases",
                                    "page_age": None,
                                    "encrypted_content": "",
                                    "snippet": "Latest release v1.95.0",
                                }
                            ],
                        },
                    ],
                },
                {"role": "user", "content": "which version?"},
            ],
            model="anthropic/claude-3-5-sonnet-20241022",
            custom_llm_provider="anthropic",
        )

    replayed = captured["messages"][1]["content"]
    assert [b["type"] for b in replayed] == ["text"]
    assert "Snippet: Latest release v1.95.0" in replayed[0]["text"]


def test_presanitized_flag_not_leaked_to_provider_params():
    """The private sentinel must be popped, never forwarded as a request param."""
    from litellm.llms.anthropic.pass_through.messages import handler

    captured = {}

    def fake_base_handler(*args, **kwargs):
        captured.update(kwargs)
        captured["optional"] = kwargs.get("anthropic_messages_optional_request_params", {})
        return "stub"

    with patch.object(
        handler.base_llm_http_handler,
        "anthropic_messages_handler",
        side_effect=fake_base_handler,
    ):
        handler.anthropic_messages_handler(
            max_tokens=10,
            messages=[{"role": "user", "content": "hi"}],
            model="anthropic/claude-3-5-sonnet-20241022",
            custom_llm_provider="anthropic",
            _litellm_messages_presanitized=True,
        )

    assert "_litellm_messages_presanitized" not in captured.get("optional", {})
    assert "_litellm_messages_presanitized" not in captured.get("kwargs", {})


@pytest.mark.asyncio
async def test_async_wrapper_sets_presanitized_and_sanitizes_once():
    """End-to-end: wrapper sanitizes (once) AND signals the handler to skip."""
    from litellm.llms.anthropic.pass_through.messages import handler, utils

    captured = {}

    def fake_handler(*args, **kwargs):
        captured["messages"] = kwargs.get("messages")
        captured["presanitized"] = kwargs.get("_litellm_messages_presanitized")
        return "stub"

    fake_loop = MagicMock()
    fake_loop.run_in_executor = lambda _e, func: _async_return(func())

    with (
        patch.object(handler, "anthropic_messages_handler", side_effect=fake_handler),
        patch("asyncio.get_event_loop", return_value=fake_loop),
        patch.object(
            utils,
            "strip_empty_content_blocks_from_anthropic_messages",
            wraps=utils.strip_empty_content_blocks_from_anthropic_messages,
        ) as spy,
    ):
        await handler.anthropic_messages(
            max_tokens=100,
            messages=_empty_block_msgs(),
            model="anthropic/claude-sonnet-4-5-20250929",
            custom_llm_provider="anthropic",
            api_key="k",
        )

    # Wrapper stripped exactly once (the handler is faked, so its skipped
    # call never runs anyway -- the point is the wrapper still sanitizes).
    assert spy.call_count == 1
    assert captured["presanitized"] is True
    assert [b["type"] for b in captured["messages"][0]["content"]] == ["tool_use"]


def _gate_stubs(monkeypatch):
    """Patch the gate's downstream dispatch targets so config selection can be
    observed without making a network call.

    Returns ``(captured, translation_calls)`` where ``captured["config"]`` is the
    provider config handed to the native passthrough path and ``translation_calls``
    counts hits on the Anthropic->OpenAI translation handlers.
    """
    from litellm.llms.anthropic.pass_through.messages import handler

    captured = {}
    translation_calls = {"count": 0}

    def fake_native(**kwargs):
        captured["config"] = kwargs.get("anthropic_messages_provider_config")
        return "native-passthrough"

    def fake_translation(**kwargs):
        translation_calls["count"] += 1
        return "translated"

    monkeypatch.setattr(handler.base_llm_http_handler, "anthropic_messages_handler", fake_native)
    monkeypatch.setattr(
        handler.LiteLLMMessagesToResponsesAPIHandler,
        "anthropic_messages_handler",
        staticmethod(fake_translation),
    )
    monkeypatch.setattr(
        handler.LiteLLMMessagesToCompletionTransformationHandler,
        "anthropic_messages_handler",
        staticmethod(fake_translation),
    )
    return captured, translation_calls


def test_gate_passthrough_when_supported_endpoints_opts_in(monkeypatch):
    """provider=openai + model_info.supported_endpoints containing /v1/messages
    must route to the native passthrough config, NOT the translation handlers."""
    from litellm.llms.anthropic.pass_through.messages.handler import (
        anthropic_messages_handler,
    )
    from litellm.llms.openai_like.messages.transformation import (
        OpenAILikeAnthropicMessagesConfig,
    )

    captured, translation_calls = _gate_stubs(monkeypatch)

    result = anthropic_messages_handler(
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello"}],
        model="openai/some-model",
        api_key="sk-test",
        api_base="https://host/v1",
        model_info={"supported_endpoints": ["/v1/chat/completions", "/v1/messages"]},
    )

    assert result == "native-passthrough"
    assert isinstance(captured["config"], OpenAILikeAnthropicMessagesConfig)
    assert translation_calls["count"] == 0


def test_gate_translates_when_supported_endpoints_absent(monkeypatch):
    """Default behavior is unchanged: without the /v1/messages opt-in, an openai
    deployment is translated (Responses API), never passed through natively."""
    from litellm.llms.anthropic.pass_through.messages.handler import (
        anthropic_messages_handler,
    )

    captured, translation_calls = _gate_stubs(monkeypatch)

    result = anthropic_messages_handler(
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello"}],
        model="openai/some-model",
        api_key="sk-test",
        api_base="https://host/v1",
    )

    assert result == "translated"
    assert translation_calls["count"] == 1
    assert "config" not in captured


def test_gate_passthrough_skipped_when_only_chat_completions_supported(monkeypatch):
    """A deployment that lists only /v1/chat/completions is still translated;
    the opt-in is specifically the /v1/messages entry."""
    from litellm.llms.anthropic.pass_through.messages.handler import (
        anthropic_messages_handler,
    )

    captured, translation_calls = _gate_stubs(monkeypatch)

    result = anthropic_messages_handler(
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello"}],
        model="openai/some-model",
        api_key="sk-test",
        api_base="https://host/v1",
        model_info={"supported_endpoints": ["/v1/chat/completions"]},
    )

    assert result == "translated"
    assert translation_calls["count"] == 1
    assert "config" not in captured


@pytest.mark.parametrize(
    "model_info, expected_ttl_support",
    [
        ({"supported_endpoints": ["/v1/messages"]}, False),
        ({"supported_endpoints": ["/v1/messages"], "cache_control_ttl": True}, True),
        ({"supported_endpoints": ["/v1/messages"], "cache_control_ttl": "yes"}, False),
    ],
)
def test_gate_passthrough_forwards_cache_control_ttl_only_when_deployment_opts_in(
    monkeypatch, model_info, expected_ttl_support
):
    """The passthrough config strips cache_control.ttl unless the deployment sets
    model_info.cache_control_ttl to exactly true."""
    from litellm.llms.anthropic.pass_through.messages.handler import (
        anthropic_messages_handler,
    )

    captured, _ = _gate_stubs(monkeypatch)

    result = anthropic_messages_handler(
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello"}],
        model="openai/some-model",
        api_key="sk-test",
        api_base="https://host/v1",
        model_info=model_info,
    )

    assert result == "native-passthrough"
    assert captured["config"].supports_cache_control_ttl() is expected_ttl_support


def test_first_party_claude_4_8_plus_cost_map_entries_carry_mid_conversation_system_flag():
    """Regional and provider-prefixed Claude 4.8+/5 entries carry
    ``supports_mid_conversation_system``, but the bare first-party keys
    (``claude-opus-4-8``) that a plain ``custom_llm_provider="anthropic"``
    lookup resolves were missed, so that lookup reports the capability as
    unset. Every mapped first-party entry the fallback rule matches must
    carry the flag."""
    import json
    import os
    import re

    import litellm

    cost_map_path = os.path.join(os.path.dirname(litellm.__file__), "model_prices_and_context_window_backup.json")
    with open(cost_map_path) as f:
        cost_map = json.load(f)
    rules = cost_map["fallback_generalizations"]["rules"]
    rule_pattern = next(
        (r["pattern"] for r in rules if r["name"] == "claude-mid-conversation-system"),
        None,
    )
    assert rule_pattern is not None, "claude-mid-conversation-system rule not found in fallback_generalizations"
    pattern = re.compile(rule_pattern, re.IGNORECASE)
    missing = [
        key
        for key, info in cost_map.items()
        if isinstance(info, dict)
        and info.get("litellm_provider") == "anthropic"
        and "claude" in key
        and pattern.search(key)
        and info.get("supports_mid_conversation_system") is not True
    ]
    assert missing == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requested_model, expected_wire_model, expected_url",
    [
        (
            "perplexity/perplexity/kimi-k3",
            "perplexity/kimi-k3",
            "https://api.perplexity.ai/v1/responses",
        ),
        (
            "perplexity/perplexity/sonar",
            "perplexity/sonar",
            "https://api.perplexity.ai/v1/responses",
        ),
        ("perplexity/sonar", "sonar", "https://api.perplexity.ai/chat/completions"),
    ],
)
async def test_messages_strips_provider_prefix_exactly_once(requested_model, expected_wire_model, expected_url):
    """
    BerriAI/litellm#37716: only the leading provider segment may be stripped on the way upstream.

    A multi-segment id such as perplexity/perplexity/kimi-k3 must reach the provider as
    perplexity/kimi-k3, matching what /v1/chat/completions and /v1/responses already send.

    The endpoint is asserted alongside the body because perplexity/perplexity/sonar is a
    Responses-only deployment whose bare id perplexity/sonar is an ordinary chat model, so
    stripping the prefix must not also move the request onto chat/completions.

    The subject is the outbound request, so the transport is cut at the wire rather than
    stubbed with a response body: these ids take different bridges (chat completions
    versus the Responses API) and would otherwise need different response shapes.
    """
    captured = {}

    async def fake_send(self, request, **kwargs):
        captured["body"] = json.loads(request.content)
        captured["url"] = str(request.url)
        raise httpx.ConnectError("cut at the wire", request=request)

    with (
        patch.object(httpx.AsyncClient, "send", fake_send),
        pytest.raises(litellm.exceptions.InternalServerError),
    ):
        await litellm.anthropic.messages.acreate(
            max_tokens=100,
            messages=[{"role": "user", "content": "ping"}],
            model=requested_model,
            api_key="test-api-key",
        )

    assert captured["body"]["model"] == expected_wire_model
    assert captured["url"] == expected_url


@pytest.mark.asyncio
async def test_native_messages_strips_replayed_provider_specific_fields_from_wire():
    captured = {}

    async def fake_send(self, request, **kwargs):
        captured["body"] = json.loads(request.content)
        raise httpx.ConnectError("cut at the wire", request=request)

    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                    "provider_specific_fields": {"signature": "sig_abc"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01",
                    "content": "Sunny",
                }
            ],
        },
    ]

    with (
        patch.object(httpx.AsyncClient, "send", fake_send),
        pytest.raises(litellm.exceptions.InternalServerError),
    ):
        await litellm.anthropic.messages.acreate(
            max_tokens=100,
            messages=messages,
            model="anthropic/claude-haiku-4-5-20251001",
            api_key="test-api-key",
        )

    assert "provider_specific_fields" in messages[0]["content"][0]
    assert "provider_specific_fields" not in captured["body"]["messages"][0]["content"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requested_model, expected_reported_model",
    [
        ("perplexity/perplexity/kimi-k3", "perplexity/kimi-k3"),
        ("perplexity/sonar", "sonar"),
    ],
)
async def test_messages_streaming_reports_provider_local_model(requested_model, expected_reported_model):
    """
    BerriAI/litellm#37716: the wire keeps every segment, so ``message_start`` must still
    report the id the provider itself knows rather than the caller's prefixed deployment id.
    """

    class _EmptyStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    with patch("litellm.acompletion", new=AsyncMock(return_value=_EmptyStream())):
        stream = await litellm.anthropic.messages.acreate(
            max_tokens=100,
            messages=[{"role": "user", "content": "ping"}],
            model=requested_model,
            api_key="test-api-key",
            stream=True,
        )
        first_event = await stream.__anext__()

    assert json.loads(first_event.decode().split("data: ", 1)[1])["message"]["model"] == expected_reported_model


def test_messages_sync_streaming_reports_provider_local_model():
    """Same guarantee as the async bridge, at the sync call site."""
    with patch("litellm.completion", new=MagicMock(return_value=iter(()))):
        stream = litellm.anthropic.messages.create(
            max_tokens=100,
            messages=[{"role": "user", "content": "ping"}],
            model="perplexity/perplexity/kimi-k3",
            api_key="test-api-key",
            stream=True,
        )
        first_event = next(iter(stream))

    assert json.loads(first_event.decode().split("data: ", 1)[1])["message"]["model"] == "perplexity/kimi-k3"


_RESPONSES_COMPLETED_BODY: Dict[str, Any] = {
    "id": "resp-1",
    "object": "response",
    "created_at": 0,
    "model": "gpt-4o-mini",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "id": "msg-1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello there"}],
        }
    ],
    "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
}

_RESPONSES_SSE_EVENTS: List[Dict[str, Any]] = [
    {
        "type": "response.created",
        "response": {
            **_RESPONSES_COMPLETED_BODY,
            "status": "in_progress",
            "output": [],
            "usage": None,
        },
    },
    {
        "type": "response.output_text.delta",
        "item_id": "msg-1",
        "output_index": 0,
        "content_index": 0,
        "delta": "hello there",
    },
    {"type": "response.completed", "response": _RESPONSES_COMPLETED_BODY},
]


def _sse_body(events: List[Dict[str, Any]]) -> bytes:
    return b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)


class _SuccessPayloadCapture(CustomLogger):
    def __init__(self, tracking_id: str):
        super().__init__()
        self.tracking_id = tracking_id
        self.payloads: List[Dict[str, Any]] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        if kwargs.get("litellm_call_id") == self.tracking_id:
            self.payloads.append(kwargs.get("standard_logging_object") or {})


@pytest.fixture
def capture_success_payloads(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    capture = _SuccessPayloadCapture(tracking_id=f"messages-stream-{uuid.uuid4()}")
    monkeypatch.setattr(litellm, "callbacks", [capture])
    return capture


async def _drain(sse_stream) -> List[bytes]:
    return [chunk async for chunk in sse_stream]


def _bind_logging_worker_to_running_loop() -> None:
    """The worker's queue keeps the loop it was built on, so one left over from an earlier
    test makes ``flush`` raise "bound to a different event loop". ``start`` runs
    ``_ensure_queue``, which rebinds the queue when the running loop has changed."""
    GLOBAL_LOGGING_WORKER.start()


async def _flush_logging_worker(capture: "_SuccessPayloadCapture") -> None:
    await asyncio.sleep(0)
    try:
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)
    except asyncio.TimeoutError:
        pass
    deadline = asyncio.get_running_loop().time() + 10.0
    while not capture.payloads and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.05)


def _assert_anthropic_sse(chunks: List[bytes]) -> None:
    body = b"".join(chunks).decode()
    assert "message_start" in body
    assert "message_stop" in body


class TestMessagesStreamingSuccessLogging:
    """A streamed /v1/messages call routed to an ``openai/`` backend must still reach
    success logging once the SSE stream is drained. The client gets a correct Anthropic
    SSE body and the provider bills the call, but no StandardLoggingPayload is produced,
    so the request is invisible to spend tracking and every success-logging integration.

    Logging is fired by the inner CustomStreamWrapper / ResponsesAPIStreamingIterator the
    Anthropic wrappers drain, not by the wrappers themselves, so these assert on the
    callback that actually reaches an integration rather than on a wrapper attribute.
    """

    MESSAGES = [{"role": "user", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_responses_bridge_streaming_emits_success_logging(self, capture_success_payloads):
        """The Responses bridge, which is the default for openai/ deployments."""
        from litellm.llms.anthropic.pass_through.responses_adapters.handler import (
            LiteLLMMessagesToResponsesAPIHandler,
        )

        _bind_logging_worker_to_running_loop()

        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = httpx.Response(
                200,
                content=_sse_body(_RESPONSES_SSE_EVENTS),
                headers={"content-type": "text/event-stream"},
            )
            sse_stream = await LiteLLMMessagesToResponsesAPIHandler.async_anthropic_messages_handler(
                max_tokens=100,
                messages=self.MESSAGES,
                model="openai/gpt-4o-mini",
                stream=True,
                custom_llm_provider="openai",
                litellm_call_id=capture_success_payloads.tracking_id,
            )
            chunks = await _drain(sse_stream)

        await _flush_logging_worker(capture_success_payloads)

        _assert_anthropic_sse(chunks)
        assert len(capture_success_payloads.payloads) == 1
        payload = capture_success_payloads.payloads[0]
        assert payload["call_type"] == "aresponses"
        assert payload["prompt_tokens"] == 11
        assert payload["completion_tokens"] == 7
        assert payload["total_tokens"] == 18
        assert payload["response_cost"] > 0

    @pytest.mark.asyncio
    async def test_chat_completions_bridge_streaming_emits_success_logging(self, capture_success_payloads):
        """The chat-completions bridge, reached via
        litellm.use_chat_completions_url_for_anthropic_messages. Its router lookup is
        stubbed to what an SDK caller with no proxy running already resolves to."""
        from litellm.llms.anthropic.pass_through.adapters.handler import (
            LiteLLMMessagesToCompletionTransformationHandler,
        )

        _bind_logging_worker_to_running_loop()

        with patch(
            "litellm.llms.anthropic.pass_through.adapters.handler._proxy_router_fallback",
            return_value=None,
        ):
            sse_stream = await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
                max_tokens=100,
                messages=self.MESSAGES,
                model="openai/gpt-4o-mini",
                stream=True,
                custom_llm_provider="openai",
                mock_response="hello there",
                litellm_call_id=capture_success_payloads.tracking_id,
            )
            chunks = await _drain(sse_stream)

        await _flush_logging_worker(capture_success_payloads)

        _assert_anthropic_sse(chunks)
        assert len(capture_success_payloads.payloads) == 1
        payload = capture_success_payloads.payloads[0]
        assert payload["call_type"] == "acompletion"
        assert payload["total_tokens"] > 0
        assert payload["response_cost"] > 0


class _FailureCapture(CustomLogger):
    def __init__(self):
        super().__init__()
        self.error_information: list[StandardLoggingPayloadErrorInformation] = []

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        payload = kwargs.get("standard_logging_object") or {}
        self.error_information.append(payload.get("error_information") or {})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "upstream_status, upstream_error_type, expected_exception",
    [
        (401, "authentication_error", litellm.AuthenticationError),
        (403, "permission_error", litellm.PermissionDeniedError),
    ],
)
async def test_anthropic_messages_maps_provider_exception_before_failure_logging(
    monkeypatch, upstream_status, upstream_error_type, expected_exception
):
    """Regression test for LIT-6164. The async /v1/messages entrypoint awaited the
    provider handler without exception_type mapping, so the @client failure
    handler (and every logger behind it, e.g. OTel error spans) saw the raw
    BaseLLMException: error.type=BaseLLMException and no llm_provider.

    The 403 row pins the upstream status on the way through the mapper: Anthropic's
    documented permission_error must reach the caller as a 403, never as the mapper's
    APIConnectionError 500 fallthrough."""
    from litellm.llms.anthropic.pass_through.messages import handler

    capture = _FailureCapture()
    monkeypatch.setattr(litellm, "callbacks", [capture])

    def upstream_rejects_the_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            upstream_status,
            json={"type": "error", "error": {"type": upstream_error_type, "message": "rejected upstream"}},
            request=request,
        )

    upstream = AsyncHTTPHandler()
    upstream.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_rejects_the_request))

    with pytest.raises(expected_exception) as excinfo:
        await handler.anthropic_messages(
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            model="anthropic/claude-haiku-4-5",
            custom_llm_provider="anthropic",
            api_key="sk-invalid",
            client=upstream,
        )

    assert excinfo.value.status_code == upstream_status
    assert excinfo.value.llm_provider == "anthropic"
    assert "AnthropicException" in excinfo.value.message
    assert f'"{upstream_error_type}"' in excinfo.value.message

    assert capture.error_information, "the failure handler must have logged the mapped exception"
    error_information = capture.error_information[0]
    assert error_information.get("error_class") == expected_exception.__name__
    assert error_information.get("llm_provider") == "anthropic"
    assert error_information.get("error_code") == str(upstream_status)


@pytest.mark.asyncio
async def test_anthropic_messages_leaves_non_provider_failures_unmapped():
    """The mapping boundary is for provider failures only. A request rejected before
    the provider call (here invalid metadata) must surface as the original exception,
    not as the mapper's APIConnectionError, whose message embeds a server traceback."""
    from litellm.llms.anthropic.pass_through.messages import handler

    def upstream_must_not_be_called(request: httpx.Request) -> httpx.Response:
        raise AssertionError("the provider must not be called for a request rejected locally")

    upstream = AsyncHTTPHandler()
    upstream.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_must_not_be_called))

    with pytest.raises(ValidationError) as excinfo:
        await handler.anthropic_messages(
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            model="anthropic/claude-haiku-4-5",
            custom_llm_provider="anthropic",
            api_key="sk-invalid",
            client=upstream,
            metadata={"user_id": 123},
        )

    assert "Traceback" not in str(excinfo.value)


def _recording_client(seen_urls: list[str]) -> AsyncHTTPHandler:
    def record_and_answer(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "deepseek-chat",
                "content": [{"type": "text", "text": "pong"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        )

    upstream = AsyncHTTPHandler()
    upstream.client = httpx.AsyncClient(transport=httpx.MockTransport(record_and_answer))
    return upstream


@pytest.mark.asyncio
async def test_provider_messages_api_base_env_is_not_shadowed_by_the_chat_default(monkeypatch):
    from litellm.llms.anthropic.pass_through.messages import handler

    monkeypatch.delenv("DEEPSEEK_API_BASE", raising=False)
    monkeypatch.setenv("DEEPSEEK_ANTHROPIC_API_BASE", "https://deepseek.internal.example/anthropic")
    seen_urls: list[str] = []

    await handler.anthropic_messages(
        max_tokens=16,
        messages=[{"role": "user", "content": "ping"}],
        model="deepseek/deepseek-chat",
        api_key="sk-test",
        client=_recording_client(seen_urls),
    )

    assert seen_urls == ["https://deepseek.internal.example/anthropic/v1/messages"]

@pytest.mark.asyncio
async def test_anthropic_messages_forwards_safeguards_and_unknown_beta_to_anthropic():
    """Shapes are what Claude Code 2.1.278 sends and api.anthropic.com returns, captured 2026-09-21."""
    from litellm.llms.anthropic.pass_through.messages import handler

    safeguards = [{"type": "dangerous_tool_use", "classifier_context": {"v": 1, "permission_mode": "auto"}}]
    client_betas = "dangerous-tool-use-2026-09-03,interleaved-thinking-2025-05-14"
    safeguard_results = [{"type": "dangerous_tool_use", "status": {"type": "available", "tool_uses": {}}}]
    captured: dict[str, object] = {}

    def upstream_records_the_request(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["anthropic-beta"] = request.headers.get("anthropic-beta")
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-4-5",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "safeguard_results": safeguard_results,
            },
            request=request,
        )

    upstream = AsyncHTTPHandler()
    upstream.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_records_the_request))

    response = await handler.anthropic_messages(
        max_tokens=16,
        messages=[{"role": "user", "content": "hi"}],
        model="anthropic/claude-haiku-4-5",
        custom_llm_provider="anthropic",
        api_key="sk-test",
        client=upstream,
        safeguards=safeguards,
        extra_headers={"anthropic-beta": client_betas},
    )

    assert captured["body"]["safeguards"] == safeguards
    assert set(captured["anthropic-beta"].split(",")) == set(client_betas.split(","))
    assert response["safeguard_results"] == safeguard_results


@pytest.mark.asyncio
async def test_anthropic_messages_streaming_forwards_safeguards_and_keeps_safeguard_results():
    """Shapes are what Claude Code 2.1.278 sends and api.anthropic.com returns, captured 2026-09-21."""
    from litellm.llms.anthropic.pass_through.messages import handler

    safeguards = [{"type": "dangerous_tool_use", "classifier_context": {"v": 1, "permission_mode": "auto"}}]
    tool_verdicts = {"toolu_01": {"type": "evaluated", "outcome": "not_flagged"}}
    safeguard_results = [{"type": "dangerous_tool_use", "status": {"type": "available", "tool_uses": tool_verdicts}}]
    captured: dict[str, object] = {}
    message_start = {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 0},
            "safeguard_results": safeguard_results,
        },
    }
    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None, "safeguard_results": safeguard_results},
        "usage": {"output_tokens": 1},
    }
    sse = "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
        for event in (message_start, message_delta, {"type": "message_stop"})
    )

    def upstream_streams_safeguard_results(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse.encode(), request=request)

    upstream = AsyncHTTPHandler()
    upstream.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_streams_safeguard_results))

    stream = await handler.anthropic_messages(
        max_tokens=16,
        messages=[{"role": "user", "content": "hi"}],
        model="anthropic/claude-haiku-4-5",
        custom_llm_provider="anthropic",
        api_key="sk-test",
        client=upstream,
        stream=True,
        safeguards=safeguards,
    )
    raw = b"".join([chunk async for chunk in stream]).decode()
    events = [json.loads(line[len("data: ") :]) for line in raw.splitlines() if line.startswith("data: ")]

    assert captured["body"]["safeguards"] == safeguards
    assert events[0]["message"]["safeguard_results"] == safeguard_results
    assert [e for e in events if e["type"] == "message_delta"][0]["delta"]["safeguard_results"] == safeguard_results


def _claude_code_auto_mode_request() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Shapes are what Claude Code 2.1.278 sends and Bedrock Invoke / Vertex rawPredict return, captured 2026-09-21."""
    safeguards = [{"type": "dangerous_tool_use", "classifier_context": {"v": 1, "permission_mode": "auto"}}]
    tool_verdicts = {"toolu_01": {"type": "evaluated", "outcome": "not_flagged"}}
    safeguard_results = [{"type": "dangerous_tool_use", "status": {"type": "available", "tool_uses": tool_verdicts}}]
    return safeguards, safeguard_results


def _upstream_answering_with(safeguard_results: list[dict[str, object]], captured: dict[str, object]) -> AsyncHTTPHandler:
    def upstream_records_the_request(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["anthropic-beta"] = request.headers.get("anthropic-beta")
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "safeguard_results": safeguard_results,
            },
            request=request,
        )

    upstream = AsyncHTTPHandler()
    upstream.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_records_the_request))
    return upstream


_CLIENT_BETA_HEADERS: Final = (
    pytest.param({"anthropic-beta": "dangerous-tool-use-2026-09-03,interleaved-thinking-2025-05-14"}, id="client_sends_beta"),
    pytest.param({"anthropic-beta": "interleaved-thinking-2025-05-14"}, id="client_omits_beta"),
    pytest.param({}, id="client_sends_no_beta_header"),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("client_headers", _CLIENT_BETA_HEADERS)
async def test_anthropic_messages_forwards_safeguards_and_dangerous_tool_use_beta_to_bedrock_invoke(
    local_beta_headers_config, client_headers
):
    """Bedrock Invoke takes betas in the body's `anthropic_beta` and 400s on `safeguards` without the beta, so the beta rides along with the field."""
    from litellm.llms.anthropic.pass_through.messages import handler

    safeguards, safeguard_results = _claude_code_auto_mode_request()
    captured: dict[str, object] = {}

    response = await handler.anthropic_messages(
        max_tokens=16,
        messages=[{"role": "user", "content": "hi"}],
        model="bedrock/us.anthropic.claude-sonnet-5",
        custom_llm_provider="bedrock",
        aws_access_key_id="test-access-key",
        aws_secret_access_key="test-secret-key",
        aws_region_name="us-east-1",
        client=_upstream_answering_with(safeguard_results, captured),
        safeguards=safeguards,
        extra_headers=client_headers,
    )

    assert captured["body"]["safeguards"] == safeguards
    assert captured["body"]["anthropic_beta"] == ["dangerous-tool-use-2026-09-03"]
    assert response["safeguard_results"] == safeguard_results


@pytest.mark.asyncio
@pytest.mark.parametrize("client_headers", _CLIENT_BETA_HEADERS)
async def test_anthropic_messages_forwards_safeguards_and_dangerous_tool_use_beta_to_vertex(
    local_beta_headers_config, client_headers
):
    """Vertex rawPredict takes the beta as the `anthropic-beta` header and 400s on `safeguards` without it, so the beta rides along with the field."""
    from litellm.llms.anthropic.pass_through.messages import handler
    from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

    safeguards, safeguard_results = _claude_code_auto_mode_request()
    captured: dict[str, object] = {}

    with patch.object(VertexBase, "_ensure_access_token", return_value=("test-token", "test-project")):
        response = await handler.anthropic_messages(
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            model="vertex_ai/claude-sonnet-5",
            custom_llm_provider="vertex_ai",
            vertex_project="test-project",
            vertex_location="global",
            vertex_credentials="{}",
            client=_upstream_answering_with(safeguard_results, captured),
            safeguards=safeguards,
            extra_headers=client_headers,
        )

    assert captured["body"]["safeguards"] == safeguards
    assert "anthropic_beta" not in captured["body"]
    assert captured["anthropic-beta"].split(",").count("dangerous-tool-use-2026-09-03") == 1
    assert response["safeguard_results"] == safeguard_results


@pytest.mark.asyncio
async def test_anthropic_messages_litellm_router_latency_metadata_tracking():
    """
    Test the anthropic_messages with routing strategy and verify that _latency_per_deployment
    field is passed in litellm_metadata when calling litellm.anthropic_messages
    """
    with unittest.mock.patch("litellm.anthropic_messages") as mock_anthropic_messages:
        # Mock the return value
        mock_response = {
            "id": "msg_123456",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Here's a joke for you!"}],
            "model": "claude-haiku-4-5-20251001",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        mock_anthropic_messages.return_value = mock_response
        # Set the __name__ attribute that the router expects
        mock_anthropic_messages.__name__ = "anthropic_messages"

        MODEL_GROUP = "claude-special-alias"
        router = Router(
            model_list=[
                {
                    "model_name": MODEL_GROUP,
                    "litellm_params": {
                        "model": "claude-haiku-4-5-20251001",
                        "api_key": os.getenv("ANTHROPIC_API_KEY"),
                    },
                }
            ],
            routing_strategy="latency-based-routing",
        )

        # Set up test parameters
        messages = [{"role": "user", "content": "Hello, can you tell me a short joke?"}]

        # Call the handler
        response = await router.aanthropic_messages(
            messages=messages,
            model=MODEL_GROUP,
            max_tokens=100,
            metadata={
                "user_id": "hello",
            },
        )

        # Verify response
        assert response == mock_response

        # Verify that litellm.anthropic_messages was called
        mock_anthropic_messages.assert_called_once()

        # Get the call arguments
        call_args = mock_anthropic_messages.call_args
        call_kwargs = call_args.kwargs

        print("Call kwargs:", json.dumps(call_kwargs, indent=2, default=str))

        # Verify that litellm_metadata was passed and contains _latency_per_deployment
        assert (
            "litellm_metadata" in call_kwargs
        ), "litellm_metadata should be passed to anthropic_messages"

        litellm_metadata = call_kwargs["litellm_metadata"]
        assert litellm_metadata is not None, "litellm_metadata should not be None"
        assert isinstance(
            litellm_metadata, dict
        ), "litellm_metadata should be a dictionary"

        # Verify _latency_per_deployment is present
        assert (
            "_latency_per_deployment" in litellm_metadata
        ), "litellm_metadata should contain _latency_per_deployment field"

        # Verify the structure of _latency_per_deployment
        latency_per_deployment = litellm_metadata["_latency_per_deployment"]
        assert isinstance(
            latency_per_deployment, dict
        ), "_latency_per_deployment should be a dictionary"

        print(f"✅ Latency per deployment data: {latency_per_deployment}")

        # Verify other expected fields in litellm_metadata
        assert "model_group" in litellm_metadata
        assert litellm_metadata["model_group"] == MODEL_GROUP
        assert "deployment" in litellm_metadata
        assert "model_info" in litellm_metadata

        # Verify other call parameters
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
        assert call_kwargs["messages"] == messages
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["metadata"] == {"user_id": "hello"}

        print(
            "✅ Successfully verified that _latency_per_deployment is passed in litellm_metadata to anthropic_messages"
        )

        return response


@pytest.mark.asyncio
async def test_anthropic_messages_with_extra_headers():
    """
    Test the anthropic_messages with extra headers
    """
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY", "fake-api-key")

    # Set up test parameters
    messages = [{"role": "user", "content": "Hello, can you tell me a short joke?"}]
    extra_headers = {
        "anthropic-version": "custom-version-for-test",
    }

    # Create a mock response
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "id": "msg_123456",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Why did the chicken cross the road? To get to the other side!",
            }
        ],
        "model": "claude-haiku-4-5-20251001",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }

    # Create a mock client with AsyncMock for the post method
    mock_client = MagicMock(spec=AsyncHTTPHandler)
    mock_client.post = AsyncMock(return_value=mock_response)

    # Call the handler with extra_headers and our mocked client
    response = await litellm.anthropic.messages.acreate(
        messages=messages,
        api_key=api_key,
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        client=mock_client,
        provider_specific_header={
            "custom_llm_provider": "anthropic",
            "extra_headers": extra_headers,
        },
    )

    # Verify the post method was called with the right parameters
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args.kwargs

    # Verify headers were passed correctly
    headers = call_kwargs.get("headers", {})
    print("HEADERS IN REQUEST", headers)
    for key, value in extra_headers.items():
        assert key in headers
        assert headers[key] == value

    # Verify the response was processed correctly
    assert response == mock_response.json.return_value

    return response


@pytest.mark.asyncio
async def test_anthropic_messages_with_thinking():
    """
    Test the anthropic_messages with thinking
    """
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY", "fake-api-key")

    # Set up test parameters
    messages = [{"role": "user", "content": "Hello, can you tell me a short joke?"}]

    # Create a mock response
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "id": "msg_123456",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Why did the chicken cross the road? To get to the other side!",
            }
        ],
        "model": "claude-haiku-4-5-20251001",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }

    # Create a mock client with AsyncMock for the post method
    mock_client = MagicMock(spec=AsyncHTTPHandler)
    mock_client.post = AsyncMock(return_value=mock_response)

    # Call the handler with extra_headers and our mocked client
    response = await litellm.anthropic.messages.acreate(
        messages=messages,
        api_key=api_key,
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        client=mock_client,
        thinking={"budget_tokens": 100},
    )

    # Verify the post method was called with the right parameters
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args.kwargs
    print("CALL KWARGS", call_kwargs)

    # Verify headers were passed correctly
    request_body = json.loads(call_kwargs.get("data", {}))
    print("REQUEST BODY", request_body)
    assert request_body["max_tokens"] == 100
    assert request_body["model"] == "claude-haiku-4-5-20251001"
    assert request_body["messages"] == messages
    assert request_body["thinking"] == {"budget_tokens": 100}

    # Verify the response was processed correctly
    assert response == mock_response.json.return_value

    return response


@pytest.mark.asyncio
async def test_anthropic_messages_bedrock_credentials_passthrough():
    """
    Test that AWS credentials are correctly passed through to BaseAWSLLM.get_credentials
    when using anthropic.messages.acreate with a bedrock model
    """
    # Mock the get_credentials method
    with unittest.mock.patch.object(
        BaseAWSLLM, "get_credentials"
    ) as mock_get_credentials:
        # Create a proper mock for credentials with the necessary attributes
        mock_credentials = unittest.mock.MagicMock()
        mock_credentials.access_key = "mock_access_key"
        mock_credentials.secret_key = "mock_secret_key"
        mock_credentials.token = "mock_session_token"
        mock_get_credentials.return_value = mock_credentials

        # We also need to mock the actual AWS request signing to avoid real API calls
        with unittest.mock.patch("botocore.auth.SigV4Auth.add_auth"):
            # Set up mock for AsyncHTTPHandler.post to avoid actual API calls
            with unittest.mock.patch(
                "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post"
            ) as mock_post:
                # Configure mock response
                mock_response = unittest.mock.MagicMock()
                mock_response.raise_for_status = unittest.mock.MagicMock()
                mock_response.json.return_value = {
                    "id": "msg_bedrock_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "This is a mock response"}],
                    "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                }
                mock_post.return_value = mock_response

                # Test AWS credentials parameters - separate from function call parameters
                aws_params = {
                    "aws_access_key_id": "test_access_key",
                    "aws_secret_access_key": "test_secret_key",
                    "aws_session_token": "test_session_token",
                    "aws_region_name": "us-west-2",
                    "aws_role_name": "test_role_name",
                    "aws_session_name": "test_session_name",
                    "aws_profile_name": "test_profile",
                    "aws_web_identity_token": "test_web_identity_token",
                    "aws_sts_endpoint": "https://sts.test-region.amazonaws.com",
                }

                # Call the function with AWS credentials
                await litellm.anthropic.messages.acreate(
                    messages=[{"role": "user", "content": "Hello, test credentials"}],
                    model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                    max_tokens=100,
                    **aws_params,
                )

                # Verify get_credentials was called with the correct parameters
                mock_get_credentials.assert_called_once()
                call_args = mock_get_credentials.call_args[1]

                # Assert that our test credentials were passed correctly
                for param_name, param_value in aws_params.items():
                    assert (
                        call_args[param_name] == param_value
                    ), f"Parameter {param_name} was not passed correctly"


@pytest.mark.asyncio
async def test_anthropic_messages_bedrock_dynamic_region():
    """
    Test that when aws_region_name is provided, it is used in request url
    """
    # Mock the HTTP response
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "id": "msg_bedrock_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "This is a mock response"}],
        "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }

    # Create a mock client with AsyncMock for the post method
    mock_client = AsyncMock(spec=AsyncHTTPHandler)
    mock_client.post = AsyncMock(return_value=mock_response)

    # Patch necessary AWS components
    with (
        unittest.mock.patch("botocore.auth.SigV4Auth.add_auth"),
        unittest.mock.patch.object(
            BaseAWSLLM, "get_credentials"
        ) as mock_get_credentials,
    ):

        # Setup mock credentials
        mock_credentials = unittest.mock.MagicMock()
        mock_credentials.access_key = "test_access_key"
        mock_credentials.secret_key = "test_secret_key"
        mock_credentials.token = "test_session_token"
        mock_get_credentials.return_value = mock_credentials

        # Test with specific region
        test_region = "us-east-1"

        # Call anthropic.messages.acreate with aws_region_name
        response = await litellm.anthropic.messages.acreate(
            messages=[{"role": "user", "content": "Hello, test region"}],
            model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            max_tokens=100,
            aws_region_name=test_region,
            client=mock_client,
        )

        # Verify response
        assert response == mock_response.json.return_value

        # Verify the post method was called with the correct URL containing the region
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args

        # Check that the URL contains the correct region
        url = call_args.kwargs.get("url", "")
        assert (
            f"bedrock-runtime.{test_region}.amazonaws.com" in url
        ), f"URL does not contain the correct region. URL: {url}"

        # Verify get_credentials was called with the correct region
        mock_get_credentials.assert_called_once()
        credentials_args = mock_get_credentials.call_args.kwargs
        assert credentials_args.get("aws_region_name") == test_region


@pytest.mark.asyncio
@pytest.mark.parametrize("client_marker", [False, True])
async def test_messages_bridge_keeps_tool_result_cache_control_on_tool_result(monkeypatch, client_marker):
    from litellm.llms.anthropic.pass_through.adapters.handler import (
        LiteLLMMessagesToCompletionTransformationHandler,
    )

    monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
    captured = {}

    async def fake_send(self, request, **kwargs):
        captured["body"] = json.loads(request.content)
        raise httpx.ConnectError("cut at the wire", request=request)

    tool_result: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": "toolu_01",
        "content": [{"type": "text", "text": "file1"}, {"type": "text", "text": "file2"}],
    }
    if client_marker:
        tool_result["cache_control"] = {"type": "ephemeral"}

    with (
        patch.object(httpx.AsyncClient, "send", fake_send),
        pytest.raises(litellm.exceptions.InternalServerError),
    ):
        await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
            max_tokens=8,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "list files"}]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01", "name": "ls", "input": {}}]},
                {"role": "user", "content": [tool_result]},
            ],
            model="anthropic/claude-sonnet-4-5",
            tools=[{"name": "ls", "description": "list", "input_schema": {"type": "object", "properties": {}}}],
            api_key="test-api-key",
        )

    sent_tool_result = captured["body"]["messages"][-1]["content"][0]
    assert sent_tool_result["type"] == "tool_result"
    assert sent_tool_result["cache_control"] == {"type": "ephemeral"}
    assert all("cache_control" not in block for block in sent_tool_result["content"])
