import json
import os
import sys
from unittest.mock import patch

import httpx
import litellm

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.xai.chat.transformation import XAIChatConfig
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    ModelResponse,
    Usage,
)


class TestXAIReasoningTokenFolding:
    """``_fold_reasoning_tokens_into_completion`` re-aligns xAI Usage to the OpenAI invariant."""

    @staticmethod
    def _make_response(
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        reasoning_tokens: int = 0,
    ) -> ModelResponse:
        details = (
            CompletionTokensDetailsWrapper(reasoning_tokens=reasoning_tokens)
            if reasoning_tokens
            else None
        )
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            completion_tokens_details=details,
        )
        response = ModelResponse()
        setattr(response, "usage", usage)
        return response

    def test_should_fold_when_total_explained_by_reasoning_gap(self):
        # xAI live shape: 14 + 10 + 312 == 336.
        response = self._make_response(
            prompt_tokens=14,
            completion_tokens=10,
            total_tokens=336,
            reasoning_tokens=312,
        )

        XAIChatConfig._fold_reasoning_tokens_into_completion(response)

        usage = response.usage
        assert usage.completion_tokens == 322
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

    def test_should_not_fold_when_already_normalised(self):
        response = self._make_response(
            prompt_tokens=14,
            completion_tokens=322,
            total_tokens=336,
            reasoning_tokens=312,
        )

        XAIChatConfig._fold_reasoning_tokens_into_completion(response)

        assert response.usage.completion_tokens == 322

    def test_should_skip_when_no_reasoning_tokens(self):
        response = self._make_response(
            prompt_tokens=14,
            completion_tokens=10,
            total_tokens=24,
            reasoning_tokens=0,
        )

        XAIChatConfig._fold_reasoning_tokens_into_completion(response)

        assert response.usage.completion_tokens == 10

    def test_should_skip_when_gap_does_not_match_reasoning(self):
        # Refuse to fold if xAI accounting changes (gap != reasoning_tokens).
        response = self._make_response(
            prompt_tokens=14,
            completion_tokens=10,
            total_tokens=999,
            reasoning_tokens=312,
        )

        XAIChatConfig._fold_reasoning_tokens_into_completion(response)

        assert response.usage.completion_tokens == 10
        assert response.usage.total_tokens == 999


class TestXAIParallelToolCalls:
    """Test suite for XAI parallel tool calls functionality."""

    def test_get_supported_openai_params_includes_parallel_tool_calls(self):
        """Test that parallel_tool_calls is in supported parameters."""
        config = XAIChatConfig()
        supported_params = config.get_supported_openai_params("xai/grok-4.20")
        assert "parallel_tool_calls" in supported_params

    def test_transform_request_preserves_parallel_tool_calls(self):
        """Test that transform_request preserves parallel_tool_calls parameter."""
        config = XAIChatConfig()

        messages = [{"role": "user", "content": "What's the weather like?"}]
        optional_params = {"parallel_tool_calls": True}

        result = config.transform_request(
            model="xai/grok-4.20",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert result.get("parallel_tool_calls") is True
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"


class TestXAIGrokConversationCache:
    def test_named_conversation_id_is_sent_as_header(self):
        client = HTTPHandler()
        raw_response = httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "grok-4.5",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
            request=httpx.Request("POST", "https://api.x.ai/v1/chat/completions"),
        )

        with patch.object(client, "post", return_value=raw_response) as mock_post:
            litellm.completion(
                model="xai/grok-4.5",
                messages=[{"role": "user", "content": "Hello"}],
                api_key="test-key",
                x_grok_conv_id="conversation-123",
                client=client,
            )

        request_kwargs = mock_post.call_args.kwargs
        assert request_kwargs["headers"]["x-grok-conv-id"] == "conversation-123"
        assert "x_grok_conv_id" not in json.loads(request_kwargs["data"])

    def test_explicit_header_takes_precedence(self):
        config = XAIChatConfig()
        optional_params = {"x_grok_conv_id": "named-parameter-value"}
        headers = {"X-Grok-Conv-Id": "explicit-header-value"}

        result = config.validate_environment(
            headers=headers,
            model="grok-4.5",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params=optional_params,
            litellm_params={},
            api_key="test-key",
        )

        assert result["X-Grok-Conv-Id"] == "explicit-header-value"
        assert "x-grok-conv-id" not in result
        assert "x_grok_conv_id" not in optional_params


class TestXAIUsageNormalization:
    def test_preserves_reasoning_tokens_in_total_usage(self):
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=200)

        XAIChatConfig._normalize_openai_compatible_usage_totals(usage)

        assert usage.total_tokens == 200

    def test_preserves_reasoning_tokens_in_streaming_usage(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 200}

        XAIChatConfig._normalize_openai_compatible_usage_totals(usage)

        assert usage["total_tokens"] == 200
