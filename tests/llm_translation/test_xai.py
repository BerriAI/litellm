import json
import os
from datetime import datetime
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from base_llm_unit_tests import BaseLLMChatTest, BaseReasoningLLMTests

from litellm import Choices, EmbeddingResponse, Message, ModelResponse, Usage, completion
from litellm.llms.xai.chat.transformation import XAI_API_BASE, XAIChatConfig












def test_xai_message_name_filtering():
    messages = [
        {
            "role": "system",
            "content": "*I press the green button*",
            "name": "example_user",
        },
        {"role": "user", "content": "Hello", "name": "John"},
        {"role": "assistant", "content": "Hello", "name": "Jane"},
    ]
    response = completion(
        model="xai/grok-3-mini-beta",
        messages=messages,
    )
    assert response is not None
    assert response.choices[0].message.content is not None


class TestXAIReasoningEffort(BaseReasoningLLMTests):
    def get_base_completion_call_args(self):
        return {
            "model": "xai/grok-4.7",
            "messages": [{"role": "user", "content": "Hello"}],
        }


class TestXAIChat(BaseLLMChatTest):
    def get_base_completion_call_args(self):
        return {
            "model": "xai/grok-3-mini-beta",
        }

    test_web_search = None


def test_xai_streaming_with_include_usage():
    """
    Test that xAI streaming correctly handles usage in the last chunk
    when stream_options={"include_usage": True} is set.

    xAI sends usage in a chunk with empty choices array, which should be
    handled by XAIChatCompletionStreamingHandler.
    """
    try:
        response = completion(
            model="xai/grok-4-1-fast-non-reasoning",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello in one word"},
            ],
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=10,
        )

        chunks = []
        usage_chunk = None

        for chunk in response:
            chunks.append(chunk)
            if hasattr(chunk, "usage") and chunk.usage is not None:
                usage_chunk = chunk

        # Verify we got chunks
        assert len(chunks) > 0, "Should receive streaming chunks"

        # Verify usage was included in one of the chunks
        assert usage_chunk is not None, "Should receive usage in streaming chunks"

        # Verify usage has expected fields
        assert hasattr(
            usage_chunk.usage, "prompt_tokens"
        ), "Usage should have prompt_tokens"
        assert hasattr(
            usage_chunk.usage, "completion_tokens"
        ), "Usage should have completion_tokens"
        assert hasattr(
            usage_chunk.usage, "total_tokens"
        ), "Usage should have total_tokens"

        # Verify usage values are positive
        assert usage_chunk.usage.prompt_tokens > 0, "prompt_tokens should be positive"
        assert (
            usage_chunk.usage.completion_tokens > 0
        ), "completion_tokens should be positive"
        assert usage_chunk.usage.total_tokens > 0, "total_tokens should be positive"

        print(f"✓ Successfully received usage in streaming chunk: {usage_chunk.usage}")

    except Exception as e:
        if "API key" in str(e) or "authentication" in str(e).lower():
            pytest.skip(f"Skipping test due to API key issue: {str(e)}")
        raise
